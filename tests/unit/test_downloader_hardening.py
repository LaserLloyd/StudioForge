"""Downloader edge cases from the WP13 audit.

* a ``.part`` that already holds every declared byte (a crash between the last
  write and the rename) is verified and published without a request -- it used
  to be truncated by the server's 416 and re-downloaded from zero;
* a verified partial whose rename fails stays on disk and is published on the
  next attempt without a request, with a message naming the likely cause;
* a gated repo's 401/403 says what to do, and is not retried;
* a secondary instance (D24) refuses to start a transfer from the API path;
* two groups sharing one destination (the mmproj every quant of a repo carries)
  take turns instead of one failing on the other's lock;
* a queue that provably overruns the disk is refused up front;
* a persistence hiccup cannot strand a group in ``running`` forever;
* file names Windows would silently rewrite are refused.
"""

from __future__ import annotations

import errno
import hashlib
from pathlib import Path
from typing import Any

import pytest

from studioforge.config import Config
from studioforge.core.downloader import (
    Downloader,
    PublishFailedError,
    TransfersDisabledError,
    _PartFile,
)
from studioforge.core.hf_search import LogicalDownload, safe_filename
from studioforge.db import Database
from studioforge.errors import BadRequestError
from tests.unit.test_downloader import (
    REPO,
    ServerState,
    blob,
    finfo,
    make_downloader,
    one,
    resolve_path,
    wait_group,
)

# The sibling module's fixtures, re-declared here: pytest registers a fixture
# under the module attribute it is bound to, so importing them by another name
# does not make them visible, and importing them by the same name is a
# redefinition to ruff.


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        models={"dir": tmp_path / "models"},
        hf={"max_concurrent_downloads": 4, "chunk_bytes": 16 * 1024},
    )


@pytest.fixture
def db(tmp_path: Path) -> Any:
    database = Database(tmp_path / "registry.sqlite3")
    database.migrate()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def hf_server() -> Any:
    import threading

    from tests.unit.test_downloader import _Handler, _Server

    state = ServerState()
    server = _Server(("127.0.0.1", 0), _Handler)
    server.state = state
    state.endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# complete .part is published without a request
# ---------------------------------------------------------------------------


async def test_a_complete_partial_is_published_without_a_request(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(64 * 1024, seed=21)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    item = one(name, data)

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # The crash window: every byte written, the rename never happened.
    dest.with_name(dest.name + ".part").write_bytes(data)

    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id)
    assert dest.read_bytes() == data
    assert hf_server.statuses_for(path) == [], "no request should have been made"
    assert not dest.with_name(dest.name + ".part").exists()
    await downloader.stop()


async def test_a_complete_partial_with_the_wrong_hash_is_redownloaded_clean(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(64 * 1024, seed=22)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    item = one(name, data)

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Same length, different bytes: a stale partial from another revision.
    dest.with_name(dest.name + ".part").write_bytes(blob(64 * 1024, seed=23))

    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id)
    assert dest.read_bytes() == data
    # A full GET, not a Range request onto the stale bytes.
    assert hf_server.statuses_for(path) == [200]
    await downloader.stop()


# ---------------------------------------------------------------------------
# publish failure
# ---------------------------------------------------------------------------


async def test_a_failed_publish_keeps_the_partial_and_names_the_cause(
    config: Config, db: Database, hf_server: ServerState, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = blob(32 * 1024, seed=24)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    item = one(name, data)

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    downloader.set_in_use_check(lambda p: Path(p) == dest)

    real_publish = _PartFile.publish
    calls = {"n": 0}

    def failing_publish(self: _PartFile, target: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            # Close like the real one, then fail the rename the way Windows
            # does when llama-server has the destination mmapped.
            self._discard = False
            self.close()
            raise PermissionError(errno.EACCES, "used by another process")
        real_publish(self, target)

    monkeypatch.setattr(_PartFile, "publish", failing_publish)

    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id, "failed")
    row = downloader.group(group_id)[0]
    assert "could not be moved into place" in (row.error or "")
    assert "unload it" in (row.error or "").lower()
    part = dest.with_name(dest.name + ".part")
    assert part.exists() and part.stat().st_size == len(data), "the verified partial is kept"
    assert hf_server.statuses_for(path) == [200], "publish failure is not retried as a transfer"

    # Resume: published from the partial, no request.
    downloader.set_in_use_check(None)
    await downloader.resume(group_id)
    await wait_group(downloader, group_id)
    assert dest.read_bytes() == data
    assert hf_server.statuses_for(path) == [200]
    await downloader.stop()


def test_publish_failed_is_not_transient() -> None:
    from studioforge.core.downloader import _is_transient

    assert _is_transient(PublishFailedError("x")) is False


# ---------------------------------------------------------------------------
# gated repo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
async def test_a_gated_repo_download_says_what_to_do_and_is_not_retried(
    config: Config, db: Database, hf_server: ServerState, status: int
) -> None:
    data = blob(8 * 1024, seed=25)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    hf_server.status_script[path] = [status]
    item = one(name, data)

    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id, "failed")
    row = downloader.group(group_id)[0]
    assert "gated or private" in (row.error or "")
    assert "hf.token" in (row.error or "")
    assert f"https://huggingface.co/{REPO}" in (row.error or "")
    assert hf_server.statuses_for(path) == [status], "401/403 must not be retried"
    await downloader.stop()


async def test_a_gated_repo_with_a_token_set_says_the_token_was_refused(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    config.hf.token = "hf_faketokenfortests000000000"
    data = blob(8 * 1024, seed=26)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    hf_server.status_script[path] = [403]
    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id, "failed")
    row = downloader.group(group_id)[0]
    assert "hf.token is set but was refused" in (row.error or "")
    await downloader.stop()


# ---------------------------------------------------------------------------
# secondary instance
# ---------------------------------------------------------------------------


async def test_a_secondary_refuses_to_start_a_transfer(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(8 * 1024, seed=27)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    downloader = make_downloader(config, db, hf_server.endpoint)
    downloader.disable_transfers("another StudioForge instance (pid 4242) owns this data dir")
    with pytest.raises(TransfersDisabledError) as excinfo:
        await downloader.enqueue(one(name, data))
    assert "pid 4242" in excinfo.value.message
    assert excinfo.value.code == "instance_secondary"
    assert downloader.all() == [], "nothing was queued"
    with pytest.raises(TransfersDisabledError):
        await downloader.resume("anything")
    await downloader.stop()


# ---------------------------------------------------------------------------
# two groups sharing one destination
# ---------------------------------------------------------------------------


async def test_two_quants_sharing_one_mmproj_take_turns(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    q4 = blob(40 * 1024, seed=28)
    q8 = blob(40 * 1024, seed=29)
    proj = blob(30 * 1024, seed=30)
    hf_server.files[resolve_path(REPO, "model-Q4_K_M.gguf")] = q4
    hf_server.files[resolve_path(REPO, "model-Q8_0.gguf")] = q8
    hf_server.files[resolve_path(REPO, "mmproj-F16.gguf")] = proj
    hf_server.slice_bytes = 4096
    hf_server.slice_delay_s = 0.005
    mm = finfo("mmproj-F16.gguf", len(proj), sha256=hashlib.sha256(proj).hexdigest())

    def logical(name: str, data: bytes) -> LogicalDownload:
        return LogicalDownload(
            repo_id=REPO,
            quant=name.split("-")[-1].removesuffix(".gguf"),
            files=[finfo(name, len(data), sha256=hashlib.sha256(data).hexdigest())],
            mmproj=mm,
            total_bytes=len(data) + len(proj),
        )

    downloader = make_downloader(config, db, hf_server.endpoint)
    a = await downloader.enqueue(logical("model-Q4_K_M.gguf", q4))
    b = await downloader.enqueue(logical("model-Q8_0.gguf", q8))
    await wait_group(downloader, a)
    await wait_group(downloader, b)
    assert downloader.dest_for(REPO, "mmproj-F16.gguf").read_bytes() == proj
    for gid in (a, b):
        for row in downloader.group(gid):
            assert row.status == "completed", row.to_dict()
    await downloader.stop()


# ---------------------------------------------------------------------------
# disk preflight
# ---------------------------------------------------------------------------


async def test_a_queue_that_overruns_the_disk_is_refused_up_front(
    config: Config, db: Database, hf_server: ServerState, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.core import diskspace

    monkeypatch.setattr(diskspace, "_usage", lambda path: (100 * 2**30, 99 * 2**30, 1 * 2**30))
    diskspace.clear_cache()
    data = blob(8 * 1024, seed=31)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    # Declare it far bigger than the bytes we would serve: 40 GiB.
    info = finfo(name, 40 * 2**30, sha256=hashlib.sha256(data).hexdigest())
    item = LogicalDownload(
        repo_id=REPO, quant=info.quant, files=[info], mmproj=None, total_bytes=40 * 2**30
    )

    downloader = make_downloader(config, db, hf_server.endpoint)
    with pytest.raises(BadRequestError) as excinfo:
        await downloader.enqueue(item)
    assert excinfo.value.code == "insufficient_disk"
    assert "GiB short" in excinfo.value.message
    rows = downloader.all()
    assert rows and rows[0].status == "failed"
    assert "not enough disk space" in (rows[0].error or "")
    assert downloader.active() == [], "nothing was launched"
    await downloader.stop()


async def test_disk_preflight_never_refuses_when_the_report_is_unavailable(
    config: Config, db: Database, hf_server: ServerState, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.core import diskspace

    def boom(path: Path) -> tuple[int, int, int]:
        raise OSError("no such volume")

    monkeypatch.setattr(diskspace, "_usage", boom)
    diskspace.clear_cache()
    data = blob(8 * 1024, seed=32)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id)
    await downloader.stop()


def test_enospc_message_names_the_drive(config: Config, tmp_path: Path) -> None:
    from studioforge.core.downloader import _FileState

    downloader = Downloader(config, db=None)  # type: ignore[arg-type]
    state = _FileState(
        id="g:f",
        group_id="g",
        repo_id=REPO,
        filename="model.gguf",
        dest=tmp_path / "model.gguf",
        status="running",
        total_bytes=10 * 2**30,
        downloaded_bytes=2 * 2**30,
    )
    message = downloader._describe_os_error(state, OSError(errno.ENOSPC, "No space left"))
    assert "disk is full" in message
    assert "GiB free" in message and "still needs 8.0 GiB" in message
    assert "Resume" in message


# ---------------------------------------------------------------------------
# a persistence hiccup cannot strand a group
# ---------------------------------------------------------------------------


async def test_a_failing_status_write_does_not_strand_the_group(
    config: Config, db: Database, hf_server: ServerState, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = blob(16 * 1024, seed=33)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    downloader = make_downloader(config, db, hf_server.endpoint)

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(db, "set_download_status", broken)
    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id)
    assert downloader.dest_for(REPO, name).read_bytes() == data
    assert downloader.active() == []
    await downloader.stop()


# ---------------------------------------------------------------------------
# names Windows would rewrite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["model.gguf.", "model.gguf ", "model.gguf:evil", "..."])
def test_names_windows_would_silently_rewrite_are_refused(name: str) -> None:
    with pytest.raises(BadRequestError):
        safe_filename(name)


def test_ordinary_names_still_pass() -> None:
    assert safe_filename("Qwen2.5-0.5B-Instruct-Q4_K_M.gguf") == "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
    assert safe_filename("mmproj-F16.gguf") == "mmproj-F16.gguf"


# ---------------------------------------------------------------------------
# a projector that fails after the weights landed says what that means
# ---------------------------------------------------------------------------


async def test_a_failed_projector_after_the_weights_landed_is_explained(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    base = blob(24 * 1024, seed=40)
    hf_server.files[resolve_path(REPO, "model-Q4_K_M.gguf")] = base
    # The projector 404s: it is not in the server's file table at all.
    item = LogicalDownload(
        repo_id=REPO,
        quant="Q4_K_M",
        files=[finfo("model-Q4_K_M.gguf", len(base), sha256=hashlib.sha256(base).hexdigest())],
        mmproj=finfo("mmproj-F16.gguf", 4096, sha256=None),
        total_bytes=len(base) + 4096,
    )
    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id, "failed")
    rows = {row.filename: row for row in downloader.group(group_id)}
    assert rows["model-Q4_K_M.gguf"].status == "completed"
    assert rows["mmproj-F16.gguf"].status == "failed"
    assert "text-only" in (rows["mmproj-F16.gguf"].error or "")
    assert "Resume" in (rows["mmproj-F16.gguf"].error or "")
    await downloader.stop()
