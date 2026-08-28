# Releasing

Cutting a release is pushing a tag. `.github/workflows/release.yml` does the rest: it builds the
wheels, the sdist and the source zip from that tag, writes `SHA256SUMS.txt`, and creates the GitHub
release with them attached. Nothing is built from a developer's working tree.

## Versions: one date, three spellings

StudioForge is calendar-versioned — a major, then the release date:

| Where | Spelling | Example |
| --- | --- | --- |
| `src/studioforge/__init__.py` `__version__` | display | `1.26-08-23` |
| `packages/studioforge-companion/src/studioforge_companion/__init__.py` | display | `1.26-08-23` |
| `pyproject.toml` (both) | PEP 440 | `1.26.8.23` |
| the git tag | display, with a `v` | `v1.26-08-23` |

The display string is what `/health`, the GUI footer, the MCP `server_status` tool and the GitHub
User-Agent report. PEP 440 has no way to spell a hyphenated date, so the *package metadata* says
`1.26.8.23` and `pip show studioforge` will show that — deliberately, not a mistake. Nothing in the
app parses `__version__` with `packaging`; the one parser that reads it is
`core/updater.py::_version_key`, which reads both spellings as the same four numbers so the updater
cannot mistake the running build for an update. `tests/unit/test_version.py` pins all four strings
together and fails if one drifts.

## Cutting one

1. Bump the four version strings above (display in the two `__init__.py`, PEP 440 in the two
   `pyproject.toml`) and update the `**Status:**` line in `README.md` and the sample `/health`
   output in `docs/OPENCLAW-SETUP.md`.
2. Run the gates — `just check` (or `make check`): `ruff check`, `ruff format --check`, `mypy` on
   the typed core, and `pytest tests/unit`. Never a bare `pytest`: `tests/contract` starts real
   `llama-server` children on the real GPUs (D23).
3. Commit.
4. Tag it, annotated, and push the tag:

   ```bash
   git tag -a v1.26-08-23 -m "StudioForge 1.26-08-23"
   git push origin main
   git push origin v1.26-08-23
   ```

The workflow triggers on `v*`, builds on `ubuntu-latest`, and attaches four assets:

| Asset | What |
| --- | --- |
| `StudioForge-1.26-08-23.zip` | the source tree at the tag, plus the built wheels and sdist under `dist/` |
| `studioforge-1.26.8.23-py3-none-any.whl` | the server |
| `studioforge_companion-1.26.8.23-py3-none-any.whl` | `sfctl` — installable on a box that has no GPU |
| `SHA256SUMS.txt` | `sha256sum` lines for all of the above |

**Only one archive is attached, on purpose.** `core/updater.py::_parse_release` takes the *first*
asset whose name ends in `.zip`, `.tar.gz` or `.tgz` as the thing to install, and asset order is
upload order. An sdist sitting beside the release zip could therefore be handed to the updater as a
release tree. The sdist is built and hashed, but it ships inside the zip's `dist/` directory rather
than as its own asset.

## Bumping the shipped engine pin

`engine.pinned_tag` ships as `b10425` — the llama.cpp build every number in these docs was measured
against (D2). Moving it is a deliberate release decision, not housekeeping: it changes what a fresh
install downloads on first run, and the flag surface it validates expert settings against.

When you do move it, **regenerate the capabilities snapshot in the same commit**:

```bash
python scripts/refresh_engine_capabilities.py b10549 --dry-run   # see what would change
python scripts/refresh_engine_capabilities.py b10549
```

`src/studioforge/data/engine_capabilities.json` is the architecture / ftype / ggml-type list behind
`studioforge capabilities` and the Setup tab's library summary, and it is a **snapshot taken at one
tag**. The script shallow-clones `ggml-org/llama.cpp` at the tag (`git clone --branch <tag>
--depth 1`, so it needs `git` and network), runs the same `extract_from_checkout` parser the live
report uses — a second parser here would drift — and rewrites the file with `source_tag` set. It
touches exactly that one file and prints a diff summary. `--checkout <path>` reuses a tree you
already have, if you are sure it is at the right tag.

Leaving it stale is not fatal, and since D49-8 it is not a lie either: when `source_tag` is not the
running engine's tag, an unrecognised architecture is reported as *unknown to the architecture list
from `<tag>`* rather than as unsupported by the engine. The cost of not regenerating is therefore a
weaker answer, not a wrong one — but a release that bumps the pin and ships a snapshot from the
previous one has given up a verdict for no reason.

Then update the four places the tag is quoted as an example: `config.example.yaml`, the README
quickstart note, `docs/SETUP.md`'s headless YAML, and D2 in `DECISIONS.md` if the measurements were
redone.

## Building the same assets by hand

```bash
uv build --wheel --sdist -o dist
(cd packages/studioforge-companion && uv build --wheel -o dist)
mv packages/studioforge-companion/dist/*.whl dist/
git archive --format=zip --prefix=StudioForge-1.26-08-23/ \
    -o dist/StudioForge-1.26-08-23.zip v1.26-08-23
python - <<'PY'
import zipfile
from pathlib import Path

dist = Path("dist")
with zipfile.ZipFile(dist / "StudioForge-1.26-08-23.zip", "a", zipfile.ZIP_DEFLATED) as zf:
    for item in sorted(dist.glob("*.whl")) + sorted(dist.glob("*.tar.gz")):
        zf.write(item, f"StudioForge-1.26-08-23/dist/{item.name}")
PY
cd dist && sha256sum *.zip *.whl *.tar.gz > SHA256SUMS.txt
```

`git archive` reads the tag, not the working tree, so an uncommitted edit cannot leak into the zip —
and neither can anything git ignores, which is what keeps `local-env.bat`, `data/` and `.venv/` out
of it. Appending the wheels into `StudioForge-<version>/dist/` gives one download that carries both
the source tree and something `pip install`-able.

## What an installed StudioForge expects to find

Self-update is off until someone sets `update.repo` to the GitHub `owner/name` that publishes these
releases (Setup tab, or `config.yaml`). Unset — or left as the shipped `studioforge/studioforge`
placeholder — the check reports `configured: false` and makes no network call at all, which is why a
build with no public home does not spend a 404 against the anonymous rate limit every 24 hours.

With it set, `GET /repos/<repo>/releases` is the whole discovery mechanism, and the release has to
match on four points:

- **The tag is the version.** `version = tag.lstrip("v")`, and the release directory is named after
  it: `<data_dir>/releases/1.26-08-23/`. `update.channel: stable` (the default) skips anything
  GitHub marks as a prerelease.
- **The zip wraps everything in one top-level directory.** `_extract` flattens *a single* top-level
  directory, so `StudioForge-1.26-08-23/README.md` installs as `releases/1.26-08-23/README.md`.
  Files at the archive root would work too (there is nothing to flatten), but *two* top-level
  directories would not: neither is unwrapped and the whole tree installs a level too deep. A
  GitHub "Source code (zip)" download has the same one-directory shape, so `git archive --prefix=`
  is simply that archive with a name we control.
- **The checksum file is optional but read.** An asset whose name contains `sha256` — or ends in
  `.sha256`, `sums.txt` or `checksums.txt` — is fetched and matched by filename in
  `sha256sum` format. A missing one logs a warning and installs anyway; a *mismatching* one aborts.
- **`/health` must report the tag's version.** After switching the `current.txt` pointer and
  restarting, the updater polls `/health` and requires `status: "ok"` **and** `version` equal to
  `1.26-08-23`. If the display `__version__` and the tag ever disagree, every update rolls itself
  back — the check exists because on Windows a respawned child can lose its port preflight to the
  old process, which then answers the poll with a cheerful `ok`.

Old releases are pruned to the newest `max(2, engine.keep_versions)`, and the previous one is always
kept so a rollback has somewhere to go — `POST /api/update/rollback`, the `rollback_update` MCP
tool, or the watchdog's own rollback when the server is too wedged to answer.

## Licence

**MIT** — see [`LICENSE`](../LICENSE) at the repository root; `pyproject.toml` carries the SPDX
`license = "MIT"` expression. A release zip must include the `LICENSE` file, and the copyright
notice must stay intact in anything packaged or redistributed.
