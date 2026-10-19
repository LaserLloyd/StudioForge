#!/bin/sh
# StudioForge container entrypoint.
#
#   1. (only when started as root, e.g. `user: "0:0"`) fix ownership of the data
#      volume and drop to PUID:PGID -- the same pattern DisPatch's entrypoint
#      uses, so a bind mount owned by the host user keeps working across updates;
#   2. (only for `serve`) check the data dir is writable, fail LOUD if not;
#   3. (only for `serve`) when the image version differs from the one that last
#      wrote this data dir, snapshot the databases and config.yaml to
#      $SF_DATA_DIR/backups/<oldversion>-<UTC timestamp>/ BEFORE the new version
#      runs its schema migrations. Keeps the newest 5 such snapshots.
#   4. (only for `serve`) create the owner account on a fresh volume, from
#      SF_OWNER_PASSWORD or a generated password saved owner-only on the volume,
#      because the container binds 0.0.0.0 and the app refuses a network bind
#      with nothing to log in with (auth_required_for_remote_bind).
#   5. exec `studioforge "$@"` so signals reach the real process.
#
# Everything else (`--help`, `version`, `config`, `engine ...`) goes straight to
# the program untouched.
set -eu

DATA_DIR="${SF_DATA_DIR:-/data}"
PUID="${PUID:-1000}"
PGID="${PGID:-1000}"
KEEP="${SF_PREMIGRATE_KEEP:-5}"
NEW_VERSION="${SF_IMAGE_VERSION:-unknown}"

log() { printf '%s sf-entrypoint: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >&2; }

# --- 1. started as root: own the data dir, then drop ---------------------------
if [ "$(id -u)" = "0" ] && [ "${1:-}" = "serve" ]; then
    log "started as root; chown ${DATA_DIR} to ${PUID}:${PGID} and drop privileges"
    # Make the image's `studioforge` account BE PUID:PGID so setpriv can resolve
    # the uid (--init-groups needs a passwd entry); -o allows a non-unique id.
    if [ "$(id -u studioforge)" != "$PUID" ] || [ "$(id -g studioforge)" != "$PGID" ]; then
        groupmod -o -g "$PGID" studioforge 2>/dev/null || true
        usermod  -o -u "$PUID" -g "$PGID" studioforge 2>/dev/null || true
    fi
    mkdir -p "$DATA_DIR"
    chown "$PUID:$PGID" "$DATA_DIR"
    if [ -n "$(find "$DATA_DIR" -not -user "$PUID" -print -quit 2>/dev/null)" ]; then
        log "files not owned by ${PUID} found: one-time recursive chown (may take a while on a big volume)"
        chown -R "$PUID:$PGID" "$DATA_DIR"
    fi
    # docker sets HOME=/root for --user 0:0; the dropped user cannot read it.
    APP_HOME="$(getent passwd studioforge | cut -d: -f6)"
    chown "$PUID:$PGID" "$APP_HOME" 2>/dev/null || true
    export HOME="$APP_HOME"
    exec setpriv --reuid="$PUID" --regid="$PGID" --init-groups --inh-caps=-all -- "$0" "$@"
fi

if [ "${1:-}" = "serve" ]; then
    # --- 2. writability --------------------------------------------------------
    mkdir -p "$DATA_DIR" 2>/dev/null || true
    if [ ! -w "$DATA_DIR" ]; then
        log "-----------------------------------------------------------------"
        log " ${DATA_DIR} is not writable by uid $(id -u):$(id -g)"
        log "   dir owned by: $(stat -c '%u:%g' "$DATA_DIR" 2>/dev/null || echo unknown)"
        log " Bind-mounted host directory? Either chown it to that uid, or run the"
        log " container as the owner (compose: SF_UID / SF_GID in .env), or start it"
        log " as root once (user: \"0:0\") and set PUID / PGID so this script chowns."
        log " A named volume needs none of this."
        log "-----------------------------------------------------------------"
        exit 1
    fi

    # --- 3. pre-migration snapshot --------------------------------------------
    MARKER="$DATA_DIR/.image-version"
    OLD_VERSION="$(cat "$MARKER" 2>/dev/null || true)"
    HAS_DATA=0
    for f in "$DATA_DIR"/*.sqlite3; do   # a seeded config.yaml alone is not "data to protect"
        [ -e "$f" ] && HAS_DATA=1
    done
    if [ "$HAS_DATA" = "1" ] && [ "$OLD_VERSION" != "$NEW_VERSION" ]; then
        log "image version ${OLD_VERSION:-<unrecorded>} -> ${NEW_VERSION}: snapshotting before migrations"
        if python - "$DATA_DIR" "${OLD_VERSION:-unrecorded}" "$KEEP" <<'PY'
import glob, os, shutil, sqlite3, sys, time

data, old, keep = sys.argv[1], sys.argv[2], int(sys.argv[3])
root = os.path.join(data, "backups")
stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
dest = os.path.join(root, f"{old}-{stamp}")
os.makedirs(dest, exist_ok=False)
copied = []
for db in sorted(glob.glob(os.path.join(data, "*.sqlite3"))):
    out = os.path.join(dest, os.path.basename(db))
    src = sqlite3.connect(db, timeout=30)
    try:
        dst = sqlite3.connect(out)
        try:
            src.backup(dst)  # WAL-safe, consistent online copy
        finally:
            dst.close()
    finally:
        src.close()
    copied.append(os.path.basename(db))
cfg = os.path.join(data, "config.yaml")
if os.path.exists(cfg):
    shutil.copy2(cfg, dest)
    copied.append("config.yaml")
open(os.path.join(dest, ".premigrate"), "w").write(f"from {old}\n")
print(f"sf-entrypoint: wrote {dest}: {', '.join(copied)}", file=sys.stderr)

# Keep the newest `keep` snapshots made by this script (marker file), by mtime.
snaps = sorted(
    (d for d in glob.glob(os.path.join(root, "*"))
     if os.path.isfile(os.path.join(d, ".premigrate"))),
    key=os.path.getmtime, reverse=True)
for old_dir in snaps[keep:]:
    shutil.rmtree(old_dir, ignore_errors=True)
    print(f"sf-entrypoint: pruned {old_dir}", file=sys.stderr)
PY
        then
            printf '%s\n' "$NEW_VERSION" > "$MARKER"
        else
            log "WARNING: the pre-migration snapshot FAILED. Starting anyway; the marker is"
            log "not updated, so the next start tries again. Check free space on the volume."
        fi
    elif [ "$HAS_DATA" = "0" ]; then
        printf '%s\n' "$NEW_VERSION" > "$MARKER"
    fi

    # --- 4. owner account (the bind rule) --------------------------------------
    # The image binds 0.0.0.0 inside the container (see the Dockerfile), and a
    # non-loopback bind refuses to start without an owner account or an admin
    # key (auth_required_for_remote_bind, exit 4). A fresh volume has neither,
    # so create the owner account once: from SF_OWNER_PASSWORD when the user set
    # it in .env, else from a generated password written to the volume as an
    # owner-only file (never to this log). Nothing happens when an account, a
    # key or the legacy server.api_key already exists.
    # The app's own rule decides (an owner account, an active admin key or the
    # legacy server.api_key all satisfy it). The answer goes to a file because
    # the first CLI call on a fresh volume also logs migrations to stdout.
    FLAG=/tmp/sf-has-admin-credential
    # A restarted container keeps /tmp (unless it is a tmpfs): never let a "no"
    # from an earlier boot decide this one if the check below fails.
    rm -f "$FLAG"
    python - "$FLAG" <<'PY' || log "WARNING: could not evaluate the credential posture; assuming one exists"
import sys

from studioforge.api.auth import has_admin_credential
from studioforge.cli_commands import _config, _store

config = _config(None)
db, store = _store(config)
try:
    found = has_admin_credential(config, store)
finally:
    db.close()
open(sys.argv[1], "w").write("yes" if found else "no")
PY
    if [ "$(cat "$FLAG" 2>/dev/null || echo yes)" = "no" ]; then
        OWNER_USER="${SF_OWNER_USER:-owner}"
        if [ -n "${SF_OWNER_PASSWORD:-}" ]; then
            if studioforge users create "$OWNER_USER" --password-env SF_OWNER_PASSWORD; then
                log "created the owner account '${OWNER_USER}' from SF_OWNER_PASSWORD (sign in on the panel)"
            else
                log "ERROR: could not create the owner account from SF_OWNER_PASSWORD (10+ characters?)"
                exit 1
            fi
        else
            PW_FILE="$DATA_DIR/owner-password.initial"
            SF_OWNER_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(18))')"
            export SF_OWNER_PASSWORD
            ( umask 077 && printf '%s\n' "$SF_OWNER_PASSWORD" > "$PW_FILE" )
            if studioforge users create "$OWNER_USER" --password-env SF_OWNER_PASSWORD; then
                log "created the owner account '${OWNER_USER}' with a generated password, stored"
                log "owner-only at ${PW_FILE}. Read it with:"
                log "    docker compose exec studioforge cat ${PW_FILE}"
                log "then sign in on the panel, change it under Advanced > Accounts & keys, and delete the file."
            else
                rm -f "$PW_FILE"
                log "ERROR: could not create the owner account; the server will refuse to start."
                exit 1
            fi
            unset SF_OWNER_PASSWORD
        fi
    fi
    log "data=${DATA_DIR} uid=$(id -u) gid=$(id -g) version=${NEW_VERSION}"

    # --- 5. first run: the llama.cpp engine ------------------------------------
    # The engine lives on the data volume, not in the image. On a fresh volume
    # `serve` alone would try to BUILD the pinned engine from source, which the
    # slim -cuda image cannot do (no git, no compiler: "No such file or
    # directory: 'git'" and a server that boots but cannot load a model). Upstream
    # ships a prebuilt Linux archive, so fetch the newest stable one once. It is
    # checksum-verified and smoke-tested by the app. A failure (offline, no GPU
    # visible) is not fatal: the server still starts and the log says how to
    # retry. SF_ENGINE_BOOTSTRAP=0 turns this off (air-gapped: mount engines/).
    if [ "${SF_ENGINE_BOOTSTRAP:-1}" != "0" ]        && ! ls "$DATA_DIR"/engines/*/llama-server >/dev/null 2>&1; then
        log "no llama.cpp engine on this volume yet: installing the newest stable prebuilt one"
        log "(needs network access to github.com once; SF_ENGINE_BOOTSTRAP=0 skips this)"
        if studioforge engine --update; then
            log "engine installed and active"
        else
            # With no GGUF on the volume yet the app can only run `llama-server --version`
            # (a "partial" smoke test) and reports the install as failed, keeping the
            # pinned source-build tag active. The download is verified and the binary
            # starts, so activate the newest installed tag; the first model load is the
            # real test.
            TAG="$(ls -1 "$DATA_DIR/engines" 2>/dev/null | grep -E '^b[0-9]+$' | sort -V | tail -1)"
            if [ -n "$TAG" ] && [ -x "$DATA_DIR/engines/$TAG/llama-server" ]                && studioforge engine --activate "$TAG"; then
                log "engine ${TAG} installed and active (no model on this volume yet, so only"
                log "its --version check ran; the first model load is the full test)"
            else
                log "WARNING: the engine install failed (offline? no GPU passed through?). The server"
                log "starts but cannot load models until one is installed: docker compose exec"
                log "studioforge studioforge engine --update, or mount a prebuilt engines/ directory."
            fi
        fi
    fi
fi

exec studioforge "$@"
