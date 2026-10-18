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
#   4. exec `studioforge "$@"` so signals reach the real process.
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
    log "data=${DATA_DIR} uid=$(id -u) gid=$(id -g) version=${NEW_VERSION}"
fi

exec studioforge "$@"
