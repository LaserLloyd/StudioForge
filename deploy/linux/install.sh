#!/usr/bin/env bash
#
# StudioForge Linux installer.
#
#   curl -fsSL https://raw.githubusercontent.com/LaserLloyd/StudioForge/main/deploy/linux/install.sh | bash
#   # or, from a checkout:
#   deploy/linux/install.sh [options]
#
# Default is a USER-level install -- nothing outside your home directory, no
# sudo:
#
#   app venv        ~/.local/share/studioforge-app/venv
#   command         ~/.local/bin/studioforge  (and studioforge-watchdog)
#   data directory  ~/.local/share/studioforge        (SF_DATA_DIR; config.yaml,
#                                                      registry, engines, logs)
#   systemd units   ~/.config/systemd/user/studioforge.service
#                   ~/.config/systemd/user/studioforge-watchdog.service
#
# --system installs for every user instead (run as root): /opt/studioforge,
# /usr/local/bin, /var/lib/studioforge, units in /etc/systemd/system running
# as a dedicated `studioforge` system user.
#
# Options
#   --system            system-wide install under /opt (needs root)
#   --uninstall         stop and remove units, venv and links. DATA IS KEPT.
#   --version TAG       release to install (default: the latest GitHub release)
#   --from DIR          install from a local source tree instead of a release
#   --no-service        install the program only; write no systemd units
#   --no-start          write and enable the units, but do not start them
#   --dry-run           print every action, change nothing
#   -h, --help          this text
#
# What this script does NOT do, on purpose: it does not install GPU drivers, a
# CUDA toolkit or a llama.cpp engine. See "Next steps" printed at the end, and
# docs/deploy/linux.md. It also never touches your model library.
#
# Immutable distros (Bazzite, Silverblue, Kinoite, SteamOS): do not layer
# packages. Run the container image instead (docs/deploy/docker.md) or a
# distrobox (docs/deploy/linux.md). This script detects them and says so.

set -euo pipefail

REPO="LaserLloyd/StudioForge"
API="https://api.github.com/repos/${REPO}/releases"

MODE=install        # install | uninstall
SCOPE=user          # user | system
VERSION=""
FROM_DIR=""
DO_SERVICE=1
DO_START=1
DRY=0

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m    warning: %s\033[0m\n' "$*" >&2; }
die()  { printf '\n\033[31minstall.sh: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    if [ -r "$0" ] && head -c 2 "$0" | grep -q '^#!'; then
        sed -n '3,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    else
        echo "usage: install.sh [--system] [--uninstall] [--version TAG] [--from DIR] [--no-service] [--no-start] [--dry-run]"
        echo "(run the script from a checkout, or read it on GitHub, for the full text)"
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --system)     SCOPE=system ;;
        --uninstall)  MODE=uninstall ;;
        --version)    [ $# -ge 2 ] || die "--version needs a value"; VERSION="$2"; shift ;;
        --from)       [ $# -ge 2 ] || die "--from needs a directory"; FROM_DIR="$2"; shift ;;
        --no-service) DO_SERVICE=0 ;;
        --no-start)   DO_START=0 ;;
        --dry-run)    DRY=1 ;;
        -h|--help)    usage; exit 0 ;;
        *)            die "unknown option: $1 (try --help)" ;;
    esac
    shift
done

# Run (or, under --dry-run, merely print) one command.
run() {
    if [ "$DRY" = 1 ]; then
        printf '    [dry-run] %s\n' "$*"
    else
        "$@"
    fi
}

# ------------------------------------------------------------------ layout
if [ "$SCOPE" = system ]; then
    APP_DIR=/opt/studioforge
    BIN_DIR=/usr/local/bin
    DATA_DIR=/var/lib/studioforge
    UNIT_DIR=/etc/systemd/system
    SYSTEMCTL=(systemctl)
else
    APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/studioforge-app"
    BIN_DIR="$HOME/.local/bin"
    DATA_DIR="${SF_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/studioforge}"
    UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
    SYSTEMCTL=(systemctl --user)
fi
VENV="$APP_DIR/venv"
UV_BIN_DIR="$APP_DIR/uv"

render_gateway_unit() {
    if [ "$SCOPE" = system ]; then
        cat <<EOF
[Unit]
Description=StudioForge - GPU-only OpenAI-compatible LLM gateway
Documentation=https://github.com/${REPO}/blob/main/docs/deploy/linux.md
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=studioforge
Group=studioforge
SupplementaryGroups=video render
Environment=SF_DATA_DIR=${DATA_DIR}
Environment=PYTHONUNBUFFERED=1
WorkingDirectory=${DATA_DIR}
ExecStart=${VENV}/bin/studioforge serve
Restart=on-failure
RestartSec=5
TimeoutStopSec=120
KillMode=mixed
KillSignal=SIGTERM
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
    else
        cat <<EOF
[Unit]
Description=StudioForge - GPU-only OpenAI-compatible LLM gateway
Documentation=https://github.com/${REPO}/blob/main/docs/deploy/linux.md
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# A user unit: it runs as the login user that owns the model library and the GPU
# device nodes. No User=/Group= here -- those are invalid in user units.
Environment=SF_DATA_DIR=${DATA_DIR}
Environment=PYTHONUNBUFFERED=1
WorkingDirectory=${DATA_DIR}
ExecStart=${VENV}/bin/studioforge serve
Restart=on-failure
RestartSec=5
TimeoutStopSec=120
KillMode=mixed
KillSignal=SIGTERM
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
EOF
    fi
}

render_watchdog_unit() {
    if [ "$SCOPE" = system ]; then
        local user_lines=$'User=studioforge\nGroup=studioforge'
        local wanted=multi-user.target
    else
        local user_lines=""
        local wanted=default.target
    fi
    cat <<EOF
[Unit]
Description=StudioForge recovery watchdog (sidecar)
Documentation=https://github.com/${REPO}/blob/main/docs/deploy/linux.md
After=network-online.target
Wants=network-online.target
# Deliberately NOT bound to studioforge.service: the watchdog exists to be up
# when the gateway is not.

[Service]
Type=simple
${user_lines}
Environment=SF_DATA_DIR=${DATA_DIR}
Environment=PYTHONUNBUFFERED=1
WorkingDirectory=${DATA_DIR}
ExecStart=${VENV}/bin/python -m studioforge.watchdog --config ${DATA_DIR}/config.yaml
Restart=always
RestartSec=3
TimeoutStopSec=20
NoNewPrivileges=true

[Install]
WantedBy=${wanted}
EOF
}

have_systemd() {
    command -v systemctl >/dev/null 2>&1 || return 1
    if [ "$SCOPE" = system ]; then
        [ -d /run/systemd/system ]
    else
        # A user manager must be reachable (not the case in plain containers,
        # `su` shells without a session, or WSL without systemd).
        "${SYSTEMCTL[@]}" show-environment >/dev/null 2>&1
    fi
}

write_file() {  # write_file PATH MODE  (content on stdin)
    local path="$1" mode="$2"
    if [ "$DRY" = 1 ]; then
        printf '    [dry-run] write %s (mode %s)\n' "$path" "$mode"
        cat >/dev/null
    else
        cat >"$path"
        chmod "$mode" "$path"
    fi
}

# --------------------------------------------------------------- preflight
[ "$(uname -s)" = Linux ] || die "this installer is for Linux (macOS is not supported yet; Windows has its own installer)."
case "$(uname -m)" in
    x86_64|aarch64|arm64) ;;
    *) die "unsupported CPU architecture $(uname -m) (x86_64 and aarch64 only)." ;;
esac

if [ "$SCOPE" = system ] && [ "$(id -u)" -ne 0 ] && [ "$DRY" = 0 ]; then
    die "--system needs root. Re-run with sudo, or drop --system for a user-level install."
fi
if [ "$SCOPE" = user ] && [ "$(id -u)" -eq 0 ] && [ "$DRY" = 0 ]; then
    warn "running a user-level install as root puts everything under /root. Did you mean --system?"
fi

# --------------------------------------------------------------- uninstall
if [ "$MODE" = uninstall ]; then
    say "Uninstalling StudioForge ($SCOPE)"
    if [ "$DO_SERVICE" = 1 ] && have_systemd; then
        for u in studioforge studioforge-watchdog; do
            run "${SYSTEMCTL[@]}" disable --now "$u.service" 2>/dev/null || true
            run rm -f "$UNIT_DIR/$u.service"
        done
        run "${SYSTEMCTL[@]}" daemon-reload
    else
        for u in studioforge studioforge-watchdog; do run rm -f "$UNIT_DIR/$u.service"; done
    fi
    run rm -f "$BIN_DIR/studioforge" "$BIN_DIR/studioforge-watchdog"
    run rm -rf "$APP_DIR"
    info "removed the program, its links and its units."
    info "KEPT your data (config, registry, engines, logs): $DATA_DIR"
    info "delete it yourself if you want it gone:  rm -rf '$DATA_DIR'"
    exit 0
fi

# ----------------------------------------------------------------- install
say "StudioForge installer ($SCOPE-level$([ "$DRY" = 1 ] && echo ', dry run'))"

if [ -e /run/ostree-booted ]; then
    warn "this is an image-based (ostree) system. Do not layer packages for StudioForge:"
    warn "use the container image or a distrobox -- see docs/deploy/linux.md."
    warn "A user-level install below still works if the prerequisites are in your toolbox."
fi

command -v curl >/dev/null 2>&1 || [ -n "$FROM_DIR" ] || [ "$DRY" = 1 ] \
    || die "curl is required to download uv and the release. Install curl and re-run."
info "app dir : $VENV"
info "data dir: $DATA_DIR"
info "command : $BIN_DIR/studioforge"

# 1. uv -----------------------------------------------------------------
say "uv (Python toolchain)"
if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
    info "found $UV ($(uv --version))"
elif [ -x "$UV_BIN_DIR/uv" ]; then
    UV="$UV_BIN_DIR/uv"
    info "found $UV"
else
    info "installing uv into $UV_BIN_DIR (official installer, https://astral.sh/uv)"
    run mkdir -p "$UV_BIN_DIR"
    if [ "$DRY" = 1 ]; then
        printf '    [dry-run] curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL=%s sh\n' "$UV_BIN_DIR"
    else
        curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="$UV_BIN_DIR" sh
    fi
    UV="$UV_BIN_DIR/uv"
fi

# 2. what to install -----------------------------------------------------
say "Resolving the package"
WORK=""
if [ -n "$FROM_DIR" ]; then
    [ -f "$FROM_DIR/pyproject.toml" ] || die "--from: $FROM_DIR has no pyproject.toml"
    TARGET="$(cd "$FROM_DIR" && pwd)"
    info "local source tree: $TARGET"
else
    if [ "$DRY" = 1 ]; then
        printf '    [dry-run] resolve %s/%s, download the wheel and verify it against SHA256SUMS.txt\n' \
            "$API" "${VERSION:+tags/$VERSION}${VERSION:-latest}"
        TARGET="<release wheel>"
    else
        WORK="$(mktemp -d)"
        trap 'rm -rf "$WORK"' EXIT
        url="$API/${VERSION:+tags/}${VERSION:-latest}"
        curl -fsSL -H 'Accept: application/vnd.github+json' "$url" -o "$WORK/release.json" \
            || die "could not read $url (offline, rate limited, or no such release). Use --from DIR with a source tree."
        # python3 (any version) parses the JSON and prints "tag|wheel-url|sums-url".
        IFS='|' read -r TAG WHEEL_URL SUMS_URL < <(python3 - "$WORK/release.json" <<'PY'
import json, sys
rel = json.load(open(sys.argv[1], encoding="utf-8"))
assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
wheel = next((u for n, u in assets.items() if n.startswith("studioforge-") and n.endswith(".whl")), "")
print(rel.get("tag_name", ""), wheel, assets.get("SHA256SUMS.txt", ""), sep="|")
PY
        ) || die "release metadata could not be parsed (is python3 installed?)"
        [ -n "$WHEEL_URL" ] || die "release ${TAG:-?} has no studioforge wheel asset. Use --from DIR with a source tree."
        info "release $TAG"
        WHEEL="$WORK/$(basename "$WHEEL_URL")"
        curl -fsSL "$WHEEL_URL" -o "$WHEEL"
        if [ -n "$SUMS_URL" ]; then
            curl -fsSL "$SUMS_URL" -o "$WORK/SHA256SUMS.txt"
            expected="$(awk -v f="$(basename "$WHEEL")" '$2==f || $2=="*" f {print $1}' "$WORK/SHA256SUMS.txt")"
            actual="$(sha256sum "$WHEEL" | awk '{print $1}')"
            [ -n "$expected" ] || die "SHA256SUMS.txt has no entry for $(basename "$WHEEL")"
            [ "$expected" = "$actual" ] || die "checksum mismatch for $(basename "$WHEEL"): expected $expected, got $actual"
            info "sha256 verified"
        else
            warn "the release publishes no SHA256SUMS.txt; the wheel is NOT verified."
        fi
        TARGET="$WHEEL"
    fi
fi

# 3. venv + install ------------------------------------------------------
say "Installing into $VENV"
run mkdir -p "$APP_DIR"
# Keep uv's managed Python INSIDE the app dir: in --system mode the default
# (~root/.local/share/uv) is unreadable to the `studioforge` service account, and
# uninstalling would otherwise leave a copy of Python behind.
export UV_PYTHON_INSTALL_DIR="$APP_DIR/python"
run "$UV" venv --python 3.12 --clear "$VENV"
if [ "$DRY" = 1 ]; then
    printf '    [dry-run] %s pip install --python %s/bin/python %s\n' "$UV" "$VENV" "$TARGET"
else
    # Non-editable, so a --from checkout can be deleted afterwards.
    "$UV" pip install --python "$VENV/bin/python" "$TARGET"
fi
run mkdir -p "$BIN_DIR"
run ln -sf "$VENV/bin/studioforge" "$BIN_DIR/studioforge"
run ln -sf "$VENV/bin/studioforge-watchdog" "$BIN_DIR/studioforge-watchdog"

# 4. data dir + (system) service account ----------------------------------
say "Data directory"
if [ "$SCOPE" = system ]; then
    if ! id studioforge >/dev/null 2>&1; then
        run useradd --system --home-dir "$DATA_DIR" --no-create-home --shell /usr/sbin/nologin studioforge
        for g in video render; do
            getent group "$g" >/dev/null 2>&1 && run usermod -aG "$g" studioforge
        done
    fi
    run mkdir -p "$DATA_DIR"
    run chown -R studioforge:studioforge "$DATA_DIR"
else
    run mkdir -p "$DATA_DIR"
fi
run chmod 0750 "$DATA_DIR"
info "$DATA_DIR"

# 5. units -----------------------------------------------------------------
SERVICE_NOTE=""
if [ "$DO_SERVICE" = 1 ]; then
    say "systemd units"
    run mkdir -p "$UNIT_DIR"
    render_gateway_unit  | write_file "$UNIT_DIR/studioforge.service" 644
    render_watchdog_unit | write_file "$UNIT_DIR/studioforge-watchdog.service" 644
    info "wrote $UNIT_DIR/studioforge.service and studioforge-watchdog.service"
    if have_systemd || [ "$DRY" = 1 ]; then
        run "${SYSTEMCTL[@]}" daemon-reload
        if [ "$DO_START" = 1 ]; then
            run "${SYSTEMCTL[@]}" enable studioforge.service studioforge-watchdog.service
            # restart (not just start): on a re-run the old venv's code is still loaded.
            run "${SYSTEMCTL[@]}" restart studioforge.service studioforge-watchdog.service
            info "enabled and (re)started"
        else
            run "${SYSTEMCTL[@]}" enable studioforge.service studioforge-watchdog.service
            info "enabled (not started: --no-start)"
        fi
        if [ "$SCOPE" = user ]; then
            info "keep it running when you are logged out:  sudo loginctl enable-linger \"\$USER\""
        fi
    else
        SERVICE_NOTE="no systemd session is reachable here; start it by hand: $BIN_DIR/studioforge serve"
        warn "$SERVICE_NOTE"
    fi
fi

# 6. PATH + prerequisites --------------------------------------------------
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) warn "$BIN_DIR is not on your PATH. Add:  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

say "Prerequisite check (informational)"
missing=0
if command -v nvidia-smi >/dev/null 2>&1; then
    info "NVIDIA driver: ok ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1))"
    for tool in git cmake nvcc; do
        if command -v "$tool" >/dev/null 2>&1; then
            info "$tool: ok"
        else
            info "$tool: MISSING"; missing=1
        fi
    done
elif [ -e /dev/kfd ]; then
    info "AMD ROCm device (/dev/kfd) found."
else
    warn "no GPU driver detected (nvidia-smi / /dev/kfd). StudioForge is GPU-only: it will start"
    warn "but /health will report can_serve=false until a driver is installed."
fi

cat <<EOF

Next steps
  1. Open the control panel:   http://localhost:8080   (gateway: http://localhost:1234/v1)
  2. Install an engine:        studioforge engine --update
EOF
if [ "$missing" = 1 ]; then
    cat <<EOF
     (on Linux + NVIDIA the engine is a SOURCE BUILD: it needs git, cmake and a CUDA
      toolkit whose nvcc matches your driver. Install those first, or use the container
      image / distrobox, which bring their own: docs/deploy/linux.md)
EOF
fi
cat <<EOF
  3. Point it at your GGUF library from the panel's Setup tab, or:  studioforge scan
  Uninstall (keeps your data):  $0 --uninstall$([ "$SCOPE" = system ] && echo ' --system')
EOF
[ "$DRY" = 1 ] && printf '\n(dry run: nothing was changed)\n'
exit 0
