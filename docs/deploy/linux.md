# StudioForge on Linux: installer, systemd, immutable distros

## One-line installer

```bash
curl -fsSL https://raw.githubusercontent.com/LaserLloyd/StudioForge/main/deploy/linux/install.sh | bash
```

Read it first if you like: [`deploy/linux/install.sh`](../../deploy/linux/install.sh). Preview
every action without changing anything: `bash install.sh --dry-run`.

**User-level by default.** Nothing outside your home directory, no `sudo`:

| What | Where |
| --- | --- |
| Program (venv + uv's Python) | `~/.local/share/studioforge-app/` |
| Commands | `~/.local/bin/studioforge`, `studioforge-watchdog` |
| Data (`SF_DATA_DIR`): config, registry, engines, logs | `~/.local/share/studioforge/` |
| systemd user units | `~/.config/systemd/user/studioforge.service`, `studioforge-watchdog.service` |

It installs `uv` if you lack it (official installer, into the app directory), creates a Python 3.12
venv, installs the **latest GitHub release wheel** (checked against the release's `SHA256SUMS.txt`),
writes the two units and starts them. It does **not** install GPU drivers, a CUDA toolkit or the
llama.cpp engine; it checks for `nvidia-smi`, `git`, `cmake`, `nvcc` and tells you what is missing.

| Option | Effect |
| --- | --- |
| `--system` | System-wide under `/opt/studioforge`, links in `/usr/local/bin`, data in `/var/lib/studioforge`, units in `/etc/systemd/system` running as a `studioforge` system user (video/render groups). Needs root. |
| `--uninstall` | Stops and removes units, program and links. **Your data is kept**; the path is printed. |
| `--version TAG` | A specific release instead of the latest. |
| `--from DIR` | Install from a local source tree (what the fresh-install tests use). |
| `--no-service`, `--no-start` | Skip the units / write and enable them but do not start. |
| `--dry-run` | Print, change nothing. |

The installer is safe to re-run: it stops the running units, rebuilds the venv, rewrites the units
and starts them again, and your data directory is untouched. The units carry
`SuccessExitStatus=130 143`, so a clean `systemctl --user stop` does not leave them `failed`;
`--uninstall` also runs `systemctl --user reset-failed` so a unit from an older install does not linger.

Keep the user service running while logged out: `sudo loginctl enable-linger "$USER"`.

After installing: open <http://localhost:8080>, then `studioforge engine --update`. On Linux + NVIDIA
that step is a **source build** needing `git`, `cmake` and a CUDA toolkit whose `nvcc` matches the
driver; if you would rather not install a toolkit on the host, use the container image
([docker.md](docker.md)) or a distrobox below.

Updating: re-run the installer (it replaces the venv, keeps data). Rolling back: re-run with
`--version <older tag>`.

## Immutable distros (Bazzite, Silverblue, Kinoite, SteamOS)

Do not layer packages with `rpm-ostree`. Two supported ways:

**1. The container image** (recommended): [docker.md](docker.md), with Podman, which these distros
ship. `podman run -d --device nvidia.com/gpu=all …` after `sudo nvidia-ctk cdi generate`.

**2. A distrobox** for the build tools, with the user-level installer inside it:

```bash
distrobox create --name studioforge --image docker.io/nvidia/cuda:13.4.3-devel-ubuntu24.04 --nvidia
distrobox enter studioforge
sudo apt-get update && sudo apt-get install -y curl git cmake ca-certificates
curl -fsSL https://raw.githubusercontent.com/LaserLloyd/StudioForge/main/deploy/linux/install.sh | bash -s -- --no-service
studioforge serve --open
```

`--no-service` because a distrobox has no systemd user session of its own. To start it at login on
the host: `distrobox-enter -n studioforge -- studioforge serve` from a host user unit, or generate
one with `distrobox generate-entry`. The data lives in your real home directory, so it survives
deleting the box.

## The unit files

The installer renders them from a template inside the script. The user gateway unit:

```ini
[Service]
Type=simple
Environment=SF_DATA_DIR=%h/.local/share/studioforge
WorkingDirectory=%h/.local/share/studioforge
ExecStart=%h/.local/share/studioforge-app/venv/bin/studioforge serve
Restart=on-failure
TimeoutStopSec=120
KillMode=mixed
NoNewPrivileges=true
PrivateTmp=true
```

and `studioforge-watchdog.service` (`Restart=always`, not bound to the gateway on purpose: it exists
to be up when the gateway is not). They were checked with `systemd-analyze verify`. The older
hand-installed units in [`deploy/`](../../deploy/README.md) assume a git checkout in `~/studioforge`
instead.

## Backing up the data directory

Stop the service first, then copy the directory (WAL checkpoint happens on a clean stop):

```bash
systemctl --user stop studioforge studioforge-watchdog
tar czf studioforge-data.tgz -C ~/.local/share/studioforge .
systemctl --user start studioforge studioforge-watchdog
```

Restore by extracting into an empty data directory with the services stopped.
