# Accounts, keys and lockout recovery

How to manage panel logins and API keys from a terminal, and how to get back in when you are locked out. The rules behind them (scopes, bind rule, lockout) are in [`SECURITY.md`](SECURITY.md).

The control panel signs in with an **owner account** (username and password, 10+ characters; the
first account is the owner, later ones are `admin`). Programs use **API keys** instead. Both are
managed from a terminal, and a forgotten password never needs a reinstall.

Every command below prints one line first, on stderr, saying which data directory and database it
is acting on (`studioforge: acting on <dir> (from <source>); database <path>`). With no `--config`
and no `SF_DATA_DIR` the CLI finds the data directory the installed service uses (a running
server's, else the Startup launcher's, the systemd unit's or `local-env.bat`'s), not an empty
`data\` next to the checkout. A launcher is adopted only when it runs the Python environment you
are running now and its data directory is yours; for any other install (a different virtualenv, a
`--system` install owned by the `studioforge` user) the command stops, prints where it is and asks
for `--config <data dir>/config.yaml` (or `SF_DATA_DIR`, or run it as that install's user with `sudo -u <user>`).
If that line shows a directory you did not expect, stop and pass `--config` yourself.

| Task | Command |
| --- | --- |
| List accounts | `studioforge users list` |
| Add an account | `studioforge users add <name> --password-stdin` |
| Change a password (also the way back in when you forgot it) | `studioforge users passwd <name> --password-stdin` |
| Delete an account | `studioforge users delete <name>` (`--yes` skips the question; the last account that can sign in needs `--force`) |
| List, create, revoke API keys | `studioforge keys list` · `keys create --name <label> --scope inference` · `keys revoke <id or name>` |
| Is `config.yaml` loadable? | `studioforge config check` |

The password is read from `--password-stdin` (one line on stdin), `--password-env VAR` (an
environment variable) or a hidden prompt; it is never a command-line argument, so it stays out of
`ps` and shell history. Deleting an account ends its panel sessions and keeps API keys (revoke those
separately); deleting the owner promotes the oldest remaining enabled account. Deleting the last account that can
sign in (with no admin key to fall back on) is refused without `--force`, and even forced a running
network-bound server keeps requiring a credential rather than opening up. Changing a password ends
that account's other panel sessions. Everything else (models, `config.yaml`, engines, logs, the
other accounts and keys) is untouched by all of these.

**Copy-paste, by platform** (stop typing the password into the command: pipe it in)

```powershell
# Windows (PowerShell); from a checkout the command is .\.venv\Scripts\studioforge.exe
$pw = Read-Host -AsSecureString "New password" | ConvertFrom-SecureString -AsPlainText
$pw | .\.venv\Scripts\studioforge.exe users passwd owner --password-stdin
.\.venv\Scripts\studioforge.exe users list
```

```bash
# Linux, and macOS (not a supported platform; the same commands work from a checkout)
read -rs -p "New password: " PW; echo
printf '%s\n' "$PW" | studioforge users passwd owner --password-stdin
studioforge users list
```

The data directory, if you need it by hand: Windows checkout `<repo>\data` or whatever
`SF_DATA_DIR` in `local-env.bat` says (`%LOCALAPPDATA%\studioforge` for a wheel install); Linux
installer `~/.local/share/studioforge` (`$XDG_DATA_HOME/studioforge`) and a checkout's
`<repo>/data`; macOS the same as Linux; Docker `/data` in the container, the named volume
`studioforge-data` on the host. `studioforge config --path` prints the config file whose folder it
is.

## Docker

```bash
# the account commands run inside the container, on /data
docker compose exec studioforge studioforge users list
read -rs -p "New password: " PW; echo
printf '%s\n' "$PW" | docker compose exec -T studioforge studioforge users passwd owner --password-stdin
printf '%s\n' "$PW" | docker compose exec -T studioforge studioforge users add alice --password-stdin
# the container will not start (restart loop) -> run it one-off on the same volume
printf '%s\n' "$PW" | docker compose run --rm -T studioforge users passwd owner --password-stdin
```

## Locked out and the CLI is not an option: the `RESET_LOGIN` file

If you can reach the data directory but not a working shell for StudioForge (a Windows service
account, a container that will not boot, a copied volume), create a file named **`RESET_LOGIN`**
in the data directory and (re)start the server. It is read once, at start, by `studioforge serve`
(the tray and the container start it), applied, and **deleted before anything changes** so it
cannot run again.

| File contents | Result |
| --- | --- |
| *(empty)* | the owner gets a generated password, written to `RESET_LOGIN.password` in the same folder (owner-only, never logged) |
| `alice` | the same for the account `alice` (created if it does not exist) |
| `alice:a-new-long-password` | that password, for `alice` (split at the first colon; 10+ characters; no leading or trailing spaces) |

```powershell
# Windows: stop StudioForge (tray: Quit), then
New-Item -ItemType File "$env:LOCALAPPDATA\studioforge\RESET_LOGIN"      # or <repo>\data\RESET_LOGIN
# start StudioForge, then read and delete the generated password
Get-Content "$env:LOCALAPPDATA\studioforge\RESET_LOGIN.password"
```

```bash
# Linux / macOS
systemctl --user stop studioforge        # or stop however you run it
touch ~/.local/share/studioforge/RESET_LOGIN
systemctl --user start studioforge
cat ~/.local/share/studioforge/RESET_LOGIN.password
```

```bash
# Docker (the volume is named studioforge-data; on Docker Desktop reach it through a helper container)
docker run --rm -v studioforge-data:/data alpine touch /data/RESET_LOGIN
docker compose restart studioforge
docker run --rm -v studioforge-data:/data alpine cat /data/RESET_LOGIN.password
```

Sign in with the new password, change it under **Advanced > Accounts & keys**, and delete
`RESET_LOGIN.password`. The reset changes only that account's password (it re-enables the account
if it was disabled); the other accounts, API keys, models and settings are kept. The start-up log
shows one line, `RESET_LOGIN consumed: ...`, naming the account and never the password. If the file
cannot be deleted (a read-only volume) the reset is refused, not applied. It cannot be triggered
over the network: no HTTP route, panel action or MCP tool reads it, and the server does not watch
for it while running.

## The server will not start because `config.yaml` is invalid

`studioforge serve` exits with code 2, names the key, and says to run `studioforge config check`,
which lists every bad key and the `config.yaml.bak` (the previous save) to copy back. The panel and
`PATCH /api/config` no longer save a combination `serve` would refuse (for example
`cors_allow_credentials: true` with `cors_origins: ["*"]`).

