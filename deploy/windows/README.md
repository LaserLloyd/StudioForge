# StudioForge for Windows: installer, service and tray (skeleton)

Status: **skeleton, not built, not signed.** `studioforge.iss` declares the
layout; the payload build and the service wrapper are described here so the
owner (or the W9 follow-up) can finish them. Nothing in this folder runs in CI
yet.

## What ships

| Piece | Choice | Why |
| --- | --- | --- |
| Installer | Inno Setup 6 (`studioforge.iss`) | Free, scriptable, per-machine or per-user, silent mode (`/VERYSILENT`) for winget. |
| Payload | embedded CPython 3.12 plus a venv built from the release wheel | No system Python needed. Build step below. |
| Service | **WinSW** wrapper (`winsw/StudioForgeSvc.exe` + `.xml`) running `studioforge serve` | The app is a normal console program with its own supervision (watchdog sidecar, Job Object for llama-server children). A wrapper keeps that unchanged; a pywin32 `ServiceFramework` port would not. |
| Tray | the existing `studioforge tray` (pystray), started at login from the Startup folder | It already shows status, base URL and an "open panel" action; the installer only launches it. |
| Updates | the in-app updater (`update.repo`) and winget | Both consume the GitHub release. |

## The service definition (ship as `winsw/StudioForgeSvc.xml`)

```xml
<service>
  <id>StudioForge</id>
  <name>StudioForge</name>
  <description>GPU-only OpenAI-compatible LLM gateway (llama.cpp).</description>
  <executable>%BASE%\..\venv\Scripts\studioforge.exe</executable>
  <arguments>serve</arguments>
  <env name="SF_DATA_DIR" value="%PROGRAMDATA%\StudioForge" />
  <workingdirectory>%PROGRAMDATA%\StudioForge</workingdirectory>
  <stoptimeout>120 sec</stoptimeout>
  <onfailure action="restart" delay="5 sec" />
  <log mode="roll-by-size"><sizeThreshold>10240</sizeThreshold><keepFiles>5</keepFiles></log>
</service>
```

Decisions the owner must make before this ships:

1. **Service account.** `LocalSystem` (the default) reaches the GPUs but sees none
   of a user's LM Studio library or `%LOCALAPPDATA%`. Either run the service as
   the installing user (WinSW `<serviceaccount>`, needs the password once) or
   keep the tray-started per-user server as the default and make the service
   opt-in. The `.iss` already leaves the `service` task unticked for that reason.
2. **Data directory.** A service uses `%PROGRAMDATA%\StudioForge`; the per-user
   default is `%LOCALAPPDATA%\studioforge`. Supporting both modes needs a
   migration or one shared `SF_DATA_DIR`.
3. **WinSW version and hash pinning** in the payload build (record it in the SBOM).

## Building the payload (CI job still to write)

```powershell
# on windows-latest
uv build --wheel -o dist
uv python install 3.12
uv venv --python 3.12 dist\windows\payload\venv
uv pip install --python dist\windows\payload\venv\Scripts\python.exe (Get-Item dist\studioforge-*.whl)
# then: pinned WinSW into dist\windows\payload\winsw\, plus studioforge-open.cmd and studioforge.ico
iscc /DAppVersion=$env:VERSION deploy\windows\studioforge.iss
```

A venv made this way holds absolute paths to the build machine's Python. Put a
python-build-standalone "install_only" tree inside the payload and create the
venv from it with `--relocatable`, or switch to an embeddable-zip layout, and
verify on a clean VM before trusting either.

## Signing and winget

- Authenticode: the `windows-sign` job in `.github/workflows/containers.yml`
  is **dormant** until the owner supplies a code-signing certificate secret
  (`docs/deploy/release-pipeline.md`). Unsigned installers trigger SmartScreen
  warnings; that is expected until then.
- winget: the manifests in `deploy/winget/` are **prepared, not submitted**. After
  a signed release exists: fill `InstallerUrl` and `InstallerSha256` (from
  `SHA256SUMS.txt`), run `winget validate deploy/winget`, test with
  `winget install --manifest deploy/winget`, then submit to microsoft/winget-pkgs.

## Tested

Not tested on Windows. The Inno script has not been compiled and the WinSW XML
has not been installed. Record results here when that happens; gate G13 lists
Windows as "untested" until a CI runner or VM is available.
