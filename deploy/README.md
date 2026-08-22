# Running StudioForge as a service on Linux

Two **systemd user units**: the gateway and its recovery watchdog. They assume the
layout below; edit the three paths at the top of each unit if yours differs.

| What | Path the units assume |
| --- | --- |
| repo checkout | `~/studioforge` |
| virtualenv | `~/studioforge/.venv` (`uv venv --python 3.12 .venv && uv pip install -e ".[dev]"`) |
| data dir (`SF_DATA_DIR`) | `~/studioforge/data` — config.yaml, registry, engines, logs |

```bash
mkdir -p ~/.config/systemd/user
cp deploy/studioforge.service deploy/studioforge-watchdog.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now studioforge studioforge-watchdog
systemctl --user status studioforge          # or: journalctl --user -u studioforge -f
```

To keep the services running when you are not logged in:

```bash
sudo loginctl enable-linger "$USER"
```

Notes

- These are *user* units on purpose: the process must run as the login user that owns the model
  library, the venv and the GPU device nodes. Do not add `User=`/`Group=` — those are only valid in
  system units and make a user unit fail to load.
- The watchdog is not `BindsTo=` the gateway: it exists to be up when the gateway is not (it can
  restart the gateway, reclaim orphaned `llama-server` processes and answer `/health` on its own port).
- First run: `~/studioforge/.venv/bin/studioforge engine --update` installs the pinned llama.cpp build
  into the data dir, or use the GUI's Setup tab. On Linux + NVIDIA that is a **source build**
  (upstream publishes no Linux CUDA archive): it needs `git`, `cmake` and a CUDA toolkit whose
  `nvcc` matches the driver, takes minutes, and is reused on later installs of the same tag.
  See `docs/SETUP.md`.
- Windows uses the tray app / `StudioForge Autostart.bat` instead; `studioforge tray` refuses to run
  on Linux and points here.
