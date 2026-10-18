# Running StudioForge in a container

OCI images: `ghcr.io/laserlloyd/studioforge:<version>-{cuda,rocm,vulkan}` (plus `-cuda-devel`,
see below). One `Dockerfile`, one build argument (`BACKEND`), three runtime bases. Works with
Docker and with Podman.

> **Status of the backends.** StudioForge is GPU-only. NVIDIA/CUDA is the verified path. The
> `rocm` and `vulkan` images carry the ROCm (amd-smi / rocm-smi / rocminfo) and Vulkan
> (vulkaninfo) probes and the backend-aware engine installer from [`ENGINES.md`](../ENGINES.md),
> which were tested with recorded probe output, not on real AMD/Intel hardware yet. Without a
> GPU passed through, every image boots, answers `/health` and reports `can_serve=false`
> (a software Vulkan device such as llvmpipe is refused: STRICT_GPU).

- [Quick start](#quick-start)
- [Which image](#which-image)
- [GPU passthrough](#gpu-passthrough)
- [The engine: first run, and offline](#the-engine-first-run-and-offline)
- [Your data survives updates](#your-data-survives-updates)
- [Backup and restore](#backup-and-restore)
- [Podman notes](#podman-notes)
- [Build it yourself](#build-it-yourself)
- [Troubleshooting](#troubleshooting)

## Quick start

```bash
git clone https://github.com/LaserLloyd/StudioForge.git && cd StudioForge/deploy/compose
cp env.example .env                       # optional; set SF_MODELS to your GGUF folder
docker compose -f compose.yml -f compose.cuda.yml up -d
```

Open <http://localhost:8080> (control panel) and sign in as the owner. The account is created on
the first start of a fresh volume: with `SF_OWNER_PASSWORD` from `.env` if you set it (10+
characters, user `SF_OWNER_USER`, default `owner`), otherwise with a generated password saved
owner-only on the volume -- read it, then change it under **Users & Keys** and delete the file:

```bash
docker compose exec studioforge \
  cat /data/owner-password.initial
```

Why an account at all: the app refuses to listen beyond loopback with nothing to log in with
(`auth_required_for_remote_bind`), and inside a container "beyond loopback" is the only way a
published port can reach it. With the account in place every API request needs a key: create one
per client on the Users & Keys page, or from a shell in the container
(`docker compose exec studioforge bash`, then `studioforge keys create --name openclaw --scope
inference`; the token is printed once). `/health` stays public. The OpenAI-compatible base URL for clients is
`http://localhost:1234/v1`. Ports are published to `127.0.0.1` only; set `SF_BIND=0.0.0.0` to share
on your LAN, and put TLS in front (`server.tls.mode: self_signed`, a reverse proxy, or
`tailscale serve`; see [`SECURITY.md`](../SECURITY.md)).

| Port | What | Env var to remap the host side |
| --- | --- | --- |
| 1234 | Gateway: `/v1`, `/api`, `/mcp`, `/health` | `SF_GATEWAY_PORT` |
| 8080 | Control panel | `SF_PANEL_PORT` |
| 1235 | Watchdog sidecar (MCP + `/health`) | `SF_WATCHDOG_PORT` |

`llama-server` children listen on loopback inside the container only.

Without compose:

```bash
docker run -d --name studioforge --init --gpus all \
  -p 127.0.0.1:1234:1234 -p 127.0.0.1:8080:8080 -p 127.0.0.1:1235:1235 \
  -v studioforge-data:/data -v /path/to/gguf:/models:ro \
  -e SF_MODELS__DIR=/models -e SF_OWNER_PASSWORD=choose-10-plus-characters \
  ghcr.io/laserlloyd/studioforge:2.0.0-cuda
```

`--init` matters: `serve` spawns a watchdog and one `llama-server` per loaded model, and PID 1
has to reap them.

## Which image

| Tag | Base image (found on Docker Hub, 2026-10-18) | When |
| --- | --- | --- |
| `<v>-cuda` | `nvidia/cuda:13.4.3-runtime-ubuntu24.04` | NVIDIA (the default). The first start downloads upstream's prebuilt engine once (no `nvcc` inside) |
| `<v>-cuda-devel` | `nvidia/cuda:13.4.3-devel-ubuntu24.04` + `git cmake build-essential ninja-build` | NVIDIA, when you want the container to **build the engine itself** (a tag upstream has no prebuilt archive for) |
| `<v>-rocm` | `rocm/dev-ubuntu-24.04:7.2.4` | AMD (forward-looking, see the status note) |
| `<v>-vulkan` | `ubuntu:24.04` + `libvulkan1 mesa-vulkan-drivers vulkan-tools` | AMD/Intel via Mesa (forward-looking) |

Override the bases at build time with `--build-arg CUDA_VERSION=… CUDA_FLAVOR=… ROCM_VERSION=…`.
Upstream publishes prebuilt Linux CUDA archives, which the `-cuda` image downloads; the `-cuda-devel`
flavour exists for the cases that need a source build: runtime is ~5 GB, devel is ~14 GB.

Every image: Ubuntu 24.04, Python 3.12 from the distro in a uv-built venv (the builder and runtime
share `/usr/bin/python3.12`, so the venv's interpreter path survives the copy), non-root user
`studioforge` (uid/gid 1000), `SF_DATA_DIR=/data`, `HEALTHCHECK` on `/health`, `STOPSIGNAL SIGTERM`.

## GPU passthrough

| Runtime | Flag / compose file | One-time host setup |
| --- | --- | --- |
| Docker + NVIDIA | `--gpus all` / `compose.cuda.yml` | NVIDIA Container Toolkit (Docker Desktop on Windows: WSL2 GPU support is built in) |
| Podman + NVIDIA | `--device nvidia.com/gpu=all` (CDI) / `compose.cuda.podman.yml` | `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` |
| AMD ROCm | `--device /dev/kfd --device /dev/dri --group-add video --group-add render` (add `--security-opt seccomp=unconfined` if KFD ioctls are blocked) / `compose.rocm.yml` | `amdgpu` driver; use **numeric** gids from `getent group video render` |
| Vulkan | `--device /dev/dri --group-add video --group-add render` / `compose.vulkan.yml` | Mesa on the host |

Check without loading anything: `docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi -L`,
then start the container and read `gpu_count` from `curl localhost:1234/health`.

SELinux hosts (Fedora, Bazzite): the compose files add `:z` to bind mounts; for CDI devices also
`--security-opt label=disable`.

## The engine: first run, and offline

The llama.cpp engine is **not** in the image: it is downloaded or built by the app into
`/data/engines/<tag>/` on first start, so it lives on the data volume and survives updates.

- **Every image, first start of a fresh volume:** the entrypoint runs `studioforge engine --update`
  once: it downloads the newest stable prebuilt archive for the detected GPU backend from
  `api.github.com` / `github.com` (checksum-verified, about 150 MB for CUDA), activates it and pins
  it in `config.yaml`. With no model on the volume yet only `llama-server --version` can be checked
  (the log says so); the first model load is the full test. If the download fails (offline, no GPU
  passed through) the server still starts and the log tells you how to retry:
  `docker compose exec studioforge studioforge engine --update`. Set `SF_ENGINE_BOOTSTRAP=0` to
  skip the download (air-gapped hosts, or when you mount a prepared `engines/`).
- `-cuda-devel`: can also build llama.cpp from source (it has `git`, `cmake` and `nvcc`), which only
  matters for a tag without a prebuilt archive. The slim `-cuda` image cannot build.
- **Offline / air-gapped:** populate the volume while online, then block the network:

  ```bash
  docker run --rm -v studioforge-data:/data ghcr.io/laserlloyd/studioforge:2.0.0-cuda-devel engine --update
  # later, on the offline host, run the slim -cuda image against the same volume
  ```

  or copy a prepared `engines/` tree into the volume (`docker cp` into a stopped container, or
  `tar … -C /data`, see [Backup and restore](#backup-and-restore)). There is currently no setting
  to point at a system `llama-server` on `PATH`.

## Your data survives updates

Everything stateful is on the `studioforge-data` volume mounted at `/data`: `config.yaml`,
`registry.sqlite3` (+ `-wal`/`-shm`), `engines/`, `logs/`, `downloads/`, `huggingface/` (HF cache),
`backups/`. The image's root filesystem is **read-only** in the compose file (`read_only: true`,
`/tmp` and the home directory are tmpfs), so a write that would have been lost with the container
fails loudly instead.

- `docker compose pull && docker compose up -d` (or a rebuild) replaces the container and keeps the
  volume. `docker compose down` keeps it too.
- **`docker compose down -v` and `docker volume rm` destroy your data.** That is the only way to
  lose it by accident.
- The volume must be a real filesystem (named volume or a bind mount on local disk), never a tmpfs:
  SQLite's WAL needs working `fsync`/locking. Avoid NFS/SMB for `/data`.
- `stop_grace_period: 60s` lets `serve` drain, stop every `llama-server` child and close the
  registry cleanly on `docker stop`. A graceful stop exits with status 130; use
  `restart: unless-stopped` (the compose file does), not `on-failure`.
- **Pre-migration snapshot.** On the first start of a *new image version* the entrypoint copies
  `registry.sqlite3` (via SQLite's online-backup API, so WAL-safe) and `config.yaml` to
  `/data/backups/<oldversion>-<UTC timestamp>/` **before** the new version runs its schema
  migrations, then records the new version in `/data/.image-version`. The newest 5 snapshots are
  kept (`SF_PREMIGRATE_KEEP`). If the snapshot fails the container still starts (the marker is not
  updated, so the next start retries) and says so in its log.
- **Bind mounts and uids.** The container runs as uid/gid 1000. A host directory owned by someone
  else is not writable; the entrypoint says so and exits. Fix it with `chown`, run the container as
  the owner (`SF_UID` / `SF_GID` in `.env`), or start it as root once (`user: "0:0"`, `PUID`/`PGID`
  set) so the entrypoint `chown`s the directory and drops privileges, the same pattern as DisPatch.

## Backup and restore

Stop the app first so SQLite's WAL is checkpointed (the container's SIGTERM handling does that),
then archive the volume:

```bash
docker compose stop studioforge
docker run --rm -v studioforge-data:/data:ro -v "$PWD":/backup alpine \
  tar czf /backup/studioforge-data.tgz -C /data .
docker compose start studioforge
```

Restore into a fresh volume:

```bash
docker volume create studioforge-data
docker run --rm -v studioforge-data:/data -v "$PWD":/backup:ro alpine \
  sh -c 'tar xzf /backup/studioforge-data.tgz -C /data && chown -R 1000:1000 /data'
docker compose up -d
```

Copying only `registry.sqlite3` from a running instance can miss the `-wal` file; the snapshot
under `/data/backups/` and the archive above are consistent, a raw copy of a live database may not
be. For Podman, replace `docker` with `podman` (add `:z` to the bind mount on SELinux hosts).

## Podman notes

- Build with `podman build --format docker …`: the default OCI format silently drops `HEALTHCHECK`.
- `podman-compose` reads the same files. Its translation of `deploy.resources.reservations.devices`
  into a CDI device only works when a CDI spec exists on the host; use `compose.cuda.podman.yml`.
- Healthchecks need a systemd user session (`loginctl enable-linger`); without one the container
  stays `starting` and anything `depends_on: service_healthy` waits.
- Rootless + bind-mounted data: see "Bind mounts and uids" above and `--userns=keep-id`.

## Build it yourself

```bash
docker build --build-arg BACKEND=cuda --build-arg VERSION=2.0.0 \
  -t ghcr.io/laserlloyd/studioforge:2.0.0-cuda .
```

This repository does not track `uv.lock` (`.gitignore`), so the image resolves dependencies from
`pyproject.toml` at build time: two builds on different days can differ. Track the lockfile and
switch the two `uv sync` lines to `--locked` for reproducible images.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `/health` says `can_serve=false`, `gpu_count=0` | No GPU passed through (`/api/gpus` lists what each probe tried). Check `docker run --gpus all … nvidia-smi -L`; for ROCm/Vulkan pass `/dev/kfd` / `/dev/dri` and the `video`/`render` gids. |
| Container exits with code 4, log says `auth_required_for_remote_bind` | The volume has no owner account or admin key and the entrypoint could not create one (for example `SF_OWNER_PASSWORD` shorter than 10 characters). Fix the password, or run `docker compose run --rm studioforge users create owner --password-stdin`. |
| `401 invalid_api_key` from a client | Expected once the owner account exists: give each client a key (`studioforge keys create`, or Users & Keys). |
| `engine not ready … No such file or directory: 'git'` in the log | The volume has no engine and the first-start download did not run or failed (`SF_ENGINE_BOOTSTRAP=0`, offline). Run `docker compose exec studioforge studioforge engine --update`, or use `-cuda-devel`, or mount a prepared `engines/`. |
| `cannot build llama.cpp from source: no CUDA compute capability` | No GPU visible when the engine was needed. |
| `… is not writable by uid 1000` | Bind-mounted data dir with the wrong owner; see above. |
| Container `Exited (130)` after `docker stop` | Normal graceful stop. |
| Port 1234/8080 already in use | LM Studio or another StudioForge on the host: remap with `SF_GATEWAY_PORT` / `SF_PANEL_PORT`. |
