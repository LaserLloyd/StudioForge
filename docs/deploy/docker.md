# Running StudioForge in a container

OCI images: `ghcr.io/laserlloyd/studioforge:<version>-{cuda,rocm,vulkan}` (plus `-cuda-devel`,
see below). One `Dockerfile`, one build argument (`BACKEND`), three runtime bases. Works with
Docker and with Podman.

> **Status of the backends.** StudioForge is GPU-only and today only drives **NVIDIA/CUDA**
> (`core/gpu.py` probes NVML only). The `rocm` and `vulkan` images build, boot and answer
> `/health`, but report `can_serve=false` until the multi-backend engine work (W1) lands. They
> exist so the packaging is ready when it does. Use `-cuda` for real work.

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

Open <http://localhost:8080> (control panel). The OpenAI-compatible base URL for clients is
`http://localhost:1234/v1`. Ports are published to `127.0.0.1` only; set `SF_BIND=0.0.0.0` to
share on your LAN (and set `server.api_key` first, on the panel's Setup tab).

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
  -e SF_MODELS__DIR=/models \
  ghcr.io/laserlloyd/studioforge:1.26-09-23-cuda
```

`--init` matters: `serve` spawns a watchdog and one `llama-server` per loaded model, and PID 1
has to reap them.

## Which image

| Tag | Base image (found on Docker Hub, 2026-10-18) | When |
| --- | --- | --- |
| `<v>-cuda` | `nvidia/cuda:13.4.3-runtime-ubuntu24.04` | NVIDIA, engine **mounted or pre-built** (no `nvcc` inside) |
| `<v>-cuda-devel` | `nvidia/cuda:13.4.3-devel-ubuntu24.04` + `git cmake build-essential ninja-build` | NVIDIA, let the container **build the engine itself** on first run |
| `<v>-rocm` | `rocm/dev-ubuntu-24.04:7.2.4` | AMD (forward-looking, see the status note) |
| `<v>-vulkan` | `ubuntu:24.04` + `libvulkan1 mesa-vulkan-drivers vulkan-tools` | AMD/Intel via Mesa (forward-looking) |

Override the bases at build time with `--build-arg CUDA_VERSION=… CUDA_FLAVOR=… ROCM_VERSION=…`.
The Linux NVIDIA engine is a **source build** (upstream ships no Linux CUDA archive), which is why
two CUDA flavours exist: runtime is ~5 GB, devel is ~14 GB.

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

- `-vulkan` / `-rocm`: the first start fetches an upstream prebuilt archive from `api.github.com`.
- `-cuda`: needs a source build, which the slim runtime image cannot do. Either use `-cuda-devel`
  (the first start clones llama.cpp and builds it, minutes of CPU), **or** supply a built engine:
  build it once with `-cuda-devel`, then mount the resulting `engines/` directory into a `-cuda`
  container.
- **Offline / air-gapped:** populate the volume while online, then block the network:

  ```bash
  docker run --rm -v studioforge-data:/data ghcr.io/laserlloyd/studioforge:1.26-09-23-cuda-devel engine --update
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
docker build --build-arg BACKEND=cuda --build-arg VERSION=1.26-09-23 \
  -t ghcr.io/laserlloyd/studioforge:1.26-09-23-cuda .
```

This repository does not track `uv.lock` (`.gitignore`), so the image resolves dependencies from
`pyproject.toml` at build time: two builds on different days can differ. Track the lockfile and
switch the two `uv sync` lines to `--locked` for reproducible images.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `/health` says `can_serve=false`, `gpu_count=0` | No GPU passed through, or the image is `-rocm`/`-vulkan` (NVIDIA-only today). Check `docker run --gpus all … nvidia-smi -L`. |
| `no usable llama-server engine … 'git'` | `-cuda` runtime image cannot build the engine. Use `-cuda-devel` or mount a built `engines/`. |
| `cannot build llama.cpp from source: no CUDA compute capability` | No GPU visible when the engine was needed. |
| `… is not writable by uid 1000` | Bind-mounted data dir with the wrong owner; see above. |
| Container `Exited (130)` after `docker stop` | Normal graceful stop. |
| Port 1234/8080 already in use | LM Studio or another StudioForge on the host: remap with `SF_GATEWAY_PORT` / `SF_PANEL_PORT`. |
