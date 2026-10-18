# syntax=docker/dockerfile:1.7
# =============================================================================
#  StudioForge -- multi-stage, multi-backend container image
# =============================================================================
#  One Dockerfile, three runtime bases, selected with --build-arg BACKEND=...
#
#    BACKEND=cuda    nvidia/cuda:<CUDA_VERSION>-<CUDA_FLAVOR>-ubuntu24.04
#    BACKEND=rocm    rocm/dev-ubuntu-24.04:<ROCM_VERSION>
#    BACKEND=vulkan  ubuntu:24.04 + the Vulkan loader and Mesa drivers
#
#  Build (docker or podman -- both are exercised in CI and by hand):
#    docker build --build-arg BACKEND=vulkan -t studioforge:vulkan .
#    podman build --build-arg BACKEND=cuda   -t studioforge:cuda   .
#
#  Run (see docs/deploy/docker.md for the GPU flags per runtime):
#    docker run --rm --init --gpus all -p 1234:1234 -p 8080:8080 -p 1235:1235 \
#      -v studioforge-data:/data studioforge:cuda
#
#  Design notes that are load-bearing (do not "simplify" these away):
#
#   * The BUILDER and the RUNTIME both take python3.12 from Ubuntu 24.04's own
#     apt archive, so the interpreter lives at /usr/bin/python3.12 in both. A
#     uv-created venv holds an absolute symlink (bin/python -> /usr/bin/python3.12)
#     and a pyvenv.cfg `home = /usr/bin`; copy it onto a base whose Python lives
#     elsewhere and every command dies with "required file not found". The
#     nvidia/cuda and rocm/dev Ubuntu 24.04 images ship no Python of their own,
#     which is what makes this safe. (Same rule as DisPatch's Dockerfile,
#     https://github.com/astral-sh/uv/issues/7758.)
#
#   * uv is NOT in the runtime image. It builds the venv and stays behind.
#
#   * The llama.cpp ENGINE is not baked in. StudioForge downloads (or, on
#     Linux + NVIDIA, source-builds) its engine into $SF_DATA_DIR/engines on
#     first run, and the engine tag is the user's decision, not the image's.
#     Consequences, per backend:
#       - vulkan / rocm: the first start fetches an upstream prebuilt archive
#         from api.github.com; the data volume keeps it across container
#         recreation. Offline: pre-populate /data/engines (docs/deploy/docker.md).
#       - cuda: upstream publishes NO Linux CUDA archive, so the engine is a
#         source build (git + cmake + nvcc). The slim `runtime` flavour has no
#         nvcc: use `--build-arg CUDA_FLAVOR=devel` (the `-cuda-devel` tag) to
#         build the engine inside the container, or mount an engines/ tree built
#         elsewhere into /data/engines.
#
#   * GPU-only product: with no GPU the process boots and /health answers, but
#     reports can_serve=false. That is the expected outcome of a smoke test on a
#     machine with no passthrough.
# =============================================================================

ARG BACKEND=vulkan
ARG CUDA_VERSION=13.4.3
ARG CUDA_FLAVOR=runtime
ARG ROCM_VERSION=7.2.4
ARG UV_VERSION=0.13.1

# ---------------------------------------------------------------------------
#  uv binary (pinned; a named stage because COPY --from cannot take an ARG
#  expanded after the first FROM unless it is declared above it)
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uvbin

# ---------------------------------------------------------------------------
#  Stage 1: builder -- resolve and install the dependency set + the app
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/studioforge/venv

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uvbin /uv /usr/local/bin/uv

WORKDIR /src

# Dependency layer: manifests only, so ordinary code edits reuse the install.
# README.md and LICENSE are named by pyproject.toml (readme / license-files);
# hatchling refuses to build the project without them.
COPY pyproject.toml README.md LICENSE ./
# NOTE: this repository git-ignores uv.lock (.gitignore:101), so a clean checkout
# has no lockfile and `uv sync --locked` cannot be used: dependencies resolve
# fresh from pyproject.toml's lower bounds at build time. If the project ever
# tracks uv.lock, add it to the COPY above and switch both syncs to --locked.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# Application layer. --no-editable installs the built wheel into the venv, so
# the runtime image does not need /src.
COPY src/ ./src/
# --reinstall-package: guard against uv serving the project's OWN previously built
# wheel from the cache mount (uv only invalidates it when pyproject.toml changes; seen
# in another repo's image). Not reproducible with uv 0.13.1 in this layout (a control
# build without the flag picked up the edit), but it costs one wheel build and removes
# the question. Verified: rebuild after a one-line source change, grep the venv.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable --reinstall-package studioforge

RUN find /opt/studioforge -name '__pycache__' -type d -prune -exec rm -rf {} + \
 && find /opt/studioforge -name '*.pyc' -delete

# ---------------------------------------------------------------------------
#  Stage 2: per-backend bases (only the one BACKEND names is built)
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS base-vulkan
ENV DEBIAN_FRONTEND=noninteractive
# libvulkan1 = the loader; mesa-vulkan-drivers = RADV (AMD), ANV (Intel) and
# llvmpipe (CPU, for a driverless smoke test); vulkan-tools = vulkaninfo for
# `docker exec ... vulkaninfo --summary`. NVIDIA's Vulkan ICD comes from the
# host driver via the NVIDIA container toolkit, not from this image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers vulkan-tools \
 && rm -rf /var/lib/apt/lists/*

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_FLAVOR}-ubuntu24.04 AS base-cuda
ENV DEBIAN_FRONTEND=noninteractive
# The `devel` flavour is the one that can build the llama.cpp engine in place;
# it still lacks git and cmake. Harmless on `runtime` (a few MB), so one path.
ARG CUDA_FLAVOR
RUN if [ "$CUDA_FLAVOR" = "devel" ]; then \
      apt-get update \
   && apt-get install -y --no-install-recommends git cmake build-essential ninja-build \
   && rm -rf /var/lib/apt/lists/*; \
    fi

FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION} AS base-rocm
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
#  Stage 3: runtime
# ---------------------------------------------------------------------------
FROM base-${BACKEND} AS runtime

ARG BACKEND
ARG VERSION=dev
ARG REVISION=unknown
ARG CREATED

LABEL org.opencontainers.image.title="StudioForge (${BACKEND})" \
      org.opencontainers.image.description="GPU-only OpenAI-compatible LLM server over llama.cpp" \
      org.opencontainers.image.source="https://github.com/LaserLloyd/StudioForge" \
      org.opencontainers.image.documentation="https://github.com/LaserLloyd/StudioForge/blob/main/docs/deploy/docker.md" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.created="${CREATED}" \
      com.laserlloyd.studioforge.backend="${BACKEND}"

ENV DEBIAN_FRONTEND=noninteractive

# python3.12: the venv's interpreter (see the note at the top).
# libgomp1 + libstdc++6 + libcurl4t64: what upstream's llama-server archives
# link against. ca-certificates: engine/model downloads. tzdata: TZ= resolves.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 ca-certificates tzdata libgomp1 libstdc++6 libcurl4t64 \
 && rm -rf /var/lib/apt/lists/*

# --- Non-root user ---------------------------------------------------------
# UID/GID 1000: the first normal user on desktop Linux, so a bind-mounted data
# directory owned by whoever ran `docker compose up` is writable without
# ceremony. Ubuntu 24.04 base images pre-create `ubuntu` at 1000; drop it.
# video/render give access to /dev/dri and /dev/kfd for the ROCm and Vulkan
# variants (a no-op group add for cuda).
RUN (userdel -r ubuntu 2>/dev/null || true) \
 && (getent group 1000 >/dev/null && groupdel "$(getent group 1000 | cut -d: -f1)" || true) \
 && groupadd --gid 1000 studioforge \
 && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin studioforge \
 && (getent group video >/dev/null  && usermod -aG video  studioforge || true) \
 && (getent group render >/dev/null && usermod -aG render studioforge || true)

COPY --from=builder --chown=studioforge:studioforge /opt/studioforge/venv /opt/studioforge/venv
COPY --chmod=0755 deploy/compose/entrypoint.sh /usr/local/bin/sf-entrypoint

ENV PATH="/opt/studioforge/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SF_DATA_DIR=/data \
    SF_IMAGE_VERSION=${VERSION} \
    HF_HOME=/data/huggingface
# Deliberately NO NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES here: the
# nvidia/cuda base already sets both for the cuda variant, and baking them into
# the vulkan/rocm variants makes an NVIDIA container hook try (and, with missing
# driver libs, fail) to inject a GPU into containers that never asked for one.

# No `VOLUME /data` on purpose: an anonymous volume would let a forgotten mount
# silently keep (then lose) state. Compose mounts it explicitly.
RUN install -d -o studioforge -g studioforge -m 0750 /data

WORKDIR /data
USER studioforge

# 1234 OpenAI/Anthropic/LM Studio gateway + /mcp, 8080 control panel,
# 1235 watchdog (MCP + /health). llama-server children bind loopback only.
EXPOSE 1234 8080 1235

# /health is liveness. It answers 200 even with no GPU (can_serve=false), so a
# GPU-less smoke test is healthy; read the body for readiness. start-period is
# generous: first boot creates the registry and probes the engine.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import json,sys,urllib.request as u; r=u.urlopen('http://127.0.0.1:1234/health',timeout=4); sys.exit(0 if r.status==200 and json.load(r).get('status')=='ok' else 1)"]

# SIGTERM -> drain, stop every llama-server child, release the data-dir lock.
STOPSIGNAL SIGTERM

# The entrypoint only acts for `serve` (ownership fix when started as root,
# writability check, and a pre-migration snapshot of registry.sqlite3 +
# config.yaml into /data/backups/ when the image version changed), then
# `exec studioforge "$@"`: signals reach the real process, and `--help` /
# `version` / `engine ...` pass straight through. Run with --init (compose:
# `init: true`) so the watchdog sidecar and llama-server children are reaped.
ENTRYPOINT ["/usr/local/bin/sf-entrypoint"]
CMD ["serve"]
