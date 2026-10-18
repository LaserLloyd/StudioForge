# Self-hosted GPU runner (setup guide, nothing registered)

The hosted GitHub runners have no GPU, so `containers.yml` can only prove that an image builds, boots
and answers `/health`. Proving that StudioForge actually *sees and uses* a GPU needs a runner you
host. This page is the recipe. **Nothing here has been done:** no runner is registered, and the
`gpu-smoke` job in `containers.yml` is `if: false`.

## Security first

A self-hosted runner executes whatever a workflow tells it to, with access to the machine. On a
public repository that means anyone who can open a pull request. So:

- Use it **only for tag pushes and `workflow_dispatch`** by maintainers. Never for `pull_request`
  from forks. In the repository settings set "Require approval for all outside collaborators".
- Do not run it on a machine that holds data you care about, or that serves other people (the
  household rig that runs the live services is exactly such a machine). A spare box, or a VM with
  GPU passthrough, is the right place.
- Run the runner as an unprivileged user in the `docker` group; prefer an *ephemeral* runner
  (`--ephemeral`) re-registered per job.
- The job must **never load a model** on GPUs another service owns. The `gpu-smoke` job only runs
  `nvidia-smi -L` and reads `gpu_count` from `/health`.

## Host prerequisites (Linux)

1. NVIDIA driver 580+ and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
   `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
2. Docker Engine, and a user (say `ghrunner`) in the `docker` group.
3. Check: `docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi -L`.

## Register the runner

Repository *Settings, Actions, Runners, New self-hosted runner* shows a one-time token and the exact
download for the current runner version. As `ghrunner`:

```bash
mkdir actions-runner && cd actions-runner
# download + extract the tarball the settings page gives you, then:
./config.sh --url https://github.com/LaserLloyd/StudioForge --token <ONE-TIME TOKEN> \
            --labels gpu-cuda --name gpu-box-1 --ephemeral --unattended
sudo ./svc.sh install ghrunner && sudo ./svc.sh start
```

The label `gpu-cuda` is what the workflow selects with `runs-on: [self-hosted, gpu-cuda]`. Add
`gpu-rocm` / `gpu-vulkan` runners the same way for those variants.

## Enable the job

In `.github/workflows/containers.yml`, `gpu-smoke`: replace `if: false` with, for example,
`if: needs.prepare.outputs.push == 'true'` (tags and manual pushes only). Keep `needs: image` so it
tests the image that was just built.

## Remove it

`./config.sh remove --token <REMOVAL TOKEN>` (from the same settings page) and delete the directory.
