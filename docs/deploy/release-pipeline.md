# Release pipeline: images, SBOM, signatures, channels

Status: **skeleton, never run.** `release.yml` (wheels, source zip, GitHub release, checksums) is
unchanged. `containers.yml` is new and sits beside it. Both trigger on a `v*` tag.

## What a tag produces

| Artifact | Made by | Notes |
| --- | --- | --- |
| Wheels, sdist, `StudioForge-<v>.zip`, `SHA256SUMS.txt`, the GitHub release | `release.yml` | Unchanged. The in-app updater and `deploy/linux/install.sh` consume these. |
| `ghcr.io/laserlloyd/studioforge:<v>-{cuda,rocm,vulkan,cuda-devel}` | `containers.yml` / `image` | linux/amd64. Also moves `stable-<variant>` or `beta-<variant>`. |
| `sbom-<variant>.cdx.json` | `anchore/sbom-action` (syft, CycloneDX) | Attached to the GitHub release and, as a signed attestation, to the image. |
| `SHA256SUMS-containers.txt` | `containers.yml` / `checksums` | Hashes of the SBOMs and the digest files. |
| cosign keyless signature | `sigstore/cosign-installer` + `cosign sign` | Signed by the workflow's OIDC identity; no key to manage or leak. |
| Build-provenance attestation | `actions/attest-build-provenance` | Stored in the registry next to the image. |

## Channels

| Tag you push | Channel | Image tags |
| --- | --- | --- |
| `v1.26-09-23` | stable | `1.26-09-23-<variant>`, `stable-<variant>` |
| `v1.26-09-23-beta.2` or `-rc.1` | beta | `1.26-09-23-beta.2-<variant>`, `beta-<variant>` |

A hyphenated calendar version is a stable tag by itself; only a trailing `-beta.N` / `-rc.N` makes
a beta. `workflow_dispatch` with `push=false` builds and smoke-tests without publishing.

## Verifying what you pulled

```bash
IMG=ghcr.io/laserlloyd/studioforge:1.26-09-23-cuda
cosign verify "$IMG" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  --certificate-identity-regexp '^https://github.com/LaserLloyd/StudioForge/.github/workflows/containers.yml@refs/tags/'
gh attestation verify "oci://$IMG" --owner LaserLloyd
cosign verify-attestation "$IMG" --type cyclonedx \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  --certificate-identity-regexp '^https://github.com/LaserLloyd/StudioForge/.github/workflows/containers.yml@refs/tags/'
```

## Dormant signing jobs (owner action to enable)

GitHub does not allow `secrets.*` in an `if:`, so the workflow has a `secrets-present` job that
turns "is the secret set" into an output, and the signing jobs gate on that output. Add the secrets
under *Settings, Secrets and variables, Actions* and the jobs wake up; remove them and they go back
to sleep. **Both jobs also need work that does not exist yet**, and fail loudly if woken early.

| Job | Secrets | Missing before it can work |
| --- | --- | --- |
| `windows-sign` (Authenticode) | `WINDOWS_CERT_PFX_BASE64`, `WINDOWS_CERT_PASSWORD` | The installer build job ([`deploy/windows/README.md`](../../deploy/windows/README.md)). |
| `macos-notarize` (TIER2) | `APPLE_DEVELOPER_ID_CERT_BASE64`, `APPLE_NOTARY_APPLE_ID`, `APPLE_NOTARY_TEAM_ID`, `APPLE_NOTARY_APP_PASSWORD` | A Metal engine and an app bundle. |

Until the Windows secret exists, installers are unsigned and Windows SmartScreen will warn.

## One-time repository setup

1. *Settings, Actions, General*: workflow permissions can stay read-only; the workflow requests what
   it needs per job (`packages: write`, `id-token: write`, `attestations: write`).
2. After the first push, set the GHCR package visibility (public) and link it to the repository.
3. Pin every `uses:` to a commit SHA (Dependabot can keep them current). The skeleton uses the
   latest majors as of 2026-10-18: checkout v7, buildx v4, login v4, build-push v7, cosign-installer v4,
   sbom-action v0, attest-build-provenance v4, upload-artifact v7, download-artifact v8.
4. The `cuda-devel` build needs ~30 GB of free disk; the workflow deletes the preinstalled
   toolchains first. If it still runs out, move that one variant to a larger runner.

## Not covered yet

Windows and macOS binaries, the app-bundle build, multi-arch (arm64) StudioForge images, and the
self-hosted GPU smoke job ([`docs/ci/self-hosted-runner.md`](../ci/self-hosted-runner.md)).
