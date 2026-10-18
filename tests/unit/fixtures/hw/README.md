# Hardware-probe fixtures (SYNTHETIC)

Every file here was written by hand from the documented / commonly observed
output format of the named tool. **None is a capture from real hardware.**
The device names, ids and memory sizes are plausible but invented; they exist
to pin the parsers in `src/studioforge/core/hw_parsers.py`, not to prove that
a given tool version prints exactly this on a given machine.

Open item for real-hardware validation (see `docs/ENGINES.md`): replace each
file with a capture from the actual tool when a machine with that GPU is
available, and keep these as the "tolerance" cases.

| File | Tool / format | Scenario |
| --- | --- | --- |
| `rocm_smi_2gpu.json` | `rocm-smi --showmeminfo vram --showproductname --json` | two discrete cards, driver in `system` |
| `rocm_smi_empty.json` | same | tool ran, no cards |
| `amd_smi_static_discrete.json` / `amd_smi_metric_discrete.json` | `amd-smi static --json` / `metric --json` (list form) | one discrete 24 GB card, gfx1100 |
| `amd_smi_static_strix_halo.json` / `amd_smi_metric_strix_halo.json` | same (`{"gpu_data": [...]}` form) | unified-memory APU, gfx1151 |
| `rocminfo_gfx1100.txt` / `rocminfo_gfx1151.txt` | `rocminfo` | CPU agent + one GPU agent |
| `vulkaninfo_summary_mixed.txt` | `vulkaninfo --summary` | AMD discrete + llvmpipe CPU device |
| `vulkaninfo_full_mixed.txt` | `vulkaninfo` (excerpt) | the same two devices with memory heaps |
| `vulkaninfo_summary_strix_halo.txt` / `vulkaninfo_full_strix_halo.txt` | same | integrated GPU with a large device-local heap |
| `vulkaninfo_single.json` | `vulkaninfo --json` | one device profile |
| `system_profiler_apple_m2_max.json` | `system_profiler SPDisplaysDataType -json` | Apple silicon (no VRAM figure) |
| `system_profiler_intel_mac_amd.json` | same | Intel Mac with a discrete AMD card (`spdisplays_vram`) |
