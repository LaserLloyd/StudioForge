# Configuration

StudioForge reads one file, `config.yaml`, from its [data directory](INSTALL.md#data-directory).
[`config.example.yaml`](../config.example.yaml) lists every key the app understands with its shipped
default and a comment. This page names the keys most people change.

Ways to change a key:

- The panel: **Advanced** has a page for each area (serving and network, GPUs and placement, engines,
  watchdog, config). Keys that need a restart say so.
- From another machine: `sfctl config get [key]` and `sfctl config set key=value` (needs an `admin` key).
- By hand, then `studioforge config check`, which validates the file exactly as `studioforge serve` will
  load it, names every bad key, and writes nothing. If `serve` finds an invalid file it exits with
  code 2 and tells you to run the check.
- First-run environment variables of the form `SF_SERVER__PORT` only seed defaults; `SF_DATA_DIR` and
  `SF_CONFIG` choose where the data directory and the file are.

| Key | Default | Meaning |
| --- | --- | --- |
| `server.host`, `gui.host`, `watchdog.host` | `127.0.0.1` | Bind addresses. A non-loopback bind makes the server refuse to start until an owner account or an admin key exists |
| `server.port`, `gui.port`, `watchdog.port` | `1234`, `8080`, `1235` | Ports of the gateway, the panel and the watchdog |
| `server.tls.mode` | `off` | `off`, `self_signed` or `cert`, for all three listeners |
| `server.behind_proxy` | `false` | Set it when a reverse proxy or `tailscale serve` fronts the server, so a proxied client is never mistaken for this machine |
| `server.allowed_hosts` | `[]` | Extra `Host` header values accepted as this machine, besides loopback names and the hosts of `server.cors_origins` |
| `server.cors_origins` | `[]` | Browser origins allowed to call the API |
| `server.api_key` | unset | Deprecated single admin key. Prefer `studioforge keys create` |
| `mcp.enabled`, `mcp.path` | `true`, `/mcp` | The MCP endpoint on the gateway |
| `models.dir`, `models.extra_dirs` | unset, `[]` | The GGUF library, indexed in place |
| `models.default_model` | unset | What the model name `local-model` (also `default`, `auto`, `current`) resolves to |
| `models.default_ctx` | `8192` | Context used when a load names none |
| `models.default_ttl_s` | `600` | Idle seconds before a model unloads; `0` means never |
| `models.ttl_by_priority` | `1: 900, 2: 900, 3: 600` | Idle seconds per [priority tier](../README.md#priority-tiers) |
| `models.auto_unload_idle_s` | unset | One idle timer for every model nobody gave a duration; replaces the two keys above |
| `watchdog.wedged_after_failures` | `3` | Consecutive failed probes before the watchdog calls the gateway wedged |
| `leases.vacate_timeout_s` | `180` | How long a lease holder has to release after a vacate request |
| `leases.vacate_retry_after_s` | `15` | Re-ask interval handed to the requester while a holder is vacating |
| `tenants.registry` | `[]` | Declare other GPU programs so they appear with live health ([`TENANTS.md`](TENANTS.md)) |
| `gateway.child_port_start`, `gateway.child_port_end` | `18100`, `18200` | Internal ports for the `llama-server` children |

Setting meanings, ranges and restart requirements are also published machine-readably in
[`contracts/settings-schema.json`](contracts/settings-schema.json).
