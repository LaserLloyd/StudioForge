# studioforge-tenant

The tenant side of StudioForge's **VRAM Tenant Protocol (`vt.1`)**, as one small Python package:
lease a GPU before you allocate on it, keep the lease alive while you work, release it when you
stop, announce yourself to the registry, and answer a vacate request the way the protocol expects.
The contract itself is [`docs/TENANTS.md`](../../docs/TENANTS.md) in the StudioForge repository;
this package is the client every tenant would otherwise write by hand.

Pure Python 3.12, **`httpx` only**, never imports `studioforge`. It does not start StudioForge, does
not start anything else, and never sends `force`.

## Install

```bash
# in the StudioForge checkout, on the GPU box
cd packages/studioforge-tenant && uv build --wheel -o dist
# in the tenant's own virtualenv
uv pip install ./dist/studioforge_tenant-*.whl
```

## Use

```python
import asyncio, secrets
from studioforge_tenant import (
    TenantClient,
    Granted,
    Refused,
    Unreachable,
    Unsupported,
    vacate_asgi_route,
)

TOKEN = secrets.token_urlsafe(32)  # minted per process; never logged

client = TenantClient(
    "http://127.0.0.1:1234",  # StudioForge
    "speechforge",  # holder == holder_family == registry name
    priority=2,  # the D46 class of your claims (see TENANTS.md §2)
    vacate_url="http://127.0.0.1:1236/v1/speech/vacate",
    vacate_token=TOKEN,
    api_key=None,  # or the server's key / MCP PIN when one is set
    version="0.9.1",
    kind="module",  # "module" shows a Warm button on StudioForge's GUI
    health_url="http://127.0.0.1:1236/health",
    warm_url="http://127.0.0.1:1236/v1/speech/warm",
    capabilities=["stt", "tts"],
)


async def boot() -> None:
    await client.release_stale()  # a crash of ours, or a StudioForge restart (D61)
    await client.register()  # idempotent; Unsupported on an older server


async def open_session() -> int | None:
    outcome = await client.acquire("auto", prefer=[3, 2], min_free_gib=8, idle_ttl_s=1200)
    match outcome:
        case Granted(lease_id=lease_id, devices=[device]):
            asyncio.create_task(client.touch_loop(lease_id, while_active=session_is_busy))
            return device  # allocate HERE, and nowhere else
        case Refused(retryable=True, retry_after_s=wait):
            ...  # cool down `wait` seconds, ask again
        case Refused():
            ...  # lease_conflict / insufficient_vram: stand down or elsewhere;
            # outcome.considered says what stood on each card
        case Unsupported():
            return (await client.acquire([3])).devices[0]  # pre-D78 server: name the card
        case Unreachable():
            ...  # SpeechForge S16: only onto a card NVML shows clearly free
    return None


# the D56 callback, mounted on the tenant's own server (Starlette / FastAPI shown):
app.mount("/v1/speech/vacate", vacate_asgi_route(client.token_matches, drain_unload_release))
```

`drain_unload_release(request: VacateRequest)` is yours: stop taking new work, finish the piece in
flight, free the VRAM, `await client.release(request.lease_id)`. Answer within the drain budget
(`request.deadline_s`, 180 s by default); the route itself already refused anything that was not a
`POST`, from loopback, carrying your token in constant time.

## Outcomes

| Outcome | Meaning | What to do |
| --- | --- | --- |
| `Granted(lease_id, devices, view, placement)` | the card is yours | allocate on `devices`; `touch_loop`; `release` when done |
| `Refused(status, code, message, retry_after_s, considered, details)` | StudioForge said no and why | `retryable` (`lease_vacating`, `model_busy`, `priority_hold`) → wait and re-ask; otherwise stand down or ask for another card |
| `Unreachable(reason)` | no answer, or not a StudioForge | S16: load unleased only onto a clearly free card; never treat silence as a grant |
| `Unsupported(missing, reason)` | the server predates what was asked (`auto_placement`, `tenant_registry`) | fall back: explicit devices, no registration |

Every call is bounded by `timeout_s` (5 s) and returns a value rather than raising.

## How SpeechForge would adopt it

SpeechForge's `lease.py` (its S3/S16) already does what this package does, in its own words. The
mechanical change: replace its `LeaseClient` with `TenantClient("...:1234", "speechforge", 2,
vacate_url=..., vacate_token=...)`; `acquire(device)` becomes `acquire([device])` — or
`acquire("auto", prefer=[3], min_free_gib=8)` to let StudioForge pick the first free 3090 instead of
walking `gpu.devices` itself; `touch`, `release`, `release_stale` keep their names and semantics;
its vacate FastAPI route becomes `app.mount(VACATE_PATH, vacate_asgi_route(client.token_matches,
residency.vacate))`. Behaviour it keeps unchanged: class 2, never `force`, the idle TTL of
`idle_unload_s + 300`, the S16 rule on `Unreachable`. New: `await client.register()` at boot with
`kind="module"`, `health_url` and `warm_url`, so the Tenants card shows it and the operator can warm
it from StudioForge's panel.

## How ClawForge2 would adopt it

ClawForge2's `comfy/lease.py` holds ComfyUI's card as `clawforge2` at class 3 (the job's class
when a job re-asks), registers `VACATE_PATH` with a per-process token, adopts an existing lease
at boot and fails open. With this package: `TenantClient("...:1234", "clawforge2", 3,
vacate_url=..., vacate_token=...)`; the pinned card is `acquire([pin.index])` (explicit, so a
better class can still ask it to vacate — automatic placement never triggers D56); the boot-time
adoption becomes `release_stale()` followed by a fresh `acquire`, or stays as it is (adoption is
a valid choice the protocol allows); `touch_loop(..., while_active=lambda: queue_busy())` replaces
its own keep-alive; `register(kind="tenant", health_url=".../health", status_url=".../ui/api/status")`
puts it on the Tenants card. Its `lease_mode` ladder (`off`/`advisory`/`hold`/`hold_forced`) stays
its own policy; the package simply never sends `force`, so `hold_forced` would remain a direct
call if the operator insists on it.

Neither repository is modified by this package existing; adoption is each project's own change.

## Licence

MIT, the same terms as StudioForge.
