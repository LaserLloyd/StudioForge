## D67 -- The quant picker says which files keep their MTP heads, from each file's own header

**Status.** Built on `lane/downloads-mtp` (four commits on top of `efd06bb`), `tests/unit` green,
ruff and mypy clean. Additive on every wire surface: three new fields per quant entry, one per
repo, one column in `sfctl models repo`. Nothing existing changes shape or meaning. Not yet seen
live; the live check is the merge's job.

**Context.** HuggingFace's "hardware compatibility" panel groups a GGUF repo's files by bit-width
and badges each one that keeps its multi-token-prediction heads -- and the same quant label can
appear twice, `Q4_K_M 18.5 GB MTP` beside `Q4_K_M 18 GB`, with the badge the only thing telling
them apart. That badge matters here: a GGUF with `<arch>.nextn_predict_layers >= 1` is launched
with `--spec-type draft-mtp` (ENGINE-FEATURES.md), a measured +34% single-stream on a 27B at no
extra VRAM, and the supervisor already does that for anything in the library. The Download tab
could not show it. It reads one header per repo -- the smallest quant, for the context matrix --
and that header cannot speak for its siblings when a publisher ships the same quant with and
without the heads. HF's API exposes nothing per file (`?expand[]=gguf` is repo-level architecture,
context length and chat template; `/tree` is sizes and scan status), so the answer has to come
from each file's header, and a full header is 2-15 MB of tokenizer strings that cannot be seeked
over. Twenty-two of those per opened repo was the reason it had not been done.

The name is not evidence either way. LIMITATIONS.md already records a `...-MTP-GGUF` repo whose
files carry no such key; and unsloth's `Qwen3.8-27B-GGUF` quants carry `nextn_predict_layers=1`
under names that never say MTP, while the `llmfan46/...-Native-MTP-Preserved-NVFP4-GGUF` repo
says it in the repo name and in no file name. (Both checked live, read-only, 2026-09-22.)

**Decision.**

- **A per-file probe that stops at the tokenizer.** `gguf._read_stream` takes a
  `stop_before(key, kv_so_far)` predicate and `GgufFile` gains `kv_complete`. Every `<arch>.*`
  key precedes the first `tokenizer.*` key in what llama.cpp's converters and `llama-quantize`
  write -- `general.architecture` is key 0, the `<arch>.*` block follows the `general.*` block,
  the keys `llama-quantize` and `llama-gguf-split` append at the *end* are `general.file_type` and
  `split.*`, not architecture keys -- and on the live 27B the `nextn_predict_layers` key sits
  inside the first 1.5 KB with a 248k-entry token array right after it. So `hf_meta.remote_mtp`
  is one 256 KiB range request per file (8 MiB hard cap), cached in memory and on disk for a day
  under its own `kind: "mtp"` entry, and free when the full header is already cached (the
  context-fit read of the smallest quant) or the exact file is registered locally
  (`registry_file_meta`, stricter than the geometry's `registry_sibling_meta` because MTP is per
  file). `mtp_from_kv` refuses to guess when the walk stopped before the architecture block (a
  writer that put the tokenizer first): that is *unknown*, not *no heads*, and is not cached.
- **Bounded, per-quant, progressive.** `repo_mtp_status` walks a repo at most four probes at a
  time (`MTP_PROBE_CONCURRENCY`), degrades each quant on its own (an unreadable header falls back
  to the name hint with the reason kept in `detail`; a probe that raises is *unknown* for that row
  and nothing else), and calls `on_result` as each answer lands so the GUI paints rows one by one.
- **The data model.** `MtpStatus(mtp: true|false|null, source: "header"|"name"|null, layers,
  detail)`. Only `"header"` is a verdict. `"name"` is `hf_search.looks_like_mtp_name`, token-based
  like `looks_like_auxiliary_gguf`, applied by one rule in `GgufRepoInfo.logical_models`: when any
  loadable file in the repo names MTP, only those files are hinted (the unmarked siblings are the
  stripped variants); otherwise the repo name speaks for all. `MTP/` draft modules are auxiliary
  and never a hint. `LogicalDownload.mtp_hint` and `GgufRepoInfo.mtp_hint` carry it.
- **Wire.** Every quant entry on `GET /api/hf/repo/{id}` and `GET /api/hf/search` carries `mtp`,
  `mtp_source`, `mtp_layers`; the repo payload carries `mtp_likely`. `with_context` (always on for
  `/hf/repo`, opt-in and capped for search) is what triggers the probes, run *after* the geometry
  read so the smallest quant's probe is a cache hit; without it the fields carry the name hint.
  MCP `repo_details` keeps `mtp`/`mtp_source` and `mtp_layers` when there is a count, and its
  description tells an agent to prefer a header-confirmed MTP quant over the same quant without
  it; `search_models` rows gain `mtp_likely` and still read no header. `sfctl models repo` gains
  an `MTP` column (`yes (1)` / `no` / `likely` / `-`).
- **The picker.** Quants sit on bit-width shelves (`2-bit` ... `16-bit`, `other` last; nominal
  bits from the label's leading precision token, `quant_bits`), smallest file first inside a
  shelf, under a quiet divider with the count, below a `Rig: 2× RTX 5090 (32 GiB) + 2× RTX 3090
  (24 GiB)` line. Each row keeps everything it had -- size, weights-only verdict, the planner's
  context line and fit badge, the disk warning, Download -- and adds an MTP badge: `likely MTP`
  (outline, `info`) from the name at first paint, then `MTP` (filled), `no MTP heads` (grey; the
  name promised, the header did not deliver) or nothing as each probe lands, painted concurrently
  with the context read and never blocking. Colours are theme tokens through the existing badge
  rules; no hard-coded colour. A label that appears twice in a repo shows its file name inline
  (the llmfan46 repo's two files both parse to `NVFP4`), every label's hover lists the files that
  would be downloaded, and a subfolder file (unsloth's `BF16/...`) gets a disabled Download with
  the reason instead of a click that `safe_filename` refuses. The dialog body scrolls.
- **Search.** Rows get the same chip from the name, or filled from a registered quant of the repo
  whose header carries the heads (a dictionary walk, no network). An `MTP only` checkbox fetches a
  wider single page (60, one request either way) and filters it locally, saying how many of the
  fetched rows it kept -- HF cannot search on this and a silent "3 results" would read as "only
  three exist".

**Not taken.** Reading every file's header on the search page (twenty rows is twenty requests per
keystroke; `mtp_likely` and the registry are what a row can afford). Deriving MTP from the
architecture name at search time (`qwen35` *can* carry the heads; a stripped quant is the same
architecture). Flattening subfolder files to their basename at download time (`safe_filename`'s
collision argument stands; the row now says why instead). Building the context-fit geometry from
the same partial read: the full read also captures `n_vocab`, the chat template and the file
type, and its cache must stay consistent with the local parser. A header-order assumption
elevated to a hard rule: the probe reports *unknown* rather than *false* when it cannot prove the
key absent.

**Tests.** `tests/unit/test_download_mtp.py` (63): the name rule (token not substring; repo name
vs file names; draft modules; the owner handle ignored); the early-stop walk (`kv_complete`,
nothing past the stop, the full parser unchanged); `mtp_from_kv` settling and refusing;
`remote_mtp` confirming from one 64 KiB chunk of a 550 KB header, denying a named-MTP file,
inconclusive on tokenizer-first, memory and disk caches, free after a full read, gated and
truncated errors, the token in a header not the URL; the registry shortcut on the exact file;
shard 1 probed; the name fallback with its reason; concurrency `<= 2` under a cap of 2 with a
raising probe and a raising callback; `GET /api/hf/repo` settling a mixed repo per file with the
smallest quant fetched once; the search route carrying only the hint and reading nothing; the 403
fallback; MCP compact and search-row fields; `quant_bits`/shelves/note/tooltip/rig line; every
badge state; the GUI pass painting three rows three ways from real headers; a real NiceGUI render
of shelves, badges, the duplicate-label file name and the disabled subfolder button; the search
filter present; the search chip filled only for a registered quant with heads.
`tests/unit/test_mcp.py`'s search-row key set gains `mtp_likely`; `tests/unit/test_gui.py`'s fake
option carries the new attributes.
