# Rejected changes

Changes tried against the Phase 0 baseline and not kept, so nobody re-derives them.

**The rule (as amended by the owner).** Improvements found along the way are kept regardless of
size. The ≥10% bar — median decode tok/s at 256k, *or* prefill tok/s at ≥128k, with no >2%
regression in the other — gates only major restructuring such as adding a new backend. So a row
lands here for one of two reasons: it *regressed* a headline metric by more than 2%, or it was
a major restructuring that did not clear the bar. A small win is never rejected on size.

Every row must point at the results file that measured it (`bench/results/<sha>[-<label>].json`),
and the file's `validity.valid` must be `true` — an invalid run cannot reject anything. Results
files are gitignored and stay on the machine that ran them, so the delta column is the published
record.

| Change | Measured delta (median decode @256k / prefill @≥128k) | Date | Results file | Why rejected |
| --- | --- | --- | --- | --- |
| _example: `--flash-attn on` forced instead of auto_ | _decode −0.3% / prefill −4.1%_ | _2026-09-10_ | _`3afb47ef7ff5-fa-on.json` vs `3afb47ef7ff5.json`_ | _prefill regression >2%_ |
