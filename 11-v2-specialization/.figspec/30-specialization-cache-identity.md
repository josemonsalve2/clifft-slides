# Figure 11.2 — `diagrams/specialization-cache-identity.svg`

**Caption it must serve:** "A cache key that hashed the generated C but not the
headers it includes. The generated file is barely half the translation unit —
every `v2_op_*` body, `v2_barrier()`, and every tunable constant lives in the
headers — so a header fix produced the same key, and the day-old `.hsaco` won
*along with its `.gate` verdict*, which is keyed off the same path. The
benchmark run submitted specifically to guarantee that every performance claim
reflected shipped, correct code dispatched pre-fix kernels for every specialized
circuit. It is provable rather than suspected: the stale binaries contain the
pre-fix constant as a bit pattern."

**Title:** Half a Translation Unit
**Subtitle:** A cache key that could not see the code that mattered

## What it must show

**Three bands, stacked.**

### Band 1 — the key, broken vs fixed

Draw the translation unit as a stack of source cards, and the hash as a funnel
that only some of them reach.

Left, **BROKEN** (red `#cc2222`): the funnel is fed by `csrc` alone. Show the
headers — `v2_ops.h`, `v2_ops_body.inc`, `device_abi.h` — sitting *beside* the
funnel with a dashed grey arrow that stops short of it, labelled **not in the
hash**. Annotate the headers with what they actually contain: *every `v2_op_*`
body, `v2_barrier()`, every tunable constant*. Annotate `csrc` with **barely
half the translation unit**.

Right, **FIXED** (green `#00cc66`): the real key, quoted verbatim from
`v2_compile_cache.cc:125-127`:

    std::string ident = csrc + "|" + llvm_bin("clang") + "|" + arch() + "|" +
                        bitcode_dir() + "|" + device_header_ident();
    size_t h = std::hash<std::string>{}(ident);

with all five terms drawn as inputs reaching the funnel. On
`device_header_ident()`, a cyan callout: **content, not mtime** — mtimes change
on every checkout and would defeat the cache for no reason.

### Band 2 — what one key produces, and therefore what goes stale

From the hash, **two** artifacts, not one — this is the part that makes the bug
expensive:

    <hash>.hsaco       the binary
    <hash>.hsaco.gate  the correctness verdict, keyed off the same path

Draw a header edit as an input event on the left, with two arrows labelled
*should invalidate* pointing at both artifacts — and under BROKEN, both arrows
crossed out. One line: **a stale "0" hides the fix**.

### Band 3 — the proof, on two independent markers

A comparison table, `build cache (what the run used)` against `fresh cache`:

| marker | build cache | fresh cache |
|---|---|---|
| barriers with `ds_*` in flight and no `lgkmcnt(0)` | **1400 / 1509 (92.8 %)** | **0 / 1509 (0.0 %)** |
| `s_barrier` preceded by `lgkmcnt(0)` within 3 instrs | **49 / 1509 (3.2 %)** | 1404 / 1509 (93.0 %) |
| `V2_DUST_EPS` baked into the binary | **`0x3c32725d…` = `1e-18`** | `0x3da5fd7f…` = `1e-11` |

Mark the dust row as **the cleaner marker** with its reason: it is a bit
pattern, not a heuristic — each constant appears **exactly twice** in its
respective binary and **zero** times in the other. *There is no interpretation
involved.*

Beside the table, a small timeline, left to right, with four marks:

    32 of 36 cached kernels   2026-07-25
    remaining 4               07-26 00:00-01:48
    barrier fix               07-26 06:33
    dust fix                  07-26 13:31

with a bracket over the first two labelled **every kernel the run dispatched
predates both fixes**.

### Bottom strip

The consequence, in the report's own terms: the stale run recorded the
`coop_r10_n1720` gate as **failing** — a verdict on the **pre-fence** binary —
and an entire section was built around *"the single largest open correctness
item in V2."* **It was already fixed.** The claim survived only because the
cache kept handing back the binary from before the fix.

Beside it, in orange, the near-miss: the hazard was *known*. `d5_fence.sh`
documents it exactly, and every `d5_*.sh` diagnostic sets its own
`V2_SPEC_CACHE_DIR` for that reason.

## DATA — verbatim, invent nothing

- Key source, `v2_compile_cache.cc:125-127`, exactly as quoted above.
- The three device headers: `v2_ops.h`, `v2_ops_body.inc`, `device_abi.h`.
- Markers: **1400 / 1509 (92.8 %)** vs **0 / 1509 (0.0 %)**; **49 / 1509
  (3.2 %)** vs **1404 / 1509 (93.0 %)**; `0x3c32725d…` = `1e-18` vs
  `0x3da5fd7f…` = `1e-11`.
- Each constant appears **exactly twice** in its binary and **zero** times in
  the other.
- Timestamps: **32 of the 36** cached kernels dated **2026-07-25**; remaining
  **4** from **07-26 00:00–01:48**; barrier fix **07-26 06:33**; dust fix
  **07-26 13:31**.
- The affected shape: `coop_r10_n1720`. The quarantined cache:
  `V2_performance/history/stale_spec_cache_20260725/`.
- `d5_fence.sh`'s warning, verbatim: *"The `.gate` verdict is keyed on the
  generated-C hash, which does **NOT** change when `v2_ops.h` changes — a stale
  '0' would hide the fix."*

## Notes

- The two artifacts (`.hsaco` **and** `.gate`) must both be drawn. A figure that
  shows only the stale binary misses why this bug produced a *false correctness
  claim* rather than merely a stale measurement.
- Do not draw this as "the cache was wrong". The cache did exactly what its key
  said; the key was incomplete. That distinction is the transferable lesson.
- Both binaries survive in-tree, so this figure's numbers are checkable — see
  §10.6's provenance note for the pre/post byte comparison of the same kernels.
