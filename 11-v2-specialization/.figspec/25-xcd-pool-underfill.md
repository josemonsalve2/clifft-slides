# Figure 10.4 — `diagrams/xcd-pool-underfill.svg`

**Caption it must serve:** "Every added qubit doubles the bytes each workgroup
needs, so a fixed 32 GB budget halves the resident pool. Rank 20 fills the
device with 2,048 workgroups. Rank 24 has 168 — on a machine with 256 CUs. At
that point most of the device is idle no matter how good the generated code is,
because the pool size follows from the budget constant and the rank, both known
before the kernel launches. The formula predicted all five grids exactly."

**Title:** By Rank 24, the Machine Cannot Be Filled
**Subtitle:** 12 bytes per amplitude against a 32 GB budget — each qubit halves the resident pool

## What to draw

Canvas ~1080 × 620. Three coupled registers, top to bottom.

### TOP — the formula, as a small annotated code panel

```cpp
const uint64_t amp          = 1ull << flat.peak_rank;
const uint64_t bytes_per_wg = amp * sizeof(GpuComplex) + (amp/2) * sizeof(GpuComplex);
const uint64_t budget       = 32ull << 30;   // 32 GB
uint64_t wgs = budget / bytes_per_wg;
if (wgs < 1)    wgs = 1;                     // rank 26+: at least one
if (wgs > 2048) wgs = 2048;
if (wgs > kNumXCDs) wgs -= wgs % kNumXCDs;   // spread evenly over 8 XCDs
```

Annotate the three clauses in the margin: **budget divide**, **floor of 1 —
rank 26 is *possible*, not *fast***, **ceiling 2,048**, **round down to a
multiple of `kNumXCDs = 8`**. Note `GpuComplex` is 8 bytes, so `bytes_per_wg` is
**12 bytes per amplitude** (a slice plus a half-size scratch).

### MIDDLE — the device, drawn five times at decreasing occupancy

This is the heart of the figure. Draw the gfx950 device as **8 XCD blocks**
side by side, each subdivided into CUs (256 CUs total across the eight dies).
Repeat that device map **five times** in a row, one per rank, and shade the
CUs that have resident workgroups:

| rank | MiB/wg | workgroups | how the device looks |
|---:|---:|---:|---|
| 20 | 12.00 | **2,048** | every CU multiply occupied — solid fill, green `#00cc66` |
| 21 | 24.00 | **1,360** | still oversubscribed — green |
| 22 | 48.00 | **680** | ~2.7 wg/CU — orange `#ff8800` |
| 23 | 96.00 | **336** | ~1.3 wg/CU — orange, gaps appearing |
| 24 | 192.00 | **168** | **fewer workgroups than CUs** — red `#cc2222`, visible empty CUs |

Under rank 24, a red brace across the empty CUs: **"88 CUs with no resident
workgroup — 168 wgs on 256 CUs"**.

Show the halving explicitly with arrows between the five device maps, each
labelled **`×2 bytes/wg → ÷2 pool`**.

### THE XCD ROUNDING — a small inset

Show why the pool is a multiple of 8: a strip of 8 XCDs receiving an even share
vs. an uneven one. Label: **"round down to a multiple of `kNumXCDs = 8` so the
resident pool spreads evenly across the compute dies — never inflate past the
budget."**

### BOTTOM — prediction vs measurement

A five-row strip, each row: **predicted = measured ✓**

| rank | predicted | measured |
|---|---|---|
| 20 | 2,048 | 2,048 ✓ |
| 21 | 1,360 | 1,360 ✓ |
| 22 | 680 | 680 ✓ |
| 23 | 336 | 336 ✓ |
| 24 | 168 | 168 ✓ |

Callout (cyan `#53d8fb`): **"Five predictions, five exact matches. This is
structural, not incidental — it follows from the budget constant and the rank,
both known before the kernel launches."**

## DATA — verbatim from §6.7 and §14.5, invent nothing

- `kNumXCDs = 8`; budget **32 GB**; ceiling **2,048**; floor **1**.
- **12 bytes per amplitude** (`GpuComplex` = 8 bytes, plus a half-size scratch).
- MiB/workgroup: rank 20 **12.00**, 21 **24.00**, 22 **48.00**, 23 **96.00**,
  24 **192.00**.
- Resident workgroups, predicted **and** measured: **2048 / 1360 / 680 / 336 /
  168**.
- The device has **256 CUs**. At rank 24, **168 workgroups** means the machine
  cannot be filled.
- Ranks 11–20 are **ceiling-limited** (2,048); ranks 21+ are **budget-limited**.

## Notes

- EXTENDS `hbm-budget-pool.svg`, which plots the pool curve. This figure adds
  the **spatial** consequence — workgroups against XCDs and CUs — which the
  curve cannot show.
- This must **replace**, not reuse, `numa-xcd-optimization.svg`: that figure is
  deck-10 **MI300X** art and this report is gfx950 throughout. Draw 8 XCDs and
  256 CUs; do not import MI300X geometry.
- Keep this mechanism distinct from register spilling (`rank22-spill-cliff.svg`).
  They are **two independent mechanisms** that bite at neighbouring ranks, and
  the report is careful not to conflate them. Footer:
  **"First of two independent mechanisms — see `rank22-spill-cliff.svg` for the
  second."**
