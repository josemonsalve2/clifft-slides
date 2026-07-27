# Figure 10.1 — `diagrams/hbm-budget-pool.svg`

**Caption it must serve:** "Resident workgroup pool under the 32 GB budget.
Below rank ~19 the pool is clamped at 2,048 workgroups; above it, the pool
halves with every rank while per-workgroup memory doubles. The product is
constant, which is the point."

**Title:** The HBM Budget — the pool shrinks as the rank grows
**Subtitle:** v2_kernel.cc:432-446 · 12 bytes per amplitude · 32 GB budget · cap 2048 · multiple of 8 XCDs

## What to draw

A chart with rank on x (say 14 → 26) and **two y-axes, mirrored**: resident
workgroups (descending staircase, cyan) and bytes per workgroup (ascending,
orange, log scale). Their **product** drawn as a flat line pinned at the 32 GB
budget — that flatness is the figure's whole claim.

### DATA — the formula, exact (quote it in a code panel)

```cpp
// Resident pool sized to a fixed HBM budget: each workgroup owns one
// amplitude slice (1<<peak_rank) + a half-size scratch, i.e. 12 bytes
// per amplitude. At rank 19 that is 6 MB/wg; at rank 26 it is 768 MB/wg.
// The 32 GB budget keeps the pool sane at both ends on a 288 GB MI355X.
const uint64_t amp = 1ull << flat.peak_rank;
const uint64_t bytes_per_wg = amp * sizeof(GpuComplex) + (amp / 2) * sizeof(GpuComplex);
const uint64_t budget = 32ull << 30;  // 32 GB
uint64_t wgs = budget / bytes_per_wg;
if (wgs < 1) wgs = 1;               // rank 26+: at least one resident wg
if (wgs > 2048) wgs = 2048;
global_grid_wgs = static_cast<uint32_t>(wgs);
// Prefer a multiple of the XCD count when the budget allows it, so the
// resident pool spreads evenly; never inflate past the budget.
if (global_grid_wgs > kNumXCDs) global_grid_wgs -= global_grid_wgs % kNumXCDs;
```

### DATA — the measured pool sizes. These are verified; plot exactly these.

| rank | resident workgroups |
|---|---|
| 20 | **2048** (clamped) |
| 21 | **1360** |
| 22 | **680** |
| 23 | **336** |
| 24 | **168** |

Note for the rounding step: 21's raw quotient rounds down to a multiple of
**kNumXCDs = 8** to give 1360; likewise 680, 336, 168 are all multiples of 8.
Mark the "× multiple of 8" rounding visually (small tick marks or a note).

### DATA — the two anchors named in the source comment

- rank 19 → **6 MB per workgroup**
- rank 26 → **768 MB per workgroup**

### Required annotations

1. A horizontal clamp line at **2,048** labelled "cap — below rank ~19/20 the
   budget would allow more; the cap holds it".
2. The flat product line labelled "**32 GB — constant**".
3. An XCD strip along the bottom: eight compute dies, with a note
   "`kNumXCDs = 8` — the pool is rounded down to a multiple of 8 so it spreads
   evenly across the device's eight compute dies."
4. A right-edge note: "the cap is kept **≤ 30** so `1u << rank` stays inside a
   `u32`."
5. Contrast callout: "**the pool now shrinks as rank grows, instead of the
   allocation exploding.**"

### Punchline band

"Per-workgroup memory doubles every rank; the pool halves. The product is the
budget."
