# Figure 10.3 — `diagrams/rank22-spill-cliff.svg`

**Caption it must serve:** "The allocator cliff at rank 22. Ranks 20–21 fit
under the VGPR 128 / AGPR 64 cap and spill nothing. Rank 22 takes the cap and
pushes 594–762 scalars into scratch — which is HBM-backed, and therefore the
same memory these kernels are already bandwidth-bound on. The speedup collapses
at exactly the rank where spilling starts. Note that the spilling kernels are
the *short* ones: 320–359 instructions against 16,521 for a rank-14 kernel that
spills 4 SGPRs. This is rank-driven live state, not program length."

**Title:** The Register Allocator Falls Off a Cliff at Rank 22
**Subtitle:** Spilling begins and the advantage collapses at the same rank — and the spilling kernels are the shortest in the corpus

## What to draw

Canvas ~1060 × 600. Two coupled panels sharing one rank axis (20 → 24).

### TOP PANEL — the register file, as an occupancy diagram

For each rank 20, 20(L8), 21, 22, 23, 24 draw a vertical stacked bar showing
allocation against the cap:

- **VGPR cap line at 128**, **AGPR cap line at 64** — draw both as dashed
  horizontal rules labelled `cap`.
- Ranks 20 / 20(L8) / 21: bars at **104/40**, **106/42**, **104/40** — visibly
  *below* the caps. Colour green `#00cc66`. Label each **`spill 0`**.
- Ranks 22 / 23 / 24: bars **pinned at 128/64** — touching the caps. Colour red
  `#cc2222`. From the top of each, draw a thick red arrow going **down and out**
  into a box labelled **`scratch → HBM`**, carrying the spill counts.

The visual point: the first three bars have headroom, the last three are flush
against the ceiling and overflow.

### THE SCRATCH BOX — HBM, and why it hurts

Under the spilling ranks, a wide red-bordered box:

> **scratch is HBM-backed** — spill traffic lands in the *same* memory these
> kernels are already bandwidth-bound on. The spill does not trade registers for
> a cheap cache; it trades them for the bottleneck.

Show scratch size per rank inside it: **448**, **576**, **480** bytes.

### BOTTOM PANEL — the ratio, on the same rank axis

A line or bar of V2/SVM ratio, aligned column-for-column with the top panel:

- 20 → **0.850**, 20(L8) → **0.732**, 21 → **0.736** *(green — V2 wins)*
- 22 → **0.980**, 23 → **0.990**, 24 → **0.882** *(red/orange — advantage gone)*

Draw a **vertical dashed cliff line between rank 21 and rank 22** spanning BOTH
panels, labelled: **"spilling begins · advantage collapses · same rank"**.
The two panels sharing that one line is the whole figure.

### CALLOUT (cyan `#53d8fb`) — the counter-intuitive part

> The spilling kernels are the **shortest** in the corpus: **320–359
> instructions**. A rank-14 kernel with **16,521 instructions** spills just
> **4 SGPRs**.
>
> Spilling tracks **rank** — how much amplitude state is live at once — not
> program length. A longer program is not a harder allocation problem; a wider
> one is.

## DATA — verbatim from §10.6 and §14.5, invent nothing

| rank | circuit | ratio | `.vgpr_count` | `.agpr_count` | `.vgpr_spill_count` | `.sgpr_spill_count` | scratch |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | `qv20_seed42` | 0.850 | 104 | 40 | 0 | 0 | 96 |
| 20 | `qv20_L8` | 0.732 | 106 | 42 | 0 | 0 | 112 |
| 21 | `qv21_L8` | 0.736 | 104 | 40 | 0 | 0 | 112 |
| 22 | `qv22_L6` | 0.980 | 128 | 64 | 136 | 762 | 448 |
| 23 | `qv23_L5` | 0.990 | 128 | 64 | 199 | 662 | 576 |
| 24 | `qv24_L4` | 0.882 | 128 | 64 | 152 | 594 | 480 |

- The three spilling kernels are the **top three by SGPR spill in the entire
  89-kernel corpus**.
- Read from the binaries with `llvm-readelf --notes`.
- The spilling kernels contain **320–359 instructions**; a rank-14 kernel
  contains **16,521 instructions** and spills **4 SGPRs**.

## Notes

- Do **not** use the `rocprofv3` VGPR numbers (52/56/64) here. §14.5 records that
  they disagree with the metadata by ~2× and are evidently a different quantity.
  This figure uses the **AMDHSA metadata note**, which is what the report treats
  as authoritative for allocation.
- This is a companion to `hbm-budget-pool.svg`, not a replacement. That figure
  explains the *resident pool* shrinking; this one explains the *register
  allocator*. They are two independent mechanisms that happen to bite at
  neighbouring ranks, and the report is careful not to conflate them — so this
  figure must not imply the pool causes the spill. Add a one-line footer:
  **"Second, independent mechanism — see `hbm-budget-pool.svg` for the first."**
