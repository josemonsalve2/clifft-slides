# Figure 3.1 — `diagrams/memory-hierarchy-tiers.svg` — EXISTS, NEEDS TWO CORRECTIONS

**Do not redraw from scratch.** Edit the existing file, preserving its layout,
palette and structure. Two facts in it are now known to be wrong, and one label
is stale:

### Correction 1 — the LDS capacity is wrong

The Tier 2 panel reads `LDS: 64KB/CU`. **gfx950 has 160 KB of LDS per CU, not
64 KB.** This is recorded in the project's verified-facts ledger (§10b) as an
audit finding: every occupancy figure derived from a 64 KB budget is wrong for
this chip. Change the label to `LDS: 160KB/CU (gfx950)`.

### Correction 2 — the rank axis stops at 19

The axis is drawn 0 → 19 and Tier 3 is labelled for ranks 11–19. **The global
tier cap is now rank 26** (`kGlobalMaxPeakRank = 26`, `gpu_types.h`). Extend
the axis to 26, keep the tick marks at 0, 4, 5, 10, 11, and add ticks at 19
(the old cap, worth marking as a historical boundary) and 26.

### Correction 3 — tier boundaries, for the record (verify the labels match)

| tier | rank | topology | amplitude storage |
|---|---|---|---|
| register | ≤ 4 | 1 shot per **thread** | `GpuComplex v[16]` in VGPRs |
| coop | 5–10 | 1 shot per **workgroup** (256 threads) | `__shared__ GpuComplex v[1024]` |
| global | 11–26 | persistent workgroup pool, work-stealing | HBM slice per workgroup |

Constants: `kThreadMaxPeakRank = 4`, `kSharedMaxPeakRank = 10`,
`kGlobalMaxPeakRank = 26` (`gpu_types.h:8-18`).

### Leave alone

The throughput figures (48M / 5.6M shots/s) — those are not re-verified here, so
do not touch them, but if they are labelled as SVM measurements keep that label.
