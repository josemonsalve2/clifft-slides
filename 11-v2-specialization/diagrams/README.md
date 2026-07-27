# Figures — deck 11 / V2 report

27 SVGs live here. **19 are used**; 7 are deliberately not. This file records
which is which, so an unused figure is not mistaken for an oversight and
re-added.

Authoring rules are in `../.figspec/STYLE.md`. Each figure has a spec under
`../.figspec/NN-<slug>.md` giving the DATA it is allowed to draw — every number
in a figure must appear verbatim in its spec, which in turn quotes the report.

## In use (19)

Referenced by both `../REPORT.md` and the assembled deck:

| figure | report § |
|---|---|
| `factored-state.svg` | §2 |
| `bytecode-layout.svg` | §2 |
| `three-loops.svg` | §6 |
| `ir-density.svg` | §5 |
| `v1-mlir-reality.svg` | §5 |
| `spec-classes-gains.svg` | §7 |
| `static-rank-tracking.svg` | §7 |
| `lowering-pipelines.svg` | §8 |
| `memory-hierarchy-tiers.svg` | §6 |
| `warp-shuffle-reduction.svg` | §6 |
| `persistent-kernel.svg` | §6 |
| `f64-attribution.svg` | §9 |
| `optimization-timeline.svg` | §9 |
| `scatter-index-folding.svg` | §9 |
| `hbm-budget-pool.svg` | §10 |
| `barrier-race.svg` | §11 (Figure 11.1) |
| `prng-desync.svg` | §12 (Figure 12.1) |
| `q2-convergence.svg` | §12 (Figure 12.2) |
| `dispatch-latency.svg` | §13 (Figure 13.1) |

## Not used, on purpose (7)

All seven were inherited from **deck 10**, which characterised **MI300X
(`gfx942`, 304 CUs, 64 KB LDS/CU)**. This report is **MI350X (`gfx950`, 256 CUs,
8 XCDs, 160 KB LDS/CU)** throughout, and the report's own standing rule is
*never compare across node types*. Dropping them into a gfx950 report would put
gfx942 geometry next to gfx950 measurements on the same page.

| figure | why it stays out |
|---|---|
| `mi300x-architecture.svg` | gfx942 chip geometry — wrong part for this report |
| `numa-xcd-optimization.svg` | MI300X XCD layout; gfx950 has 8 XCDs and different L2 |
| `circuit-translation.svg` | MI300X-era, and §2's pipeline is covered by `factored-state` + `bytecode-layout` |
| `compiled-megakernel.svg` | describes deck 10's megakernel, not V2's one-call-per-instruction design |
| `hiprtc-vs-aot.svg` | HIPRTC JIT vs AOT — V2 has no HIP at all, so the comparison does not arise |
| `gpu-execution-timeline.svg` | its occupancy figures ("4 waves/SIMD") are deck-10 measurements |
| `svm-interpreter.svg` | "256 VGPRs/CU" is deck-10 geometry; §2's interpreter story needs no figure |

If any of these is wanted here, it needs re-authoring against gfx950 numbers
from `V2_performance/runs/20260727T125310Z_report-final-allfixtures/` — not a
re-reference.
