# Figures — deck 11 / V2 report

33 SVGs live here. **26 are used**; 7 are deliberately not. This file records
which is which, so an unused figure is not mistaken for an oversight and
re-added.

Authoring rules are in `../.figspec/STYLE.md`. Each figure has a spec under
`../.figspec/NN-<slug>.md` giving the DATA it is allowed to draw — every number
in a figure must appear verbatim in its spec, which in turn quotes the report.
`../.figspec/check.py` verifies layout (overflow, overlap, glyph coverage)
against cairosvg's rendering model; every figure below passes it with 0 findings.

## In use (26)

Referenced by both `../REPORT.md` and the assembled deck. **The figure number is
the report's**; the deck uses the same numbers, and where a figure is reused on a
second slide that reuse is left unnumbered so each number has one definition.

| figure | number | subject |
|---|---|---|
| `factored-state.svg` | 2.1 | the factored state, dormant qubits cost zero |
| `bytecode-layout.svg` | 2.2 | the 32-byte instruction encoding |
| `memory-hierarchy-tiers.svg` | 3.1 | rank → memory tier |
| `warp-shuffle-reduction.svg` | 3.2 | the two-phase cooperative reduction |
| `v1-mlir-reality.svg` | 5.1 | V1's intended vs actual MLIR usage |
| `ir-density.svg` | 5.2 | emitted source lines per bytecode instruction |
| `three-loops.svg` | 6.1 | the three loops and their rules |
| `one-library-two-consumers.svg` | 6.2 | one operand library, two consumers, two tiers |
| `persistent-kernel.svg` | 6.3 | the global tier's persistent kernel |
| `spec-classes-gains.svg` | 7.1 | per-class specialization gains |
| `scalar-pipe-deletion.svg` | 7.2 | the operand-fetch dependency chain, deleted |
| `static-rank-tracking.svg` | 7.3 | static rank tracking through a program |
| `noise-loop-cannot-fold.svg` | 7.4 | folded constants that do not shorten the loop |
| `lowering-pipelines.svg` | 8.1 | both pipelines with per-stage sizes |
| `f64-attribution.svg` | 8.2 | where V1's f64 instructions come from |
| `optimization-timeline.svg` | 9.1 | V2/SVM ratio over the optimization sequence |
| `register-tier-topology.svg` | 9.2 | P0 — cooperation as a compile-time parameter |
| `scatter-index-folding.svg` | 9.3 | the global-tier bottleneck |
| `hbm-budget-pool.svg` | 10.1 | resident pool vs rank under the 32 GB budget |
| `rank22-spill-cliff.svg` | 10.2 | the allocator cliff at rank 22 |
| `barrier-race.svg` | 11.1 | why an execution-only barrier loses a partial |
| `prng-desync.svg` | 12.1 | the desynchronization mechanism |
| `q2-convergence.svg` | 12.2 | Q2 convergence of the relative gap |
| `dispatch-latency.svg` | 13.1 | per-dispatch latency, six modes |
| `hsa-aql-dispatch-path.svg` | 13.2 | what an AQL dispatch actually does |
| `xcd-pool-underfill.svg` | 14.1 | pool underfill across 8 XCDs / 256 CUs |

Figures 6.2, 7.2, 7.4, 9.2, 10.2, 13.2 and 14.1 were added in the round-2 figure
pass (specs `../.figspec/20-`…`26-`), which prioritized **mechanism** — why a
number is what it is — over restating tables the report already carries.

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
