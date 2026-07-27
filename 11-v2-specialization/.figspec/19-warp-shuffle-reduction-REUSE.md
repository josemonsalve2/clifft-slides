# Figure 3.2 — `diagrams/warp-shuffle-reduction.svg` — ALREADY EXISTS

Review only, do not regenerate. Verify against these facts and **report**
(do not silently edit) any contradiction:

- Phase 1: a **6-step XOR butterfly** within each **64-lane** wavefront via
  `ds_bpermute`.
- Phase 2: the **four** wavefront partials combine through LDS.
- Both backends implement the identical pattern: `hip_sampler.hip:948`,
  mirrored in `v2_ops.h:224-262` (`coop_reduce2`).
- The summation **order is part of the ABI**. The in-tree comment
  (`v2_ops.h:235-236`) is emphatic:
  `// MUST reproduce SVM coop_reduce2's exact summation order or f64 rounding`
  `// diverges at measurement branch points.`
- A reduction is not associative in floating point: two implementations summing
  the same values in different orders differ in the last bits — which flips
  measurement branches and desynchronizes the PRNG (see Figure 12.1).
