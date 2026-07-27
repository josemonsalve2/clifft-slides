# Figure 6.2 — `diagrams/persistent-kernel.svg` — ALREADY EXISTS

Do **not** regenerate. Review only, and report (do not edit) any place where it
contradicts these verified facts:

- The resident pool is sized from a **32 GB HBM budget**, not a fixed count:
  `wgs = budget / bytes_per_wg`, clamped to **≤ 2048**, floored at 1, then
  rounded **down to a multiple of kNumXCDs = 8**.
- Bytes per workgroup = `2^peak_rank × 12` (amplitude slice + half-size scratch).
- Measured pool sizes: rank 20 → 2048, 21 → 1360, 22 → 680, 23 → 336, 24 → 168.
- Shots are drawn with `__atomic_fetch_add(&work_counter[0], 1, RELAXED)` by
  thread 0, then broadcast through LDS behind `v2_barrier()`.
