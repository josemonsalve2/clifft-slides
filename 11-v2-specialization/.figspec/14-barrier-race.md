# Figure 11.1 — `diagrams/barrier-race.svg`

**Caption it must serve:** "Why an execution-only barrier loses a partial sum.
Wave 0 retires `s_barrier` with its `ds_write` to `lds_red0` still in flight;
wave 1 passes the same barrier and reads the stale word. The reduction total is
wrong, so `sample_branch` compares against a corrupted probability and takes the
other side — consuming a different number of PRNG draws from that shot onward."

**Title:** An Execution-Only Barrier Is Not a Memory Barrier
**Subtitle:** `s_barrier` on AMDGCN is `IntrNoMem` — it orders waves, not memory

## What to draw — a two-lane timeline (wave 0, wave 1), twice: BROKEN and FIXED

### TOP: BROKEN (red `#cc2222`)

Wave 0 lane: `v_add` (partial sum) → `ds_write lds_red0` **[in flight]** →
`s_barrier` (retires immediately) → continues.
Wave 1 lane: … → `s_barrier` → `ds_read lds_red0` → **reads the STALE word**.

Draw the in-flight `ds_write` as a dashed arrow that lands *after* wave 1's
read, crossing the barrier line. Put a red ✗ on the read. Annotate the barrier
line: "`s_barrier` alone is **execution-only** — LLVM models it `IntrNoMem`."

### BOTTOM: FIXED (green `#00cc66`)

Wave 0 lane: `ds_write lds_red0` → **release fence** → `s_barrier` →
**acquire fence** → continues.
Wave 1 lane: … → release fence → `s_barrier` → acquire fence →
`ds_read lds_red0` → **reads the correct word** (green ✓).

Show the fence as generating a wait: the audit's marker is
`s_barrier` **preceded by `lgkmcnt(0)`**.

### DATA — the ISA audit, exact. Both columns, both meanings.

| `.hsaco` | barriers | fenced (`lgkmcnt(0)` within 3) | `ds_*` in flight, unfenced |
|---|---|---|---|
| pre-fix `coop_r10_n1720` | 1,509 | **49 (3.2 %)** | **1,400 (92.8 %)** |
| post-fix `coop_r10_n1720` | 1,509 | **1,404 (93.0 %)** | **0 (0.0 %)** |

Annotate why both columns matter: "the pre-fix kernel has **49** barriers that
*happen* to sit behind a wait emitted for an unrelated reason — accidentally
correct, not correct by construction. Post-fix the in-flight column is **zero**,
which is the property that actually matters."

### DATA — the cost, exact

The interpreter build: **5,629 → 5,848 static instructions, +3.89 %**.
With `-DV2_REGISTER` there are no barriers at all, so the two builds are
**byte-identical at 4,457 instructions — the fix costs the register tier
nothing.**

### DATA — the payoff, exact (job 50389, median of 5, 10,000 shots, one node, both arms back-to-back)

| circuit | interpreter | specialized | speedup |
|---|---|---|---|
| circuit_d5_p0.0005 | 14.76 ms | 8.56 ms | **1.72×** |
| circuit_d5_p0.001 | 14.89 ms | 8.56 ms | **1.74×** |
| circuit_d5_p0.002 | 15.09 ms | 8.63 ms | **1.75×** |
| circuit_d5_p0.003 | 15.30 ms | 8.68 ms | **1.76×** |
| circuit_d5_p0.005 | 15.62 ms | 8.87 ms | **1.76×** |
| circuit_d3_p0.001 | 0.624 ms | 0.379 ms | **1.65×** |

Draw this as a small paired-bar strip. Annotate: "with the fence in place the
gate **passes** on `coop_r10_n1720`, and the specializer is selected for all
six circuits."

### The consequence chain — draw it as a labelled cascade, it is the point

`stale partial sum` → `wrong reduction total` → `sample_branch compares against
a corrupted probability` → `takes the other branch` → `consumes a different
number of PRNG draws` → **`the two backends never resynchronize for that shot`**

### Punchline band

"The gate caught it. It was diagnosed for months as an FP-rounding problem in
the noise ops — it was a missing fence."
