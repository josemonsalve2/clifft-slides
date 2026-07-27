## 16. Conclusions and open items

### 16.1 What was built, and what it is worth

Three rewrites of the `clifft` GPU backend, each answering the previous one's
failure:

| | approach | outcome |
|---|---|---|
| **SVM** | one interpreter kernel, three tiers, bytecode in HBM | the baseline. Correct, general, and paying interpretation cost on every instruction of every shot. |
| **Hybrid** | interpreter + per-circuit HIP source compiled at runtime | worked, did not push execution further. Compilation cost and HIP's launch path ate the gain. |
| **V1** | per-circuit MLIR → single monolithic kernel, no interpreter, no runtime | **failed.** Unrolling the circuit into straight-line IR produced kernels too large to compile or schedule. |
| **V2** | per-circuit specialization *over* a runtime, direct C → amdgcn, HSA dispatch | **0.573 geometric mean against SVM, 26 of 26 circuits faster.** |

The load-bearing lesson sits between V1 and V2, and it is not "MLIR was the
wrong tool." It is that **specialization and unrolling are separable, and V1
conflated them.** V1 assumed that to specialize a circuit you must emit it as
straight-line code. V2 specializes exactly as aggressively — the same constants
folded, the same branches removed — while keeping loops as loops and keeping a
runtime underneath. The result compiles in seconds instead of failing to
compile, and it is faster.

### 16.2 Where the speedup comes from

Three mechanisms, in descending order of how much of the corpus they explain.

**1. Scalar instruction deletion (§14.3).** The dominant effect, and not the one
the project expected. On **26 of 26 circuits** the SALU ratio is below the VALU
ratio — V2 issues 2.8× to 9.1× fewer scalar instructions. What disappears is the
interpreter's per-instruction overhead: bytecode address arithmetic (`pc * 40`,
a 64-bit multiply), the load, the `v_readfirstlane` to move the operand to the
scalar unit, the `active_k` reload, the bounds check, the switch dispatch. None
of it computed an amplitude.

This was documented for several drafts as work *moved* from the vector pipe to
the scalar pipe. That was wrong, and §14.3 retracts it: nothing migrates, both
pipes issue less, and the scalar pipe issues much less.

**2. Resource footprint (§14.6, §15.5).** V2's global-tier kernels declare 1,024
bytes of LDS against SVM's 8,704 — 8.5× — because with the amplitudes in HBM
there is nothing to stage. On the register tier V2 allocates 656–1,040 bytes of
scratch against SVM's uniform 4,480, and executes **zero** LDS instructions
against SVM's 25,280. These are consequences of specialization knowing what the
circuit contains, not of tuning.

**3. Nothing exotic.** `SQ_INSTS_MFMA` is **0 across all 52 cells** — 26
circuits × 2 backends. The workload is a butterfly reduction over pairs of
complex amplitudes, not a GEMM, and no part of this result comes from the matrix
cores. Any future proposal to use them is unsupported by this corpus.

The proximate check on all of it: `SQ_BUSY_CYCLES` ratio predicts kernel-time
ratio at **Pearson r = 0.942**, and `GRBM_GUI_ACTIVE` — an independent activity
counter — agrees with `SQ_BUSY_CYCLES` within a few percent on 24 of 26.

### 16.3 Where it does not come from

**The rank ceiling is real and it is register spilling (§14.5, §10).** Above
rank 21 the win collapses to 0.88–0.99. The cause is visible in the AMDHSA
metadata rather than inferred from counters:

| rank | spill (VGPR / SGPR) | ratio |
|---:|---|---:|
| 20–21 | 0 / 0 | 0.732–0.850 |
| 22 | 136 / 762 | 0.980 |
| 23 | 199 / 662 | 0.990 |
| 24 | 152 / 594 | 0.882 |

Spilling begins at exactly the rank where the speedup dies, and it spills to
HBM-backed scratch — the memory these kernels are already bandwidth-bound on.
Compounding it, the resident workgroup pool halves with every added qubit: the
HBM budget formula yields 2048/1360/680/336/168 workgroups for ranks 20–24
(predicted from source constants, matching the measured grids **five for five**),
so by rank 24 there are 168 workgroups on a 256-CU device and the machine cannot
be filled.

**Specialization does not help what it cannot fold.** §7's taxonomy predicted
this circuit-by-circuit before the corpus was measured: the surface family, all
frame ops and dormant measurements, folds 5.50× on VALU and runs at 0.256; the
d5 family is 19.1 % noise ops that fold 1.02×, and runs at 0.79–0.86. The
weakest coop-tier result was predicted from a single microbenchmark and was
correct.

### 16.4 Open items

Ordered by how much they would change the result.

**1. The rank-21/22 allocation cliff.** *The top item.* The three spilling
kernels are the three **shortest** in the tier (320–359 instructions); a rank-14
kernel emitting 16,521 instructions spills 4 SGPRs. Circuit length is not the
mechanism. What changes at the boundary is one baked constant — `amp_capacity`
crossing 2²² — and the emitted C is otherwise structurally identical. The cause
is inside LLVM's allocator and has not been isolated. **First step is an LLVM
allocation study at the boundary, not a code change.**

**2. The d5 anomaly: three counters, one population, no mechanism.** On all six
d5-family circuits, both activity counters say the GPU is ~0.63 as busy on V2
while kernel time says 0.79–0.86 — *less busy, and yet slower than that predicts*.
The same six circuits carry the entire busy-cycle residual (17.9–28.2 %, and
essentially zero elsewhere) and a residual L2 deficit (91 % against SVM's 98 %,
down from 71.5 % pre-fence but not gone). Three independent signals, one circuit
family, no explanation. This is the largest unexplained result in the report.

**3. The specializer does not yet bake per-circuit LDS sizing.** It knows each
circuit's peak rank and could size the coop tier's LDS exactly; it currently
inherits a conservative static bound (§9.3). A known, bounded win nobody has
taken.

**4. `batched16` is conditional, not a missed win.** HSA batched dispatch
measures 2,320 ns against the persistent path's 6,326 — 2.7× — but V2 issues
**one dispatch per run**, so there is nothing to amortize. It would pay only if
V2 moved to a multi-dispatch structure (chunked shot batches for progress
reporting, or memory-bounded rank-26 runs). Capability, not usage.

**5. The residual f32-vs-f64 branch-probability effect.** §12 established that
the d5 divergence was *not* precision — it was a threshold constant calibrated
for f64 and left in place when storage moved to f32. The genuine precision
effect predicted in §12.1, branch probabilities good to ~1e-6 relative, remains
real and unobserved: nothing in this corpus has isolated an instance. Bounded,
rare, and still open.

**6. The `rocprofv3` VGPR discrepancy.** The profiler reports 52–64 VGPRs and
`Accum_VGPR_Count = 0` where the `.hsaco` metadata reports 104–128 plus 40–64
AGPRs, for kernels whose scratch sizes match one-to-one. Not a threat to any
conclusion — the metadata is authoritative and the spill counts come from it —
but the report should not carry two numbers for one quantity indefinitely.

**7. `SQ_WAVES` on persistent global-tier kernels.** The counter does not
reconcile with launch geometry above rank 19: three identical 512-workgroup SVM
grids report 8,832 / 10,240 / 8,000 waves where geometry implies 2,048, while on
the other 20 circuits it equals workgroups × 4 exactly. §14.5 retracted a claim
built on it. Worth understanding before any future analysis uses this counter.

### 16.5 What this report is, and how to check it

Every quantitative claim here is backed by an artifact in the tree: a run
directory under `V2_performance/runs/`, a `.s` file under `lowering/`, or a line
of committed source. `V2_performance/VERIFIED_FACTS.md` is the ledger — sixteen
audit passes, each recording what was checked, what was wrong, and a method note
about how the error survived.

The instruction this report was written under was *trust data, not text*, and it
was aimed first at the report itself. The audits found, among others: a headline
that said 20 wins where the data said 26; a claimed mechanism (scalar
substitution) contradicted by a counter the project had already collected; a
residual attributed to the wrong circuit family because the surrounding argument
predicted it would land there; a per-wave cost computed by dividing counters from
two different profiling passes; and a hedge about spilling written against
evidence sitting in the binaries. Four of those five errors flattered the result
or the story. **That asymmetry is the argument for the ledger.**

Every number above is reproducible with one command:

```bash
sbatch V2_performance/tools/bench_all.sh <label>
```

If `manifest.json` reads `"dirty": true`, or the circuit count is not 26, the run
is void.
