## 14. Performance evaluation: V2 against SVM

Everything to this point has been mechanism — what the specializer folds, what
the compiler does with it, what an individual opcode costs. This chapter is the
end-to-end evaluation: 26 circuits, both backends, hardware counters, and an
attempt to answer the only question that matters — *where does the time actually
go, and why is V2 faster?*

### 14.1 Method

One SLURM job, **50793**, on one node.

| | |
|---|---|
| run | `20260727T125310Z_report-final-allfixtures` |
| node | `smci350-rck-g03-d13-21` (`mi350x-es`, gfx950) |
| commit | `79d4463`, working tree clean |
| circuits | 26, all paired (V2 + SVM), **zero `rocprofv3` aborts** |
| metric | **GPU kernel time**, `per_dispatch_ns_median` from `rocprofv3` |

Counters are collected in **three separate profiling passes**, because the
hardware cannot capture them all simultaneously:

| pass | counters |
|---|---|
| `pmcA` | `SQ_INSTS_{VALU,SALU,LDS,MFMA}`, `SQ_WAVES` |
| `pmcB` | `TCC_HIT_sum`, `TCC_MISS_sum` |
| `pmcC` | `GRBM_GUI_ACTIVE`, `SQ_BUSY_CYCLES`, `SQ_WAIT_INST_LDS`, `SQ_WAVE_CYCLES` |

Each pass is a **separate execution of the kernel**. Counters from the same pass
may be combined freely; counters from different passes may not be divided into
each other without an argument that both executions did the same work. §14.5
turns on exactly this point, and an earlier draft of this chapter got it wrong.

Three further methodological commitments, each learned the hard way and each now
enforced rather than merely intended:

1. **Kernel time, not host wall time.** The host path differs between the two
   backends (§13 is entirely about that difference), so wall time would measure
   the dispatch mechanism rather than the generated code. Where dispatch
   overhead is the subject, §13 measures it directly.
2. **Both arms in one job on one node.** `mi350x-es` is heterogeneous;
   `node.json` in every run directory carries the warning `"Do NOT compare
   ratios across nodes."` §1.3(d) records what happens when this is violated —
   a projection off by 2×, in the flattering direction.
3. **The denominator is checked against the expected denominator.** Job 50785
   reported `wins 18/18` over a 26-circuit corpus because eight fixture paths
   had gone stale and `rocprofv3` aborted on each. `bench_all.sh:131-138` now
   aborts the run rather than continuing past a missing input:

   ```bash
   # Fail loudly on a stale fixture path. Job 50785 lost 8 of 26 circuits to
   # renamed fixtures: rocprofv3 aborted on every pass, stderr went to
   # /dev/null, and the summary reported "wins 18/18" over a silently
   # truncated corpus. A missing input must not look like a clean result.
   if [ ! -f "$c" ]; then
     echo "  *** FIXTURE MISSING: $c -- aborting run ***" >&2
     exit 1
   fi
   ```

   That truncation was not a wash: the six dropped QV circuits are the corpus's
   *weakest* wins, so their absence moved the median from 0.670 to 0.518. A
   silent loss of data flattered the result by 23 %.

### 14.2 The result

**Mean 0.626, median 0.670, wins 26/26.** Every circuit dispatches
`clifft_v2_spec`; not one falls back to the interpreter.

| circuit | V2 (µs) | SVM (µs) | V2/SVM | speedup |
|---|---:|---:|---:|---:|
| `surface_d11_t19` | 20,423 | 79,842 | **0.256** | 3.9× |
| `surface_d11_t15` | 19,589 | 75,947 | 0.258 | 3.9× |
| `surface_d9_t19` | 11,926 | 45,208 | 0.264 | 3.8× |
| `surface_d9_t15` | 11,369 | 41,325 | 0.275 | 3.6× |
| `surface_d7_t19` | 6,247 | 20,152 | 0.310 | 3.2× |
| `qv10` | 1,386 | 4,350 | 0.319 | 3.1× |
| `surface_d11_t10` | 37,986 | 81,369 | 0.467 | 2.1× |
| `surface_d9_t10` | 21,632 | 44,094 | 0.491 | 2.0× |
| `circuit_d3_p0.001` | 221 | 434 | 0.509 | 2.0× |
| `surface_d7_t15` | 11,225 | 21,998 | 0.510 | 2.0× |
| `surface_d7_t10` | 11,044 | 20,674 | 0.534 | 1.9× |
| `four_t` | 13.1 | 22.3 | 0.587 | 1.7× |
| `surface_d7_t5` | 2,051 | 3,372 | 0.608 | 1.6× |
| `qv20_L8_seed42` | 1,583,246 | 2,162,954 | 0.732 | 1.4× |
| `qv21_L8_seed42` | 2,977,386 | 4,045,316 | 0.736 | 1.4× |
| `frame_h` | 12.1 | 16.3 | 0.742 | 1.3× |
| `cultivation_d5` | 16,421 | 20,901 | 0.786 | 1.3× |
| `circuit_d5_p0.005` | 8,727 | 10,714 | 0.815 | 1.2× |
| `circuit_d5_p0.003` | 8,809 | 10,516 | 0.838 | 1.2× |
| `circuit_d5_p0.002` | 8,730 | 10,319 | 0.846 | 1.2× |
| `circuit_d5_p0.001` | 8,600 | 10,158 | 0.847 | 1.2× |
| `qv20_seed42` | 1,804,649 | 2,123,250 | 0.850 | 1.2× |
| `circuit_d5_p0.0005` | 8,643 | 10,102 | 0.856 | 1.2× |
| `qv24_L4_seed42` | 7,241,815 | 8,211,443 | 0.882 | 1.1× |
| `qv22_L6_seed42` | 3,179,445 | 3,243,842 | 0.980 | 1.0× |
| `qv23_L5_seed42` | 6,086,167 | 6,146,952 | 0.990 | 1.0× |

The spread — 3.9× down to 1.01× — is the chapter's real subject. A single
average would hide the entire mechanism.

### 14.3 Where the time goes: the scalar pipe

The strongest and most consistent signal in the corpus is **not** the vector
instruction count. It is the scalar one.

| circuit | V2/SVM time | VALU ratio | **SALU ratio** | SALU reduction |
|---|---:|---:|---:|---:|
| `four_t` | 0.587 | 0.432 | **0.110** | 9.1× |
| `surface_d9_t19` | 0.264 | 0.482 | 0.137 | 7.3× |
| `surface_d11_t15` | 0.258 | 0.482 | 0.138 | 7.3× |
| `surface_d11_t19` | 0.256 | 0.490 | 0.138 | 7.2× |
| `circuit_d3_p0.001` | 0.509 | 0.817 | 0.150 | 6.7× |
| `circuit_d5_p0.001` | 0.847 | 0.639 | 0.156 | 6.4× |
| `qv10` | 0.319 | 0.445 | 0.198 | 5.1× |
| `frame_h` | 0.742 | 0.429 | 0.218 | 4.6× |
| `qv20_seed42` | 0.850 | 0.821 | 0.345 | 2.9× |
| `qv24_L4_seed42` | 0.882 | 0.863 | **0.361** | 2.8× |

**On 26 of 26 circuits the SALU ratio is below the VALU ratio** — the scalar
count falls faster than the vector count, without exception. The range is
0.110–0.361: V2 issues between 2.8× and 9.1× fewer scalar instructions than the
SVM interpreter.

On the surface and `circuit_d5` families the reduction is larger in *absolute*
terms as well, by a factor of 3.8×–7.3×:

| circuit | ΔSALU | ΔVALU | \|ΔSALU\|/\|ΔVALU\| |
|---|---:|---:|---:|
| `circuit_d3_p0.001` | −6 M | −0.8 M | **7.31** |
| `surface_d11_t10` | −17,152 M | −2,832 M | 6.06 |
| `surface_d7_t15` | −4,603 M | −761 M | 6.05 |
| `cultivation_d5` | −4,053 M | −693 M | 5.85 |
| `circuit_d5_p0.001` | −2,029 M | −352 M | 5.76 |
| `surface_d11_t19` | −8,691 M | −2,275 M | 3.82 |

On the **QV family this inverts**: `qv20_seed42` removes 9,094 M scalar
instructions against 37,752 M vector ones, a ratio of 0.24. The scalar *ratio*
still improves more (0.345 vs 0.821), but the QV circuits are so vector-dominated
in absolute terms that the vector reduction is the larger number. Both statements
are true of the same data, and which one matters depends on which pipe is the
limiter — a distinction §14.5 develops.

**What those scalar instructions were.** The interpreter's inner loop is

```c
for (u32 pc = 0; pc < num_instrs; ++pc) {
    CV2Instr ins = instrs[pc];      // a load
    u32 k = st->active_k;           // another load
    switch (ins.opcode) { ... }     // a jump table
```

Per bytecode instruction that is: an address computation (`pc * 40`, needing a
64-bit multiply — see §7.3), a `global_load`, a `v_readfirstlane` to move the
operand to the scalar unit, an `s_load` of `active_k`, a bounds compare, and a
jump through a switch table. None of it computes an amplitude. V2 emits the call
directly with its operands as literals, and all of it disappears — which is
exactly what the eight microbenchmarks of §7 predicted, case by case.

> **A retraction, because an earlier draft of this report got the mechanism
> backwards.** §7 previously described this as work *moved* from the vector pipe
> to the scalar pipe — "the same computation, once per wavefront instead of once
> per lane." That is wrong. Counting scalar instructions in the same sixteen
> `.s` files that produced §7's VALU numbers shows SALU falling in all eight
> cases (1.26×–3.09×), and the corpus shows it falling *faster* than VALU on
> 26 of 26 circuits. Nothing migrates. Both pipes issue less, and the scalar
> pipe issues much less. The claim survived several drafts because it was
> plausible and because `stats.csv` had a `v_alu` column and no `s_alu` column —
> the falsifying number was one `grep` away and structurally invisible.

### 14.4 Why the speedup varies: instruction mix

§7 ended with a taxonomy — what specialization folds and what it cannot. The
corpus tests it.

The **surface family** (0.256–0.534) is dominated by frame ops and dormant
measurements, precisely the S1 and S2 classes that fold hardest (5.50× and 4.53×
on VALU, 18 branches → 0). The **`circuit_d5` family** (0.786–0.856) is 19.1 %
noise ops by call count — the S7 class, which folds at 1.02× — and those ops are
individually the most expensive in the ISA (447 interpreter-form instructions
against 136 for a frame op), so their share of *time* far exceeds their share of
*call sites*. The prediction §7.9 made from a single microbenchmark, before any
of these circuits were measured, was that `circuit_d5` would be the corpus's
weakest coop-tier result. It is.

The **QV family** (0.732–0.990) is weak for a different reason, developed next.

### 14.5 Why the QV family resists: the resident pool shrinks

The QV circuits have *better* VALU and SALU ratios than `circuit_d5` (0.82–0.86
and 0.34–0.36 against 0.64–0.66 and 0.16), yet worse time ratios. Instruction
count is not what limits them.

**The resident pool shrinks with rank.** Global-tier kernels size a persistent
workgroup pool from an HBM budget: each workgroup owns an amplitude slice plus a
half-size scratch — 12 bytes per amplitude, since `GpuComplex` is
`{float re; float im;}` — against a 32 GB budget, capped at 2,048 and rounded
down to a multiple of `kNumXCDs = 8` (`v2_kernel.cc:436-445`):

```cpp
const uint64_t amp = 1ull << flat.peak_rank;
const uint64_t bytes_per_wg = amp * sizeof(GpuComplex) + (amp / 2) * sizeof(GpuComplex);
const uint64_t budget = 32ull << 30;  // 32 GB
uint64_t wgs = budget / bytes_per_wg;
if (wgs < 1) wgs = 1;               // rank 26+: at least one resident wg
if (wgs > 2048) wgs = 2048;
global_grid_wgs = static_cast<uint32_t>(wgs);
if (global_grid_wgs > kNumXCDs) global_grid_wgs -= global_grid_wgs % kNumXCDs;
```

Evaluating that arithmetic against the grids `rocprofv3` actually recorded:

| rank | bytes/wg | predicted wgs | measured wgs |
|---|---:|---:|---:|
| 20 | 12 MB | 2,048 (capped) | **2,048** ✓ |
| 21 | 24 MB | 1,360 | **1,360** ✓ |
| 22 | 48 MB | 680 | **680** ✓ |
| 23 | 96 MB | 336 | **336** ✓ |
| 24 | 192 MB | 168 | **168** ✓ |

Five predictions, five exact matches. Every added qubit halves the pool. By rank
24 there are 168 workgroups on a device with 256 CUs — **the machine cannot be
filled**, and no amount of instruction-level improvement changes that. This is
the clearest single explanation for the QV band, and it is structural rather
than incidental: it follows from the budget constant and the rank, both known
before the kernel launches.

A second observation is worth recording as a *lead* rather than a finding. The
three weakest circuits — `qv22_L6` (0.980), `qv23_L5` (0.990), `qv24_L4` (0.882)
— are exactly the three where V2's specialized kernel is allocated **64 VGPRs**,
against 52–56 on the ranks that win comfortably (`qv20` 52, `qv20_L8` 56,
`qv21_L8` 52), and where V2's scratch allocation jumps to 448–576 B from 96–112 B.
64 VGPRs is a plausible occupancy cliff, and the correlation across six circuits
is perfect. But six points is not an experiment, VGPR count and rank are
confounded here, and this counter set cannot distinguish "allocated but not
spilled" from "actually spilling." §16 records it as the report's strongest open
lead, not as a mechanism.

> **What this section does not claim.** An earlier draft argued the QV band was
> explained by an occupancy inversion — "V2 runs a third as many waves, each 3×
> longer" — computed as `SQ_WAVE_CYCLES / SQ_WAVES`. That quotient is not
> admissible here. `SQ_WAVES` is collected in pass `pmcA` and `SQ_WAVE_CYCLES`
> in pass `pmcC`: **different executions of the kernel**. And `SQ_WAVES` does not
> reconcile with launch geometry on precisely these circuits — SVM launches an
> identical 512-workgroup, 256-thread grid on `qv20`, `qv21_L8` and `qv22_L6`
> and reports 8,832, 10,240 and 8,000 waves respectively, where the geometry
> implies 2,048 in all three. On the other 20 circuits `SQ_WAVES` equals
> workgroups × 4 exactly. Whatever the rank-≥20 counts measure, dividing a
> pass-C cycle count by a pass-A wave count that disagrees with the launch
> geometry does not measure per-wave cost. The claim was removed rather than
> repaired.

### 14.6 What V2 does *not* spend

Three counters are worth reporting for what they rule out.

**MFMA = 0 on all 52 cells.** 26 circuits × 2 backends, none missing. Neither
backend issues a single matrix-core instruction. The workload is a butterfly
reduction over amplitude pairs — `cadd`/`csub`/`cmul` on 2-element complex values
with a strided access pattern — not a GEMM. Any proposal to bring the matrix
cores to bear on this workload is unsupported by this corpus, and the zero is
measured rather than assumed. (An earlier draft could only claim 51 of 52:
`qv24_L4_seed42`'s SVM counter block was missing from the truncated job 50785,
and was recorded as *unmeasured* rather than rounded down to zero.)

**LDS is cheaper on V2 by construction, not by optimization.**

| tier | V2 LDS | SVM LDS | ratio |
|---|---:|---:|---:|
| coop | 13,312 B | 23,040 B | 0.578 |
| global | 1,024 B | 8,704 B | **0.118** |
| register | 0 B | 0 B | — |

The coop figure is the sum of the declared arrays — `lds_v[1024]` and
`lds_red_scratch[512]` at 8 B each is 12,288 B, plus `lds_state` — and the
global-tier kernel declares neither, because at rank > 10 the amplitudes live in
HBM. Its 1,024 B is `lds_state` and `lds_shot` alone
(`v2_specializer.cc:204-205`). The 8.5× gap is therefore a consequence of the
tier's design, not a tuning result.

The dynamic counter agrees. `SQ_INSTS_LDS` on the coop tier runs 0.60–0.77× SVM;
on the register tier V2 executes **zero** LDS instructions against SVM's 25,280.
On the surface family's coop and global circuits, however, V2 issues *more*
(1.13–1.25×) — folded axes turn some LDS traffic into direct indexed access, and
the direction of that trade depends on the circuit.

**Scratch, on the register tier, is 4.3× smaller**: 656–1,040 B against SVM's
uniform 4,480 B. The SVM kernel must provision for the worst opcode it might
interpret; V2 provisions for the opcodes this circuit actually contains.

### 14.7 A sanity check: does the counter model explain the time?

If the story above is right, a single derived quantity should predict the
measured ratio. `SQ_BUSY_CYCLES` — cycles during which the SQ has work — is the
natural candidate, and unlike §14.5's rejected quotient it is a ratio of the same
counter between two arms, not a division across passes.

Across all 26 circuits, the busy-cycle ratio against the kernel-time ratio gives
**Pearson r = 0.942**, median relative error **5.5 %**, mean **9.0 %**.

The residual is informative. The six largest errors are, in order,
`circuit_d5_p0.0005` (28.2 %), `circuit_d5_p0.001` (27.2 %),
`circuit_d5_p0.002` (26.4 %), `circuit_d5_p0.003` (25.0 %),
`circuit_d5_p0.005` (21.2 %) and `cultivation_d5` (17.9 %) — **the entire d5
family, and nothing else**. §14.8 is about the same six circuits, and the two
observations are plausibly the same phenomenon seen through two counters.

### 14.8 One result that does not fit

The `circuit_d5` family wins, but its **L2 hit rate is still depressed**: 91.1–92.1 %
for V2 against 97.6–98.4 % for SVM, and `cultivation_d5` shows 90.9 % against
98.9 %. This is a residue of the regression documented in §11.1 — the same
circuits, the same direction, at a fraction of the magnitude (pre-fence the gap
was 71.5 % against 98.0 %, `VERIFIED_FACTS.md:158`).

They win anyway, on the strength of a 0.64 VALU ratio and a 0.16 SALU ratio. But
a 6-point L2 deficit on the one circuit family V2 finds hardest, showing up in
the same six circuits as §14.7's busy-cycle residual, is not explained by
anything in this chapter. §16 records it as open. The honest summary is that V2
wins these circuits *despite* a memory-system disadvantage it did not have to
have.

A second unexplained result, in the opposite direction: on the two smallest
register-tier circuits V2's L2 hit rate is far *worse* than SVM's — `four_t`
52.3 % against 74.4 %, `frame_h` 52.2 % against 66.8 % — and V2 wins both anyway
(0.587, 0.742). At 13 µs and 12 µs these kernels are small enough that cache
behaviour is dominated by cold misses on first touch, and the hit *rate* is
computed over a small denominator. The result is recorded because it was
measured, not because it is understood.
