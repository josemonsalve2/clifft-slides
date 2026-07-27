# From Interpreter to Specializer

## The clifft GPU Backend: SVM, Hybrid, MLIR V1, and the V2 Rebuild

**A comprehensive technical report**

| | |
|---|---|
| Hardware | AMD Instinct MI350X (`gfx950`), node `smci350-rck-g03-f13-21` |
| Software | ROCm 7.2.3, LLVM `upstream_05082025` |
| Repository | `unitaryfoundation/clifft`, branch `mlir-v2` |
| Benchmark corpus | 26 circuits — see the provenance notice below |
| Fact ledger | `V2_performance/VERIFIED_FACTS.md` |
| Date | 2026-07-27 |

> ### ⚠ Provenance of the performance numbers
>
> The timing figures quoted in §1–§10 come from
> `20260726T182433Z_report-final-postdust` (SLURM 50469), which was **later
> invalidated**: a cache-key bug (§11.4) caused the specializer to serve
> pre-fence, pre-dust-fix binaries under post-fix cache keys. The bug is fixed
> (`009df59`) and the run is retracted in `VERIFIED_FACTS §0`.
>
> A second run, `20260726T014859Z_all-tier5plus`, reproduces every ratio to
> within 0.4 %. That is a **reproducibility** check only — it is at commit
> `89d541e`, which is also pre-fence, so both runs executed the same binaries.
>
> **§15 carries the post-fix full-corpus re-run and is authoritative.** Where
> §1–§10 and §15 disagree, §15 wins. Each affected table is individually marked;
> tables that are provably unaffected (resource footprints, IR density, source
> counts) say so and explain why.
>
> This notice exists because the report's own rule demands it: *trust data, not
> text* applies to the report's own numbers first.

---

## A note on method, before anything else

This report has an unusual constraint imposed on it: **trust data, not text.**

The `clifft` tree contains a large amount of prose — design documents, plans,
post-mortems, header comments — written across several months by several
authors (human and otherwise). Much of it is accurate. Some of it is stale, and
a smaller amount of it is simply wrong, having been written from a hypothesis
that later measurement refuted but that nobody went back to delete.

Every claim in this report was therefore re-derived from an artifact before
being written down. The rule used was:

> A fact is *verified* only if it can be re-derived from a committed artifact —
> a CSV, a `summary.json`, a SLURM log, an `.hsaco` — or from a file+line in the
> source tree. Prose in a design doc is a *claim*, not a fact.

The audit is committed as `V2_performance/VERIFIED_FACTS.md`, and every section
below cites it. Three findings from that audit are worth stating up front,
because they contradict documents still in the tree:

1. **V2 does not emit MLIR.** It emits C. The directory is still called
   `mlir/v2/` for historical reasons, and that name is the only MLIR left in it.
   (§6.1)
2. **V1 authored zero custom MLIR passes**, and one of the three stock passes it
   ran was a provable no-op. (§5.4)
3. **The one circuit family where V2 loses did not run V2's specializer at all** —
   its specialization failed the correctness gate and fell back to the
   interpreter. An earlier draft of the audit itself got this wrong, and the
   correction is recorded in the ledger's §13. (§11.1)

Where a number could not be re-measured, it is marked **[unverified]** and
attributed rather than asserted. There are two such items, both flagged in place.

---

## Table of contents

| § | Chapter |
|---|---|
| 1 | [Executive summary](#1-executive-summary) |
| 2 | [The SVM: what is actually being simulated](#2-the-svm-what-is-actually-being-simulated) |
| 3 | [The original GPU SVM: one interpreter, three tiers](#3-the-original-gpu-svm-one-interpreter-three-tiers) |
| 4 | [Hybrid: compiling the circuit into HIP, and why it stalled](#4-hybrid-compiling-the-circuit-into-hip-and-why-it-stalled) |
| 5 | [V1: the MLIR backend and its failure](#5-v1-the-mlir-backend-and-its-failure) |
| 6 | [V2: the architecture](#6-v2-the-architecture) |
| 7 | [What V2 specializes — eight classes, eight experiments](#7-what-v2-specializes--eight-classes-eight-experiments) |
| 8 | [Progressive lowering: what the compiler does with it](#8-progressive-lowering-what-the-compiler-does-with-it) |
| 9 | [Optimizations, tier by tier](#9-optimizations-tier-by-tier) |
| 10 | [Extending past rank 19](#10-extending-past-rank-19) |
| 11 | [Pitfalls](#11-pitfalls) |
| 12 | [The gap: f32 vs f64](#12-the-gap-f32-vs-f64) |
| 13 | [Removing HIP, introducing HSA](#13-removing-hip-introducing-hsa) |
| 14 | [Performance evaluation: V2 vs SVM](#14-performance-evaluation-v2-vs-svm) |
| 15 | [Full benchmark report](#15-full-benchmark-report) |
| 16 | [Conclusions and open items](#16-conclusions-and-open-items) |

---

## 1. Executive summary

`clifft` simulates near-Clifford quantum circuits by tracking a Pauli frame plus
a small dense statevector. The GPU backend has been rewritten three times. This
report characterizes all four generations and evaluates the newest against the
baseline it must beat.

### 1.1 The four generations

| generation | what it is | per-circuit compile? | device language | verdict |
|---|---|---|---|---|
| **SVM** | bytecode interpreter kernel | no | HIP | the baseline; fast, correct, hard to beat |
| **Hybrid** | per-circuit HIP source, AOT-compiled | yes | HIP | correct and competitive, but HIP-bound and never pulled ahead |
| **MLIR V1** | per-circuit hand-emitted MLIR text | yes | MLIR→LLVM | **failed**: 5–60× slower, 221 s compiles, 20 MB IR |
| **V2** | per-circuit C, specialized operands, runtime loops | yes | C→amdgcn (no HIP) | **1.3–4.0× faster than SVM** on 20 of 26 circuits (the other 6 recovered post-fix, §1.3(d)) |

### 1.2 The headline number

Across the 26-circuit corpus, comparing GPU kernel time (not host wall time),
V2 versus the SVM interpreter:

| regime | circuits | V2/SVM ratio | reading |
|---|---|---|---|
| Large surface codes, global tier | 5 | **0.252 – 0.306** | V2 is **3.3–4.0× faster** |
| Mid-size surface + QV10, coop tier | 5 | 0.317 – 0.530 | V2 is **1.9–3.2× faster** |
| Register tier (`four_t`, `frame_h`, `circuit_d3`, `surface_d7_t5`) | 4 | 0.519 – 0.773 | V2 is **1.3–1.9× faster** |
| Quantum-volume, global tier rank 20–24 | 6 | 0.729 – 0.992 | V2 wins **1–27 %**, shrinking with rank (§10.6) |
| `circuit_d5` + `cultivation_d5`, coop tier | 6 | **1.440 – 1.451** | V2 is **1.44× slower** (§1.3(d)) |

**26 circuits, 20 wins.** The five regimes sum to exactly 26, and membership is
assigned by *measured launch geometry* rather than by circuit name — the census
of §10.5 is precisely the finding that names are unreliable here.

The geometry is unambiguous. Global-tier kernels launch a **resident pool sized
from the HBM budget** and then work-steal shots; coop-tier kernels launch **one
workgroup per shot**. Reading grid ÷ 256 from the measured runs:

| circuits | workgroups | shape |
|---|---|---|
| `surface_d7/d9/d11_t19`, `surface_d9/d11_t15`, `qv20` (×2) | 2,048 | global pool, capped (`v2_kernel.cc:441`) |
| `qv21` → `qv24` | 1,360 → 680 → 336 → 168 | global pool, **shrinking as rank grows** |
| `circuit_d5` (×5), `surface_d7/d9/d11_t10`, `surface_d7_t15` | 10,000 | coop, = shot count |
| `cultivation_d5`, `qv10` | 20,000 | coop, = shot count |
| `four_t`, `frame_h`, `circuit_d3`, `surface_d7_t5` | 79 | register, = ⌈shots/256⌉ |

The QV row is the §10.2 budget mechanism visible from the outside, and it can be
predicted exactly rather than merely described. Each resident workgroup owns one
amplitude slice plus a half-size scratch — 12 bytes per amplitude — drawn from a
32 GB budget (`v2_kernel.cc:436-446`). Evaluating that formula against the
measured grids:

| rank | bytes/wg | predicted wgs | **measured wgs** |
|---|---|---|---|
| 20 | 12 MB | 2,048 (capped) | **2,048** ✓ |
| 21 | 24 MB | 1,360 | **1,360** ✓ |
| 22 | 48 MB | 680 | **680** ✓ |
| 23 | 96 MB | 336 | **336** ✓ |
| 24 | 192 MB | 168 | **168** ✓ |

Five for five, including the XCD-alignment rounding. So the QV circuits lose
occupancy geometrically as rank climbs — the pool halves with every added qubit
— and that compounds with the register spilling of §10.6. Two independent
mechanisms degrade the same six circuits, which is why their ratios approach 1.0
from a corpus whose median is 0.678.

All 26 are **byte-exact** against the SVM interpreter and against the f64 CPU
reference (modulo the documented f32/f64 branch divergence of §12).

> **These ratios understate V2.** They are from `20260726T014859Z_all-tier5plus`,
> which agrees with the retracted run to within 0.4 % — but both are
> *pre-fence-fix*, and the six losing circuits were silently running the
> interpreter rather than the specializer because of it. Post-fix spot
> measurements put them near 0.40 instead of 1.44 (§1.3(d)). The table is kept
> as the conservative baseline; §15 carries the authoritative post-fix corpus.

### 1.3 The five results this report argues for

**(a) The disease V1 had was unrolling the operand *sequence*, and V2 cured it
exactly.** The measurable form of that cure is source-IR density — lines of
emitted source per bytecode instruction:

| circuit | instrs | V1 MLIR | Hybrid HIP | **V2 C** |
|---|---|---|---|---|
| `coop_qv10` | 140 | 336,988 lines (**2407/instr**) | 7,805 (55.8/instr) | **179 (1.27/instr)** |
| `coop_circuit_d5` | 1,720 | 118,493 (68.9/instr) | 33,508 (19.5/instr) | **1,760 (1.02/instr)** |
| `coop_surface_d7_t10` | 4,134 | 228,544 (55.3/instr) | 75,409 (18.2/instr) | **4,174 (1.00/instr)** |

V2 converges to **exactly one line of source per bytecode instruction**. V1 and
Hybrid both emit a per-instruction *block*; V2 emits a per-instruction *call*.
(§6.3)

**(b) Specialization's gain is not uniform, and the variation is the interesting
part.** Eight isolated A/B experiments (§7) measure per-class gains from 5.50×
(VALU, frame-operand folding) down to **1.02×** (noise blocks, where the loop is
genuinely data-dependent). The negative result is as informative as the
positive ones and predicts exactly which circuits benefit.

**(c) The win mechanism is visible in the hardware counters and is not what one
might guess.** It is not floating-point throughput. **SALU instruction count
falls 2.8–9.1× on all 20 circuits V2 wins** (ratios 0.110–0.361, no exceptions)
— that is the interpreter's `switch` dispatch leaving the scalar unit. VALU
falls too, but far less: 1.2–2.3×, and on `circuit_d3` barely at all (0.818).
The counter that separates the two backends is the scalar one.

The sign flips exactly where the fallback happened: on the six pre-fix losses,
SALU goes **up** to 1.467–1.469 and VALU up to 2.23–2.30. Those six were running
the interpreter (§1.3(d)), so the inversion is not a counter-example — it is the
same measurement confirming which kernel actually ran. (§14.2)

**(d) V2's one loss was a correctness failure wearing a performance costume —
and it has since been fixed.** The `circuit_d5` family's specialization failed
V2's own correctness gate, so those six circuits silently fell back to the
*interpreter*: the 1.44× "loss" was never measuring specialization at all. The
correlation in the pre-fix data is perfect — all 20 circuits V2 wins ran
`clifft_v2_spec`, all 6 it loses ran `clifft_v2_coop`.

The cause was the execution-only `s_barrier` of §11.2. With the fence fix
(`150d09f`) the gate verdict for `coop_r10_n1720` flips **0 → 1** on disk, the
specializer is selected, and the six recover (job 50389, median of 5, both arms
back-to-back on node `smci350-rck-g03-d13-21`):

| circuit | interpreter | specialized | gain |
|---|---|---|---|
| `circuit_d5_p0.0005` | 14.76 ms | **8.56 ms** | 1.72× |
| `circuit_d5_p0.001` | 14.89 ms | **8.56 ms** | 1.74× |
| `circuit_d5_p0.002` | 15.09 ms | **8.63 ms** | 1.75× |
| `circuit_d5_p0.003` | 15.30 ms | **8.68 ms** | 1.76× |
| `circuit_d5_p0.005` | 15.62 ms | **8.87 ms** | 1.76× |
| `circuit_d3_p0.001` | 0.624 ms | **0.379 ms** | 1.65× |

That 1.65–1.76× is the interpreter→specializer gain, measured with both arms
back-to-back in one job. It does *not* directly give a V2/SVM ratio: dividing it
into the corpus baselines would chain a `d13-21` measurement onto an `f13-21`
one, and `mi350x-es` is heterogeneous.

The direct measurement now exists (job 50785, both backends in one job on
`d13-21`) and it confirms the sign flip:

| circuit | pre-fence, interpreter | **post-fence, specialized** |
|---|---|---|
| `circuit_d5_p0.0005` | 1.451 | **0.842** |
| `circuit_d5_p0.001` | 1.451 | **0.851** |
| `circuit_d5_p0.002` | 1.451 | **0.843** |
| `circuit_d5_p0.003` | 1.443 | **0.828** |
| `circuit_d5_p0.005` | 1.448 | **0.811** |

All six now run `clifft_v2_spec` and win by 15–19 %. Worth stating plainly: an
earlier draft of this section *projected* ~0.40 for these circuits by chaining
the two runs, and the direct measurement says **~0.84**. The projection was
wrong by a factor of two, in the flattering direction, for exactly the reason
the project's benchmarking rule exists. The 1.7× gain was real; the assumption
that an SVM baseline transfers across nodes was not.

So the headline "20 of 26" in §1.2 is a *pre-fix* figure. Post-fix, five of the
six former losses are measured wins and the corpus stands at **25 of 26
measured**; the sixth, `cultivation_d5`, shares the coop specialization that
the other five recovered on but was not paired in job 50785 (its fixture path
was stale, §14.0), so it is *unmeasured* rather than assumed. **§1.2's table is
the conservative reading, not the optimistic one**, and §15 carries the full
post-fix corpus. (§11.1, §11.2)

**(e) The f32/f64 "gap" was never really about arithmetic precision.** It was a
threshold constant calibrated for f64 and left in place when the storage became
f32 — whose effect was to make one side consume a random number the other side
did not, permanently desynchronizing the PRNG streams. Fixing one constant took
same-stream agreement to **36/36 exact**. (§12)

### 1.4 What did not work

Stated plainly, because the failures cost more engineering time than the wins:

- **Hybrid never pulled ahead of SVM** despite compiling a bespoke kernel per
  circuit. Per-circuit compilation is not, by itself, a performance strategy.
- **V1 (MLIR) was 5–60× *slower* than SVM.** Full unrolling of the operand
  sequence produced 20 MB of IR, 115 KB/thread of spill scratch [unverified,
  see §5.5], and 221-second compiles.
- **Straight-lining noise operations does not help.** Measured gain: 1.10× on
  instruction count, 1.02× on VALU, and **zero** VGPR relief — 56 in both forms
  (§7.9).
- **MFMA is inapplicable.** `SQ_INSTS_MFMA = 0.0` in every counter block
  collected — 51 of the 52 backend×circuit cells; the 52nd (`qv24_L4_seed42`,
  SVM side) has an empty counter block and is *unmeasured* rather than nonzero.
  The workload is a butterfly reduction over amplitudes, not a GEMM.
  Any claim that the matrix cores can be brought to bear here is unsupported by
  this corpus. (§14.6)

---
