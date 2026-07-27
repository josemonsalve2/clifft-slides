# From Interpreter to Specializer

## The clifft GPU Backend: SVM, Hybrid, MLIR V1, and the V2 Rebuild

**A comprehensive technical report**

| | |
|---|---|
| Hardware | AMD Instinct MI350X (`gfx950`), node `smci350-rck-g03-f13-21` |
| Software | ROCm 7.2.3, LLVM `upstream_05082025` |
| Repository | `unitaryfoundation/clifft`, branch `mlir-v2` |
| Benchmark corpus | 26 circuits, run `20260726T182433Z_report-final-postdust` (SLURM 50469) |
| Fact ledger | `V2_performance/VERIFIED_FACTS.md` |
| Date | 2026-07-26 |

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
| **V2** | per-circuit C, specialized operands, runtime loops | yes | C→amdgcn (no HIP) | **1.5–3.9× faster than SVM** on 20 of 26 circuits |

### 1.2 The headline number

Across the 26-circuit corpus, comparing GPU kernel time (not host wall time),
V2 versus the SVM interpreter:

| regime | circuits | V2/SVM ratio | reading |
|---|---|---|---|
| Large surface codes, global tier | 5 | **0.255 – 0.298** | V2 is **3.4–3.9× faster** |
| Mid-size surface + QV10, coop tier | 8 | 0.310 – 0.607 | V2 is **1.6–3.2× faster** |
| Quantum-volume, global tier rank 20–24 | 6 | 0.723 – 0.990 | V2 wins 1–38%, shrinking with rank |
| `circuit_d5` family, coop tier | 6 | **1.443 – 1.459** | V2 is **1.44× slower** |
| register tier | 4 | 0.497 – 0.626 | V2 is **1.6–2.0× faster** |

Twenty of twenty-six circuits are wins. All 26 are **byte-exact** against the
SVM interpreter and against the f64 CPU reference (modulo documented f32/f64
branch divergence, §12).

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
falls 3–9× on every circuit V2 wins** — that is the interpreter's `switch`
dispatch disappearing from the scalar unit. (§14.2)

**(d) V2's one loss is a correctness failure wearing a performance costume.**
The `circuit_d5` family's specialization fails V2's own correctness gate, so
those six circuits run the *interpreter*, and the 1.44× measures V2's
interpreter against SVM's. Every circuit V2 wins ran `clifft_v2_spec`; every
circuit V2 loses ran `clifft_v2_coop`. The correlation is perfect. (§11.1)

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
- **Straight-lining noise operations does not help.** Measured gain: 1.07× on
  code size, 1.02× on VALU (§7.7).
- **MFMA is inapplicable.** `SQ_INSTS_MFMA = 0.0` on all 26 circuits in both
  backends. The workload is a butterfly reduction over amplitudes, not a GEMM.
  Any claim that the matrix cores can be brought to bear here is unsupported by
  this corpus. (§14.6)

---
