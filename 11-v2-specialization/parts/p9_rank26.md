## 10. Extending past rank 19

The global tier's cap was `kGlobalMaxPeakRank = 19` for the entire life of the
GPU backend. Raising it to 26 required exactly one substantive change, and
finding a benchmark that could exercise it required a separate investigation
that produced a more interesting result than the change itself.

### 10.1 Why the cap could not simply be raised

The blocker was an addressing convention, not a memory limit. Every global-tier
kernel strided its per-workgroup HBM slice by **`kGlobalMaxAmplitudes`** — the
*cap* — rather than by the circuit's actual amplitude count:

```c
// before: every slot is 2^kGlobalMaxPeakRank amplitudes wide, regardless of circuit
CV2Complex* v = global_v + (u64)slot * kGlobalMaxAmplitudes;
```

At rank 19 that reserves 6 MB per resident workgroup, which is fine. At rank 26
it reserves **512 MB per workgroup** — for a rank-3 circuit as much as for a
rank-26 one. With a 2,048-workgroup pool that is a terabyte of HBM to run
`frame_h`. The cap was therefore self-limiting: raising the constant made *every
circuit* unaffordable, not just the large ones.

### 10.2 The fix (`b266f80`)

Derive every per-slot stride from `flat.peak_rank`. The commit had to change it
in **four** places that must agree with each other:

| File | What it addresses |
|---|---|
| `hip_sampler.hip` | `sample_kernel_global_coop` device addressing **and** the matching host `v_count`/`scratch_count` allocation |
| `emit_global_kernel.cc` | the Hybrid compiled-codegen kernel |
| `mlir_emit.cc` | the MLIR V1 global kernel |
| `v2_kernel.cc` | already rank-derived on the device side; only the resident-pool sizing needed the same treatment |

The `emit_global_kernel.cc` change fixed a **latent out-of-bounds bug** that had
nothing to do with the cap:

> `scratch_v` used the FULL `v` stride while the host allocated scratch at HALF
> the amplitude count.

Scratch is only ever `2^(k-1)` amplitudes — it holds the fold half — so the host
allocated half. The device strided as if it were full. Any global-tier circuit
using more than one workgroup was addressing past its scratch slice. It had
never been caught because the overrun landed in the *next* workgroup's scratch,
which is written before it is read on the next op.

The second half of the fix is a **HBM budget** replacing the fixed pool
(`v2_kernel.cc:432-446`):

```cpp
// Resident pool sized to a fixed HBM budget: each workgroup owns one
// amplitude slice (1<<peak_rank) + a half-size scratch, i.e. 12 bytes
// per amplitude. At rank 19 that is 6 MB/wg; at rank 26 it is 768 MB/wg.
// The 32 GB budget keeps the pool sane at both ends on a 288 GB MI355X.
const uint64_t amp = 1ull << flat.peak_rank;
const uint64_t bytes_per_wg = amp * sizeof(GpuComplex) + (amp / 2) * sizeof(GpuComplex);
const uint64_t budget = 32ull << 30;  // 32 GB
uint64_t wgs = budget / bytes_per_wg;
if (wgs < 1) wgs = 1;               // rank 26+: at least one resident wg
if (wgs > 2048) wgs = 2048;
global_grid_wgs = static_cast<uint32_t>(wgs);
// Prefer a multiple of the XCD count when the budget allows it, so the
// resident pool spreads evenly; never inflate past the budget.
if (global_grid_wgs > kNumXCDs) global_grid_wgs -= global_grid_wgs % kNumXCDs;
```

So the pool now **shrinks as rank grows** instead of the allocation exploding.
At rank 19 the pool clamps at 2,048; at rank 26 it falls to a handful of
resident workgroups. The XCD-alignment step (`kNumXCDs = 8`) keeps the pool
spread evenly across the device's eight compute dies.

One constraint worth recording: the cap is kept `≤ 30` so `1u << rank` stays
inside a `u32`.

<figure>
<img src="diagrams/hbm-budget-pool.svg" alt="Resident pool size vs circuit rank under the 32 GB budget" width="100%">
<figcaption><b>Figure 10.1</b> — Resident workgroup pool under the 32 GB budget.
Below rank ~19 the pool is clamped at 2,048 workgroups; above it, the pool
halves with every rank while per-workgroup memory doubles. The product is
constant, which is the point.</figcaption>
</figure>

### 10.3 The gate had the same bug

The specializer's correctness gate dispatched a **fixed 64-workgroup pool and a
fixed 5,000 shots** regardless of rank (`6960527`). At rank ≤ 19 that is
harmless. At rank 20+ it means the gate validates on a pool tens of times
narrower than the real dispatch, while per-shot cost grows as `2^peak_rank`:

> The gate for `qv20` had not finished after 10 minutes — longer than the
> benchmark it gates.

Fixed by giving the gate the *same* budget formula (capped at 512 — it only
needs enough parallelism to finish promptly) and tapering the shot sample above
rank 19: 1,000 shots at rank 20–21, 250 at rank 22+. The justification is
statistical rather than budgetary — *"a divergence needs far fewer shots to
surface at high rank, since each shot touches millions of amplitudes"* — and
`V2_GATE_SHOTS` overrides it for paranoid validation. Verdicts were unchanged
everywhere they had already been computed.

### 10.4 Finding a circuit that actually reaches rank 20

This is the part worth reading. With the cap raised, nothing in the tree could
exercise it — and the reason is a compiler pass working correctly.

**`StatevectorSqueezePass` defeats most "high rank" circuits.** It reorders
operations so that T-injection circuits with only local Clifford glue collapse
almost completely. From `4b9efc7`:

> `rank19_q100_d10_wide` compiles to **peak_rank = 1**, and the `surface_d11`
> family tops out around 14.

A fixture named for rank 19 that compiles to rank 1. This is not a bug — it is
the entire premise of the SVM, which is that near-Clifford circuits have a small
*active* rank regardless of how many qubits they name.

**Quantum Volume circuits do not squeeze.** Each QV layer applies a random SU(4)
to a fresh random pairing of *all* qubits, so the whole register stays entangled
and `peak_rank == num_qubits` exactly. `generate_qv_circuits.py` emits them with
U3 angles in clifft's normalized convention. Verified with a host-side rank
probe running the identical compile path as `run_v2` (trace → HIR passes →
reference syndrome → lower → bytecode passes):

```
qv20_L8 = 20    qv21_L8 = 21    qv22_L6 = 22    qv23_L5 = 23    qv24_L4 = 24
```

Layer counts taper with width (`L8` → `L4`) to keep compile and run time
bounded.

### 10.5 The census — 22 of 353

"Are these all the circuits?" was a fair challenge, and it was answered by
measurement (`89d541e`). `probe_all.sh` ran the rank probe over **every** live
`.stim` in the tree — 353 files across `tests/fixtures`, `tools/bench/fixtures`,
and `docs/guide/circuits` — recording the **compiled** `peak_rank`, i.e. what
the tier dispatcher actually sees after `StatevectorSqueezePass`.

The raw probe output is `V2_performance/scratch/all_ranks.txt`, one
`<rank> <path>` line per fixture. Its full distribution — every fixture in the
tree, by compiled rank:

| compiled rank | fixtures | tier |
|---|---|---|
| 0 | 82 | register |
| 1 | 244 | register |
| 3 | 3 | register |
| 4 | 2 | register |
| **0–1 subtotal** | **326** | **register** |
| 7 | 3 | coop |
| 10 | 8 | coop |
| 11 | 2 | global |
| 12 | 1 | global |
| 13 | 1 | global |
| 14 | 1 | global |
| 20 | 2 | global |
| 21 | 1 | global |
| 22 | 1 | global |
| 23 | 1 | global |
| 24 | 1 | global |
| **≥ 5 subtotal** | **22** | **coop + global** |
| **total** | **353** | |

Note what is *absent*: **nothing in the entire tree compiles to rank 2, 5, 6,
8, 9, 15, 16, 17, 18 or 19.** The distribution is not a smooth spread with a
long tail — it is two dense clusters (0–1 and 10) plus a thin spine of surface
codes and the six hand-built QV circuits. The coop tier, which the bulk of the
optimization work in §9 targets, is exercised by exactly **eleven** fixtures,
eight of which are the `circuit_d5` parameter sweep.

Every one of `tests/fixtures/sweep` (188 files) and `rank_sweep` (56) squeezes
flat, *despite names like `rank_q17_r12_d1.stim` promising rank 12.*

Two consequences, both recorded in `BENCHMARKS.md` so the next person does not
have to re-derive them:

1. **A benchmark set covering "all the high-rank circuits" is 22 circuits, not
   hundreds.** The 26-circuit corpus in §15 is therefore not a sample — it is
   effectively the population, plus four parameter variants.
2. **Do not infer a fixture's tier from its name or qubit count. Probe it.**

This also reframes the whole project's scope honestly: the GPU backend's tiering
matters for a small, specific class of circuits. Most of what a user throws at
`clifft` collapses to rank 0–1 and never leaves the register tier.

### 10.6 What rank 20–26 actually costs

The five QV circuits are the only ones in the corpus above rank 19, and they are
where V2's advantage thins out:

| circuit | rank | V2 (µs) | SVM (µs) | ratio | VGPR spill | SGPR spill | scratch |
|---|---|---|---|---|---|---|---|
| qv20_seed42 | 20 | 1,780,109 | 2,121,123 | 0.839 | 0 | 0 | 112 |
| qv20_L8_seed42 | 20 | 1,560,026 | 2,158,259 | 0.723 | 0 | 0 | 96 |
| qv21_L8_seed42 | 21 | 2,946,337 | 4,045,623 | 0.728 | 0 | 0 | 112 |
| qv22_L6_seed42 | 22 | 3,176,355 | 3,237,575 | **0.981** | **136** | **762** | 448 |
| qv23_L5_seed42 | 23 | 6,083,877 | 6,142,585 | **0.990** | **199** | **662** | 576 |
| qv24_L4_seed42 | 24 | 7,240,955 | 8,225,562 | 0.880 | **152** | **594** | 480 |

The break is sharp and it is at rank 22. Below it, zero spilling and V2 wins
0.72–0.84. At and above it, the three worst-spilling kernels in the entire
89-kernel corpus — SGPR spill 762 / 662 / 594, the top three by that metric and
by VGPR spill — and the advantage collapses to 0.98–0.99.

> **Provenance — read this before quoting the timing columns.**
>
> The **resource columns are solid.** Both binaries survive in-tree — pre-fence
> under `V2_performance/history/stale_spec_cache_20260725/`, post-fence under
> `V2_performance/scratch/fence_cache/all/` — so this is directly checkable:
>
> | kernel | pre (B) | post (B) | Δ | bytes differing | `vgpr_spill` | `sgpr_spill` | `sgpr_count` | scratch |
> |---|---|---|---|---|---|---|---|---|
> | `global_r22_n359` | 217,424 | 218,576 | +1,152 | **83.9 %** | 136 = 136 | 762 = 762 | 108 = 108 | 448 = 448 |
> | `global_r23_n335` | 190,480 | 191,568 | +1,088 | **83.0 %** | 199 = 199 | 662 = 662 | 108 = 108 | 576 = 576 |
> | `global_r24_n320` | 172,880 | 173,904 | +1,024 | **82.9 %** | 152 = 152 | 594 = 594 | 108 = 108 | 480 = 480 |
>
> **Five-sixths of the binary changed and not one resource number moved.**
> Register pressure is a property of what the specializer emits, not of the
> fences around it, so the spill table is unaffected by the §11.4 invalidation.
>
> The **timing columns are provisional.** They come from a run that predates the
> fence fix (`150d09f`, 2026-07-26 06:33). A second run agrees to within 0.4 %
> — ratios 0.844 / 0.729 / 0.733 / 0.980 / 0.992 / 0.881, identical scratch
> sizes — but that run is `20260726T014859Z_all-tier5plus` at commit `89d541e`
> (01:52), which `git merge-base --is-ancestor` confirms is **also pre-fence**.
> Two pre-fence runs agreeing is a reproducibility check, **not** independent
> corroboration: both executed the same unfenced binaries.
>
> Whether that matters here is itself checkable, and the answer is *probably
> not for these six circuits* — the fence bug corrupted reduction totals in the
> coop tier, and these are global-tier kernels — but "probably" is not the
> standard this report holds itself to. The post-fix full-corpus re-run is the
> arbiter; §15 carries the final numbers and this table is superseded by it if
> they disagree.

**Why spilling appears exactly there — and what the data rules out.** The
tempting explanation is that the specializer emits one call per instruction with
constants baked in, so longer circuits carry more live scalars until the SGPR
budget breaks. **The corpus refutes this.** Sorting every `global_*` kernel by
emitted instruction count:

All fifteen `global_*` kernels in `lowering/kernel_resources.csv`, nothing
omitted:

| rank | instrs | VGPR | AGPR | SGPR | VGPR spill | SGPR spill | scratch |
|---|---|---|---|---|---|---|---|
| 14 | 16,521 | 128 | 64 | 106 | 0 | 4 | 352 |
| 11 | 16,415 | 128 | 64 | 106 | 0 | 4 | 320 |
| 11 | 16,415 | 128 | 64 | 106 | 0 | 4 | 320 |
| 13 | 9,359 | 128 | 64 | 106 | 0 | 4 | 336 |
| 13 | 9,359 | 128 | 64 | 106 | 0 | 4 | 336 |
| 11 | 8,952 | 128 | 64 | 106 | 0 | 4 | 320 |
| 11 | 8,952 | 128 | 64 | 106 | 0 | 4 | 320 |
| 12 | 4,296 | 128 | 64 | 106 | 0 | 4 | 320 |
| 12 | 4,296 | 128 | 64 | 106 | 0 | 0 | 320 |
| 20 | 418 | **104** | **40** | 106 | 0 | 0 | 96 |
| 20 | 387 | **106** | **42** | 106 | 0 | 0 | 112 |
| 21 | 393 | **104** | **40** | 106 | 0 | 0 | 112 |
| **22** | **359** | 128 | 64 | 108 | **136** | **762** | 448 |
| **23** | **335** | 128 | 64 | 108 | **199** | **662** | 576 |
| **24** | **320** | 128 | 64 | 108 | **152** | **594** | 480 |

The three spilling kernels are the three **shortest** in the tier. A rank-14
kernel emits 16,521 instructions — 46× more than rank 22's 359 — and spills 4
SGPRs. Instruction count is not the mechanism; if anything the correlation runs
backwards. That also disposes of the "qv24 is an anomaly because it has the
fewest instructions" story: fewest instructions is the norm among the spillers,
not an exception.

The full listing sharpens the boundary. **Rank 20–21 is the only region in the
tier where the allocator does not take the VGPR 128 / AGPR 64 cap** — three
kernels at 104–106 / 40–42, all with zero spill and the smallest scratch in the
tier (96–112 B). Every other kernel, at rank 11–14 and at rank 22–24 alike, is
at the cap. So the rank-22 break is not "the allocator hits the cap"; the
long rank-11–14 kernels are at the cap too and spill 4 SGPRs at most. It is that
rank 22 hits the cap **and** finds nothing left to overflow into.

What actually changes at the boundary is narrower than expected. Diffing the
emitted C for rank 21 against rank 22 with all numeric literals normalized shows
**no structural difference** — same preprocessor preamble, same `spec_body`
shape, near-identical opcode mix (169/80/42 `array_u2`/`array_u4`/`array_rot` vs
142/66/44). The tier wrapper is rank-independent by construction
(`v2_specializer.cc:196-224`). The one thing that differs is a single baked
constant:

```c
const u64 amp_capacity = 2097152ull;   // rank 21
const u64 amp_capacity = 4194304ull;   // rank 22
```

Crossing 2²² also moves the allocator's chosen register split: the non-spilling
rank 20/21 kernels sit at VGPR 104–106 / AGPR 40–42, while every spilling kernel
jumps to the VGPR 128 / AGPR 64 cap. On CDNA the AGPR file is repurposed as
overflow storage when MFMA is unused (and `SQ_INSTS_MFMA = 0` throughout here),
so the pattern is consistent with the allocator exhausting AGPR overflow at the
cap and falling through to scratch. **That is an inference from resource
metadata, not a verified mechanism** — the emitted C is essentially unchanged, so
the cause lies inside LLVM's allocation for the wider constant, which has not
been isolated. Stated as an open question rather than an answer.

What *is* solid: the correlation is exact. Every global kernel at rank ≤ 21
spills ≤ 4 SGPRs and wins 0.72–0.84; every one at rank ≥ 22 spills 594–762 SGPRs
to HBM-backed scratch — the same memory the kernel is already bandwidth-bound on
— and the win collapses to 0.98–0.99. So the honest conclusion holds:
**V2's specialization advantage decays above rank 21 and is essentially gone by
rank 23.** §16 lists it as the top open item; the first step is an LLVM
allocation study at the rank-21/22 boundary, not a code change.

---
