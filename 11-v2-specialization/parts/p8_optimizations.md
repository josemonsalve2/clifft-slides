## 9. Optimizations, tier by tier

V2 did not arrive at its current numbers by specializing. Specialization was
the fifth thing that happened, and if it had been the first, most of the
speedup would have been invisible — the earliest V2 build was **28× slower than
SVM** on `frame_h`, and no amount of constant folding closes a 28× gap.

This chapter walks the optimization history in the order it happened, with the
progressive ratio table as the spine. Every row of that table is a real
benchmark run archived under `V2_performance/runs/`; the full progression is in
`V2_performance/analysis/TRENDS.md`.

### 9.1 The progression

V2/SVM kernel-time ratio, lower is better, `< 1` means V2 wins. Seven of the
twelve archived runs, chosen at the points where something changed. Column 5 is
included **because it regressed** — see the note below it:

| circuit | baseline | after P0+P1 | after specializer | *fenced+gated* | noise-specialized | global-specialized | **final** |
|---|---|---|---|---|---|---|---|
| frame_h | **28.258** | 1.215 | 0.612 | *2.859* | 0.614 | 0.645 | **0.626** |
| circuit_d3_p0.001 | 15.193 | 1.081 | 1.116 | *2.117* | 0.524 | 0.504 | **0.525** |
| qv10 | 1.313 | 1.237 | **0.252** | *0.675* | 0.308 | 0.310 | **0.310** |
| surface_d7_t15 | 1.650 | 1.469 | 1.787 | *1.427* | **0.503** | 0.507 | **0.505** |
| surface_d9_t10 | 1.642 | 1.473 | 1.796 | *1.402* | **0.484** | 0.481 | **0.481** |
| surface_d7_t19 | 0.968 | 0.872 | 1.014 | 1.017 | 1.015 | **0.312** | **0.298** |
| surface_d9_t19 | 0.906 | 0.817 | 0.973 | 0.971 | 0.972 | **0.262** | **0.262** |
| surface_d11_t15 | 0.938 | 0.854 | 1.005 | 1.005 | 1.005 | **0.259** | **0.256** |

Read the bolded cells: **each tier's win arrives in exactly one step, and does
not move afterwards.** That is the signature of a structural change rather than
a tuning change. Register tier lands at P0. Coop lands at the specializer.
Noise-heavy coop lands at the noise fence. Global lands at the global emitter.

> **The italic column is a measurement artifact, and it is shown rather than
> dropped.** At `9d9cc68` the correctness gate existed but its verdict was
> cached only in-process. `rocprofv3` spawns a fresh process per invocation, so
> the gate's own validation dispatches re-ran *inside the profiled region* and
> the digester summed them into the kernel time — `frame_h` 0.612 → 2.859,
> `circuit_d3` 1.116 → 2.117, `qv10` 0.252 → 0.675. Nothing had actually slowed
> down. The next commit (`bbb5e42`) persisted the verdict to `<hsaco>.gate` so it
> is computed once ever, and the numbers returned to trend. Its message names the
> mechanism exactly: *"that polluted kernel traces with clifft_v2_coop+spec
> dispatches summed by the digester → inflated times."* The lesson is worth more
> than the column: **a correctness mechanism that runs on the measurement path
> becomes a performance number**, and three of these eight circuits would have
> been reported as 2–4× regressions by anyone reading the table without the
> commit history.

Two further caveats on this table, both of which follow from the project's own
benchmarking rule that ratios must not be compared across nodes:

- **The runs were not all on the same node.** Columns 1–8 ran on
  `smci350-rck-g03-d13-21`; columns 9–12, including **final**, ran on
  `smci350-rck-g03-f13-21` (recovered via `sacct`; only the final run recorded
  node identity in `node.json`, whose own note warns that "mi350x-es is
  heterogeneous — do not compare ratios across nodes"). Because every cell is a
  V2/SVM *ratio* with both halves measured in the same job, the node change
  affects the two backends together and the comparison survives; but the
  step-to-step deltas between columns 8 and 9 carry a node change as well as a
  code change, and should not be read as pure code effects.
- **`frame_h`'s 0.085 in the omitted `specializer-verify` column is not a real
  10× step.** That run's SVM side measured 128.2 µs against 16–20 µs everywhere
  else — a baseline outlier on a 12 µs kernel, not a V2 improvement. It is left
  out for that reason, and named here so the omission is not silent.

<figure>
<img src="diagrams/optimization-timeline.svg" alt="V2/SVM ratio over the optimization sequence" width="100%">
<figcaption><b>Figure 9.1</b> — V2/SVM kernel-time ratio across the twelve
archived benchmark runs. Each tier's curve is flat until the one change that
addresses it, then flat again. The y-axis is log-scaled to fit
<code>frame_h</code>'s 28.258 baseline.</figcaption>
</figure>

---

### 9.2 P0 — the shot-packed register tier (28.3× → 1.2×)

**The problem.** The first V2 was a single kernel topology: 256 threads
cooperate on one shot, amplitudes in LDS, `s_barrier` between every op. That is
the right shape for rank 10. For `frame_h` — rank 0, four instructions, no
amplitudes worth speaking of — it means 256 threads doing the work of one, with
a barrier after each of four instructions, while SVM's register path runs one
shot per thread. Hence 28.258×.

**The fix** (`715f8d0`). Rather than write a second kernel, the shared
`execute_shot` body was parameterized on *how threads cooperate*, through five
macros:

```c
#ifdef V2_REGISTER
#  define V2_STRIDE 1u
   static inline u32 v2_tid(void)  { return 0u; }
#  define IS_OWNER 1
   /* v2_barrier() -> nothing; V2_REDUCE2 -> identity (a stride-1 loop already summed) */
#else
#  define V2_STRIDE 256u
   static inline u32 v2_tid(void)  { return __builtin_amdgcn_workitem_id_x(); }
#  define IS_OWNER (t == 0)
   /* v2_barrier() -> fenced s_barrier; V2_REDUCE2 -> coop_reduce2 butterfly */
#endif
```

**The same opcode arithmetic now compiles two ways from one source.** Under
`-DV2_REGISTER`, `for (u64 i = t; i < iters; i += V2_STRIDE)` becomes
`for (i = 0; i < iters; i += 1)` — a plain serial loop in one thread — the
barriers vanish, `IS_OWNER` is constant-true so the tid0 guards evaporate, and
the reduction is the identity because a stride-1 loop has already summed
everything. `GpuComplex v[16]` lives in VGPRs.

The commit's measured result, quoted verbatim:

> Clean kernel-time vs GPU-SVM: frame_h 30.6x→1.14x, circuit_d3 14.1x→1.13x.
> The 15-28x low-rank catastrophe is eliminated; V2 register tier is now at the
> 1-shot-per-thread floor (SVM's own topology). Byte-exact vs SVM: 38/38.

Two things to note. First, "at the floor" — P0 did not make V2 *faster* than
SVM, it made V2 *the same shape* as SVM. The specialization win in §9.4 is
measured from that floor. Second, the commit message's own closing line
identifies why this refactor mattered beyond its number:

> The thread=shot topology and single tier-parameterized body are exactly what
> the MLIR specializer emits per-tier — **this refactor is the specializer's
> substrate.**

Without P0 there is no `spec_body` that all three tiers can share, and the
specializer would have needed three separate emitters.

---

### 9.3 P1a/P1b — LDS reclamation on the coop tier (2 → 4 workgroups/CU)

The coop tier's occupancy is set by LDS. At the P0 baseline the coop kernel
declared 25,088 bytes per workgroup, which on a 64 KB LDS budget allows
**2 workgroups per CU**. Two commits took it to four.

**P1a (`d873997`) — right-size the reduction buffers.** Three changes, all of
them "the buffer was declared for the worst case that cannot happen":

| Buffer | Before | After | Why it is safe |
|---|---|---|---|
| `lds_red0`, `lds_red1` | `[256]` | `[8]` | `coop_reduce2` only touches warps 0–3 |
| `lds_red_scratch` | `[1024]` | `[512]` | The coop `SWAP_MEAS` fold half is `≤ 2^(10-1) = 512` |

25,088 → 16,896 bytes, confirmed by `rocprofv3`'s `LDS_Block_Size`, occupancy
2 → 3 wg/CU.

**P1b (`7802423`) — bit-pack the measurement array.** Measurement records are
booleans and every access is tid0-only, so `u8 lds_meas[4096]` (4 KB) packs to
`u64 lds_meas[64]` (512 B) behind three accessors (`mget`/`mset`/`mxor1`)
replacing all 15 access sites. 16,896 → 13,312 bytes, occupancy 3 → 4 wg/CU.

Net: **occupancy doubled**, byte-exact vs SVM 27/27 at each step. The
progressive table shows P0+P1 together taking `surface_d9_t19` from 0.906 to
0.817 and `surface_d11_t15` from 0.938 to 0.854 — roughly a 10 % gain on the
coop-tier circuits, which is what a 2× occupancy improvement buys when the
kernel is not purely latency-bound.

Both commit messages note the forward connection, which is worth recording
because it is how the project actually proceeded:

> This is the same per-circuit LDS sizing the MLIR specializer will bake at
> compile time; here a conservative static bound.

The specializer knows each circuit's peak rank, so it could size these exactly.
It currently does not — it inherits the static bound. That is listed as an open
item in §16.

---

### 9.4 The specializer, in four commits

**Phase 1 (`eea6e13`) — extract the operand library.** Behavior-preserving
refactor: the interpreter becomes a thin `for(pc) switch` dispatching to
`static inline v2_op_*(st, v, scratch, active_k, operands...)` in a new shared
header. Each op takes its operands **and the pre-op `active_k` by value**. That
signature is the whole design:

> the interpreter passes runtime values (unchanged behavior), and the
> per-circuit specializer (next phase) emits a straight-line sequence of the
> same calls with COMPILE-TIME-CONSTANT operands → `-O2` folds loop bounds
> (`1<<(active_k-2)` → literal) and matrix indices, **WITHOUT unrolling the
> `2^k` sweeps (avoids V1's IR-bloat disease).**

Byte-exactness is then true *by construction*, not by testing: both paths call
the same function bodies. 38/38 byte-exact, timing unchanged.

A follow-up (`4b55871`) marked the bodies `always_inline` so the interpreter
gets them inlined into `execute_shot` exactly as the pre-refactor switch bodies
were. That commit also honestly records a cost that had not yet been explained:

> a ~20 % coop-interpreter regression vs pre-Phase1 remains on noise-heavy
> fallback circuits (under investigation)

That regression is still visible in today's numbers as the `circuit_d5` family's
1.44–1.46× — see §11.1, where it turns out not to be a performance problem at
all but the visible consequence of a correctness gate doing its job.

**Phase 2+3 (`b31b400`) — the register-tier emitter.** The first per-circuit
kernel. Measured against the *interpreter* on the same tier:

| circuit | interpreter (V2/SVM) | specialized (V2/SVM) | gain over interpreter |
|---|---|---|---|
| frame_h | 0.16× | **0.08×** | 2.0× |
| circuit_d3 | 1.07× | **0.32×** | 3.3× |
| color_d3 | 1.09× | **0.36×** | 3.0× |

**~3× for compile-time opcode resolution.** The commit states the thesis it was
testing and the verdict: *"This validates R5's thesis: compile-time opcode
resolution + constant folding is worth ~3x."*

**Coop tier (`60d5728`)** — the same `spec_body`, a 1-workgroup-per-shot
wrapper, LDS state. 5.5× on `qv10` and 7.6× on `surface_d7_t15` over the coop
interpreter. This is the commit where `qv10` drops to 0.252 in the progressive
table.

It is also the commit that introduced the **correctness gate**, and it did so
because coop specialization of noise-heavy circuits diverged. The gate is worth
restating: every `.hsaco` is validated against the interpreter on a multi-seed
shot sample before it is allowed to run, with the verdict cached to disk
(`bbb5e42` added the multi-seed + disk cache). A circuit that fails the gate
silently falls back to the interpreter. **Byte-exactness is guaranteed for every
circuit, not just the tested ones.**

**Global tier (`5d10409`).** This one began with a bottleneck analysis rather
than a hypothesis, and the commit records it:

> Bottleneck analysis (rocprofv3, V2 global vs SVM): V2 did 2× the VALU (2.34B
> vs 1.18B) and 25× more L2 misses at the same wall time — the extra VALU is
> per-amplitude scatter-index recompute (`scatter_bits_2` → `insert_zero_bit`,
> ~8–12 VALU each) that the runtime interpreter pays 256-way per shot and **SVM
> avoids via a scatter LUT it leaves OFF for global.**

That is S5 (§7.7) showing up as a whole-kernel bottleneck. SVM's answer was a
runtime lookup table; V2's answer is to fold the index arithmetic at compile
time, which is strictly better — no table, no LDS for the table, no lookup.
Result: 3.37× / 3.78× / 3.87× over the global interpreter on
`surface_d7_t19` / `d9_t19` / `d11_t15`, and in the progressive table those
three go from ~1.0 to ~0.26–0.31 in a single step.

<figure>
<img src="diagrams/scatter-index-folding.svg" alt="scatter_bits_2 at runtime vs folded" width="100%">
<figcaption><b>Figure 9.2</b> — The global-tier bottleneck. Left: the
interpreter recomputes <code>scatter_bits_2</code> per amplitude, 8–12 VALU
each, 256 threads deep. Centre: SVM's runtime scatter LUT, which it disables on
the global tier. Right: V2's compile-time fold — three constant masks and a
<code>v_or3</code>.</figcaption>
</figure>

---

### 9.5 The noise fence (`9d9cc68`) — and what it was actually fixing

After the coop emitter, noise-heavy circuits still failed the gate. The fix that
shipped had two parts:

1. **`V2_NOISE_ATTR`** — emit the FP-carrying noise ops and the
   reduction-carrying measurement ops **`noinline`** in the specialized build,
   so each straight-lined op is an optimization barrier the `-O2` scheduler
   cannot move FP across. The commit describes this as *"the specializer analog
   of 'resolve to a loop instead of unrolling'."*
2. **Identical compile flags.** The runtime compile had drifted:
   `-fno-vectorize -fno-unroll` had been added, which themselves differed from
   the build-time interpreter. Dropping them made the two paths compile the same
   way.

This took `surface_d7_t15` from 1.427 to 0.503 and `surface_d9_t10` from 1.402
to 0.484 — the "noise-specialized" column in §9.1 — because those circuits could
now pass the gate and run specialized.

**But the stated explanation was wrong, and the correction matters.** The commit
attributed the divergence to `-O2` reassociating FP across inlined call
boundaries. Four commits later (`150d09f`) that theory was refuted:

> That theory was refuted: the build is `-O2 -ffp-contract=off` with no
> fast-math, under which **inlining cannot legally change an FP result**, and
> `V2_GATE_SELFTEST` caught the kernel **disagreeing with itself across two runs
> of the same binary.**

A deterministic kernel that disagrees with itself is not a rounding problem. It
is a race. §11.2 tells that story. `V2_NOISE_ATTR` was retained — but for code
size, not correctness — and `V2_SPEC_NOISE_INLINE=1` exists precisely so the
hypothesis can be A/B tested rather than assumed.

This is the single best example in the project of the report's ground rule
paying off. The fence *worked*; the *reason given for why it worked* was false;
and believing the false reason would have left the actual race in the code.

---

### 9.6 One more correctness fix that looked like a performance result (`842d646`)

Worth including because it is a trap this kind of work sets constantly. V2
counted the **raw** observable parity; every reference implementation — GPU-SVM
in `hip_sampler.hip` and the CPU sampler in `svm.cc` — XORs it against the
noiseless reference syndrome first. On any circuit whose reference observable is
1, V2 reported the exact **complement** of SVM's count: `surface_d11_t19` gave
674 where SVM gave 1326 of 2000 passed shots.

The fix threads the reference through as a packed `u32 expected_obs_mask`. The
implementation detail is the interesting part:

> placed in the implicit pad slot after `num_noise_sites` so the kernarg struct
> **SIZE is unchanged** (a 120-vs-116 mismatch silently breaks every dispatch).

With HSA dispatch there is no runtime checking the kernarg layout for you (§13).
The ABI version was bumped to 3 and `device_abi_checks.cc` static-asserts the
layout, which is how a hand-rolled dispatch path stays safe.

---

### 9.7 Where the wins came from, summarized

| Change | Mechanism | Who it helped | Magnitude |
|---|---|---|---|
| P0 shot-packed register tier | topology: 1 shot/thread instead of 256 threads/shot | rank ≤ 4 | 28.3× → 1.2× |
| P1a/P1b LDS reclamation | occupancy 2 → 4 wg/CU | all coop | ~10 % |
| Specializer, register | constant-fold operands, delete dispatch | rank ≤ 4 | 3.3× over interpreter |
| Specializer, coop | same, plus rank-folded loop bounds | rank 5–10 | 5.5–7.6× over interpreter |
| Noise fence + gate | made noise circuits *eligible* to specialize | noise-heavy coop | 1.43 → 0.50 |
| Specializer, global | fold `scatter_bits_2`, no LUT needed | rank 11–19 | 3.4–3.9× over interpreter |
| Rank cap 19 → 26 | size HBM from circuit rank, not the cap | rank 20–26 | new capability (§10) |

The pattern across the whole table: **every large win came from removing
something the runtime was doing, not from making the arithmetic faster.** The
arithmetic — the complex multiplies in the butterfly — is essentially unchanged
from SVM. What changed is that V2 stopped fetching instructions, stopped
switching on opcodes, stopped recomputing indices, and stopped reloading the
rank.

---
