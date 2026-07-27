## 9. Optimizations, tier by tier

V2 did not arrive at its current numbers by specializing. Specialization was
the fifth thing that happened, and if it had been the first, most of the
speedup would have been invisible — the earliest V2 build was **28× slower than
SVM** on `frame_h`, and no amount of constant folding closes a 28× gap.

This chapter walks the optimization history with the progressive ratio table as
the spine. Every row of that table is a real benchmark run archived under
`V2_performance/runs/`; the full progression is in
`V2_performance/analysis/TRENDS.md`.

The sections are ordered by *magnitude*, not by date, and the two differ. The
actual commit order on `mlir-v2`, from `git log --reverse`, is:

| # | commit | time | what |
|---|---|---|---|
| 1 | `d873997` | 07-25 04:46 | P1a — shrink reduction LDS |
| 2 | `7802423` | 07-25 04:50 | P1b — bit-pack `lds_meas` |
| 3 | `715f8d0` | 07-25 05:11 | **P0 — shot-packed register tier** |
| 4 | `eea6e13` | 07-25 07:36 | specializer Phase 1 — operand library |
| 5 | `b31b400` | 07-25 07:47 | specializer Phase 2+3 — register emitter |
| 6 | `60d5728` | 07-25 08:24 | specializer — coop tier + correctness gate |
| 7 | `4b55871` | 07-25 08:34 | force-inline `v2_op_*` in the interpreter |
| 8 | `9d9cc68` | 07-25 16:24 | noise/measurement `noinline` fencing |
| 9 | `bbb5e42` | 07-25 16:35 | gate: multi-seed + disk cache |
| 10 | `5d10409` | 07-25 17:18 | specializer — global tier |
| 11 | `842d646` | 07-25 23:06 | XOR observable parity |
| 12 | `150d09f` | 07-26 06:33 | `v2_barrier` → memory barrier |

**P1 shipped 25 minutes before P0**, despite being numbered after it and being
worth ~10 % against P0's 28×. It was picked first because it was judged the
safest change in the plan (layout only, no algorithm change), which is a
reasonable way to start and a misleading way to report — so the ordering is
stated here rather than smoothed over.

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

That reading does not have to be taken on trust. Every archived run kept its
`rocprofv3` kernel trace, and the kernel *name* in the trace says which code
path produced each cell — `clifft_v2_register` / `_coop` / `_global` are the
three interpreter kernels, `clifft_v2_spec` is a specialized one. Recovering it
per cell:

| circuit | after-specializer | *fenced+gated* | noise-specialized | global-specialized | **final** |
|---|---|---|---|---|---|
| frame_h | `spec` | `register`+`spec` | `spec` | `spec` | `spec` |
| circuit_d3 | `spec` | `register`+`spec` | `spec` | `spec` | `spec` |
| qv10 | `spec` | `coop`+`spec` | `spec` | `spec` | `spec` |
| surface_d7_t15 | **`coop`** | `coop`+`spec` | `spec` | `spec` | `spec` |
| surface_d9_t10 | **`coop`** | `coop`+`spec` | `spec` | `spec` | `spec` |
| surface_d7_t19 | **`global`** | `global` | **`global`** | `spec` | `spec` |
| surface_d9_t19 | **`global`** | `global` | **`global`** | `spec` | `spec` |
| surface_d11_t15 | **`global`** | `global` | **`global`** | `spec` | `spec` |

Each ~1.0 in the table is a cell where the trace shows an *interpreter* kernel:
the specializer had not reached that tier yet, or the gate had rejected the
circuit. The step to ~0.5 or ~0.3 is always the same cell flipping to `spec`.
**Nothing in this table is a tuning effect** — every movement is a change of
which kernel ran.

The trace also settles the italic column independently of the commit history:
`noise-fenced-gated` is the **only** run in which any circuit dispatched *two*
different kernels. That is the gate's validation dispatch, caught in the act.

> **The italic column is a measurement artifact, and it is shown rather than
> dropped.** At `9d9cc68` the correctness gate existed but its verdict was
> cached only in-process. `rocprofv3` spawns a fresh process per invocation, so
> the gate's own validation dispatches re-ran *inside the profiled region* and
> the digester summed them into the kernel time — `frame_h` 0.612 → 2.859,
> `circuit_d3` 1.116 → 2.117, `qv10` 0.252 → 0.675. Nothing had actually slowed
> down. The next commit (`bbb5e42`) persisted the verdict to `<hsaco>.gate` so it
> is computed once ever, and the numbers returned to trend. Its message names the
> mechanism exactly: *"that polluted kernel traces with clifft_v2_coop+spec
> dispatches summed by the digester → inflated times."*
>
> The kernel-trace stats show precisely what was summed — **three dispatches
> where every other run has one**:
>
> ```
> noise-fenced-gated   frame_h    clifft_v2_register x1 (19.1us) + clifft_v2_spec x2 (28.8us)
>                      qv10       clifft_v2_coop     x1 (1270.2us) + clifft_v2_spec x2 (1667.2us)
> noise-specialized    frame_h    clifft_v2_spec     x1 (10.5us)
>                      qv10       clifft_v2_spec     x1 (1340.0us)
> ```
>
> One interpreter run and two specialized runs: the gate executing the circuit
> both ways to compare them, plus the real sampling dispatch. The reported
> "regression" is the sum of all three. The lesson is worth more than the
> column: **a correctness mechanism that runs on the measurement path becomes a
> performance number**, and three of these eight circuits would have been
> reported as 2–4× regressions by anyone reading the table without the commit
> history.

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

### 9.3 P1a/P1b — LDS reclamation on the coop tier

The coop tier's occupancy is set by LDS. At the P0 baseline the coop kernel
declared 25,088 bytes per workgroup. Two commits took that to 13,312.

**P1a (`d873997`) — right-size the reduction buffers.** Both changes are of the
form "the buffer was declared for a worst case that cannot happen":

| Buffer | Before | After | Why it is safe |
|---|---|---|---|
| `lds_red0`, `lds_red1` | `[256]` | `[8]` | `coop_reduce2` only touches warps 0–3 |
| `lds_red_scratch` | `[1024]` | `[512]` | The coop `SWAP_MEAS` fold half is `≤ 2^(10-1) = 512` |

25,088 → 16,896 bytes, confirmed by `rocprofv3`'s `LDS_Block_Size`
(`V2_performance/tools/ldscheck_50017.log`).

**P1b (`7802423`) — bit-pack the measurement array.** Measurement records are
booleans and every access is tid0-only, so `u8 lds_meas[4096]` (4 KB) packs to
`u64 lds_meas[64]` (512 B) behind three accessors (`mget`/`mset`/`mxor1`)
replacing all 15 access sites. 16,896 → 13,312 bytes (`ldscheck_50021.log`).
Today's HEAD reports 13,064 in the ELF metadata
(`.group_segment_fixed_size: 13064`); the extra 248 bytes were reclaimed by
later changes.

> **Correction — the "2 → 4 wg/CU" claim in both commit messages is wrong on
> this hardware, and the mechanism was not occupancy.** Both messages derive
> occupancy from `floor(64 KB / LDS)`, and the P1 planning documents state
> "CDNA4, 64 KB LDS/CU" as a premise
> (`gpu_kernel_static_characterization.md:77`). **gfx950 has 160 KB of LDS per
> CU, not 64 KB.** Asking the compiler directly — the same clang that builds
> these kernels — by binary-searching the largest accepted `address_space(3)`
> array:
>
> ```
> gfx90a: max LDS per workgroup = 65,473 B    (64 KB limit)
> gfx942: max LDS per workgroup = 65,473 B    (64 KB limit)
> gfx950: max LDS per workgroup = 163,681 B   (160 KB limit)
> ```
>
> and over-allocating by one byte prints the constant verbatim:
> `error: local memory (163844) exceeds limit (163840) in 'k'`. LLVM's own
> `; Occupancy:` comment, for a 256-thread workgroup at each historical budget:
>
> | LDS bytes | gfx942 | **gfx950** |
> |---|---|---|
> | 25,088 (baseline) | 2 | **6** |
> | 16,896 (after P1a) | 3 | **8** |
> | 13,312 (after P1b) | 4 | **8** |
> | 8,704 (global tier) | 7 | **8** |
>
> The gfx942 column is exactly the "2 → 3 → 4" the commits claim: the reasoning
> was right for the *previous* generation. On gfx950 the model is
> `min(8, 163840 / LDS)`, verified against LLVM at every step boundary
> (occupancy first drops below 8 at 20,992 bytes, and `163840/20992 = 7`).
> **Both P1 steps therefore moved entirely inside the flat region: 8 wg/CU
> before, 8 wg/CU after.** The occupancy cap here is the hardware's 8
> waves/SIMD, and the kernel was already at it. Sweeping VGPR from 32 to 256 at
> each LDS size leaves every entry unchanged, so registers were never binding
> either.

**So what did P1 actually buy?** The progressive table shows P0+P1 together
taking `surface_d9_t19` from 0.906 to 0.817 and `surface_d11_t15` from 0.938 to
0.854 — about 10 %, real and reproducible. But it cannot be occupancy on this
node, and the two commits are not separable in the archived runs (the
`after-P0-P1` run measures all three changes at once, on top of P0's topology
change, which is itself worth 28×). **The honest statement is that the LDS
reclamation is a correctness-preserving footprint reduction whose measured
benefit on gfx950 is not isolated, and whose stated mechanism does not hold
here.** It would be the difference between 2 and 4 wg/CU on an MI300X
(gfx942) — which is where the plan was written — and it becomes binding again
on gfx950 for any future kernel that pushes past 20 KB. What it definitely did
buy is headroom: the rank-26 work in §10 spends LDS that P1 freed.

This is also the one place where the report's ground rule cuts against a result
the project was pleased with. The measurement (`LDS_Block_Size` 25,088 →
16,896 → 13,312) was correct at every step; the *inference from it* used a
constant from the wrong chip.

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

`4b55871` marked the bodies `always_inline` so the interpreter gets them inlined
into `execute_shot` exactly as the pre-refactor switch bodies were. (It is
numbered as a follow-up to Phase 1 here because that is what it fixes, but it
landed *after* the coop emitter — 08:34 against 08:24 — so the two
`after-specializer` / `specializer-final` runs bracket it and are otherwise the
same code.) That commit also honestly records a cost that had not yet been
explained:

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
wrapper, LDS state. This is the commit where `qv10` drops to 0.252 in the
progressive table.

The commit message claims *"qv10 5.5x, surface_d7_t15 7.6x"* over the coop
interpreter. **Those were measured ad-hoc during development and are not
reproducible from the archived runs**, which give a consistently smaller
number. Taking each circuit's last interpreter run against its first
specialized run — same node, same shot count, kernel identity confirmed by
trace:

| circuit | interpreter (µs) | specialized (µs) | gain |
|---|---|---|---|
| surface_d7_t15 | 39,352.8 (`coop`) | 11,117.4 (`spec`) | **3.54×** |
| surface_d9_t10 | 79,242.3 (`coop`) | 21,318.3 (`spec`) | **3.72×** |
| surface_d7_t19 | 20,295.2 (`global`) | 6,229.2 (`spec`) | **3.26×** |
| surface_d9_t19 | 43,588.2 (`global`) | 11,726.6 (`spec`) | **3.72×** |
| surface_d11_t15 | 75,154.0 (`global`) | 19,412.2 (`spec`) | **3.87×** |

**Specialization is worth ~3.3–3.9× over the interpreter, uniformly across coop
and global.** That is a cleaner result than the commit messages' spread of
2.0×–7.6×, and it agrees with the register tier's independently-measured ~3×
(§9.4, `b31b400`) and with the per-op instruction counts in §7 (a 2.9×
median). Three different measurements of the same lever converge on ~3×.

`qv10` is the exception and is instructive: 5,358.7 µs interpreter → 1,085.9 µs
specialized is **4.93×**, but its specialized time then *rises* to 1,340–1,361 µs
in every later run and settles at 0.310 rather than 0.252. The counters name the
cause without ambiguity:

| run | V2 µs | V2/SVM | VGPR | VALU |
|---|---|---|---|---|
| after-specializer | 1085.9 | 0.252 | 24 | 5.10e+08 |
| specializer-final | 1088.6 | 0.251 | 24 | 5.31e+08 |
| noise-specialized | 1340.0 | 0.308 | **36** | 5.02e+08 |
| global-specialized | 1346.4 | 0.310 | **36** | 5.02e+08 |
| final | 1361.3 | 0.310 | **36** | 5.02e+08 |

**VALU goes *down* while time goes up, and VGPR jumps 24 → 36 at exactly the
run that follows the noise fence.** Fewer vector instructions taking more time
with more registers live is the signature of the `noinline` boundaries in
`9d9cc68`: the ops can no longer be interleaved across call edges, so the
scheduler has less to overlap and the ABI forces values live across the calls.
It is not noise, not the node (all five runs are `d13-21`), and not the SVM
baseline (4,317–4,388 µs throughout). **`qv10` pays about 23 % for
byte-exactness**, and the progressive table reports the post-fence number
rather than quoting the faster incorrect one.

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

This took `surface_d7_t15` from **1.793 to 0.503** and `surface_d9_t10` from
**1.800 to 0.484** — a **3.5×** and **3.7×** step — because those circuits could
now pass the gate and run specialized.

(The step must be measured from the `specializer-final` column, not from the
italic `1.427`/`1.402`. Those intermediate cells are the gate-polluted ones:
they are *already partly specialized*, being the sum of an interpreter dispatch
and two specialized dispatches, so they sit between the two real values and
understate the step. The VALU counter makes the three states plain —
4.71e+09 interpreter, 3.49e+09 polluted mixture, 1.15e+09 specialized.)

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

The layout confirms it — compiling `device_abi.h` and printing the offsets:

```
sizeof(CV2KernArgs)        = 152
offsetof num_noise_sites   = 112
offsetof expected_obs_mask = 116
```

The new field sits at 116, in the four bytes of padding that the `u32` at 112
already forced the compiler to reserve ahead of the next 8-byte-aligned member.
**A whole new argument was added for free.** (The commit's "120-vs-116" refers
to that boundary, not to the total, which is 152 both before and after.)

With HSA dispatch there is no runtime checking the kernarg layout for you (§13):
the AQL packet carries a pointer and the kernel casts it. A size or offset
mismatch is not an error, it is silently reading the wrong bytes. That is why
`device_abi_checks.cc` exists — 55 `static_assert`s pinning every offset and
size in the device header to the host structs in `gpu_types.h`, plus the opcode
numbering:

```c
static_assert(static_cast<int>(clifft::Opcode::OP_FRAME_CNOT) == 0, "opcode FRAME_CNOT");
static_assert(static_cast<int>(clifft::Opcode::OP_EXPAND)     == 19, "opcode EXPAND");
static_assert(sizeof(CV2Instr) == sizeof(GpuInstr), "CV2Instr size");
static_assert(offsetof(CV2Instr, opcode) == offsetof(GpuInstr, opcode), "instr.opcode");
```

Its header comment records that this is not hypothetical: *"this caught an
off-by-one: EXPAND/MEAS were numbered -1."* A build error instead of a silent
GPU miscompute is the whole return on the file.

---

### 9.7 Where the wins came from, summarized

| Change | Mechanism | Who it helped | Magnitude |
|---|---|---|---|
| P0 shot-packed register tier | topology: 1 shot/thread instead of 256 threads/shot | rank ≤ 4 | **28.3× → 1.2×** |
| P1a/P1b LDS reclamation | 25,088 → 13,312 B/workgroup | all coop | ~10 %, mechanism unconfirmed on gfx950 (§9.3) |
| Specializer, register | constant-fold operands, delete dispatch | rank ≤ 4 | 3.3× over interpreter |
| Specializer, coop | same, plus rank-folded loop bounds | rank 5–10 | **3.5–3.7×** over interpreter |
| Noise fence + gate | made noise circuits *eligible* to specialize | noise-heavy coop | 1.79 → 0.50 |
| Specializer, global | fold `scatter_bits_2`, no LUT needed | rank 11–19 | **3.3–3.9×** over interpreter |
| Rank cap 19 → 26 | size HBM from circuit rank, not the cap | rank 20–26 | new capability (§10) |

Two rows changed under audit and are worth naming: the coop specializer is
**3.5–3.7×**, not the 5.5–7.6× its commit message claims (§9.4), and the P1
occupancy mechanism does not hold on this chip (§9.3). Both corrections came
from the archived runs rather than from the commit log.

The specialization rows now agree with each other — **~3.3–3.9× across all
three tiers** — which is a stronger claim than the scattered original figures,
because it says the lever is the same lever everywhere: resolve the opcode at
compile time.

The pattern across the whole table: **every large win came from removing
something the runtime was doing, not from making the arithmetic faster.** The
arithmetic — the complex multiplies in the butterfly — is essentially unchanged
from SVM. What changed is that V2 stopped fetching instructions, stopped
switching on opcodes, stopped recomputing indices, and stopped reloading the
rank.

---
