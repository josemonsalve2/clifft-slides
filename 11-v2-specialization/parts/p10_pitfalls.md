## 11. Pitfalls

Four things went wrong in V2 that are worth writing down, because in every case
a *plausible* explanation was adopted, documented, and built on — and in every
case the plausible explanation was wrong. The pattern is uniform enough to state
up front:

> Each bug was diagnosed by analogy ("this looks like an FP problem", "this looks
> like a fixture problem") and each was actually resolved by an experiment that
> asked the system a question it could only answer one way. The diagnosis-by-
> analogy was never merely incomplete; it pointed at the wrong subsystem.

### 11.1 The noise regression, and the theory that was wrong

**The symptom.** The `circuit_d5` / `cultivation_d5` family — rank 10, 1,720
instructions, the noisiest circuits in the corpus — produced measurement
outcomes from the specialized kernel that differed from the interpreter on
roughly one shot in five thousand, on some seeds.

**The theory.** Straight-lining the bytecode duplicates each noise op's body at
every call site. Noise ops carry scalar FP: an `ocml_log_f64` hazard draw,
cumulative-probability sums, RNG. If `-O2` reassociated that FP differently
across copies, a probability could shift by ~1 ULP, `sample_branch` could take
the other side of a comparison, and — because a branch decides *whether a PRNG
draw is consumed* — the two streams would decorrelate permanently and never
resynchronize.

This story is mechanically coherent, it names a real hazard, and it fits the
observation that only the noisiest circuit family was affected. It motivated
`V2_NOISE_ATTR` (`9d9cc68`): emit the noise ops `noinline` in the specialized
build so `-O2` cannot merge or reassociate across the copies.

**Why it was wrong.** Two facts, either of which is sufficient:

1. **The build cannot legally do what the theory requires.** The specializer
   compiles with `-O2 -ffp-contract=off` and no fast-math. Under those flags
   reassociation of floating-point is *not permitted*, so inlining cannot change
   an FP result. `v2_compile_cache.cc` even says so at the flag site:

   ```
   // Flags are IDENTICAL to the build-time interpreter pipeline
   // (ClifftAmdgcn.cmake) so the shared v2_op_*() bodies compile to the same
   // code -> byte-exact. -ffp-contract=off with no fast-math is what makes that
   // hold regardless of inlining decisions: without reassociation or
   // contraction the optimizer cannot change an FP result. Do NOT add
   // vectorize/unroll/fast-math flags that would diverge from the interpreter.
   ```

2. **The kernel disagreed with itself.** `V2_GATE_SELFTEST` (`ca82260`) runs a
   kernel against *itself*, twice, same seed, same shots, same binary. Rounding
   is deterministic for a fixed binary, so any disagreement is proof of
   nondeterminism. Result:

   Per-seed, 5,000 shots each, `circuit_d5_p0.001` (job 50361). The counts are
   `obs0` ones from run A vs run B of the *same binary*:

   | seed | interp vs interp | verdict | spec vs spec | verdict |
   |---|---|---|---|---|
   | 1 | 1620 / 1620 | deterministic | 1620 / **1619** | **NONDETERMINISTIC** |
   | 7 | 1724 / 1724 | deterministic | 1724 / **1723** | **NONDETERMINISTIC** |
   | 42 | 1704 / 1704 | deterministic | 1704 / 1704 | deterministic |
   | 99 | 1720 / 1720 | deterministic | 1721 / **1720** | **NONDETERMINISTIC** |
   | 123 | 1691 / 1691 | deterministic | 1690 / **1691** | **NONDETERMINISTIC** |
   | 2718 | 1681 / 1681 | deterministic | 1681 / 1681 | deterministic |
   | | **0 / 6 diverge** | | **4 / 6 diverge** | |

   Note that `passed` was 5000/5000 in *every* cell — the shots all completed;
   only the observable flipped. The divergence is one sample in ~1,700, which is
   exactly the "one shot in five thousand" the symptom section describes, now
   attributed to the right cause. No rounding model explains a binary disagreeing
   with itself. This is a race.

`V2_GATE_BISECT` then localized it. The instrumentation
(`v2_kernel.cc:226-247`) re-runs each shot as its own dispatch —
`run_one(..., i, i+1, ...)` — and in the coop tier one shot is exactly one
workgroup. Scanning all 5,000 shots of the worst seed emitted **not a single
`DIVERGES` line**: at one workgroup the two kernels agree on every shot. The bug
needed *concurrent* workgroups, which also rules out any single shot's
arithmetic — the same instruction stream, on the same inputs, is correct when
run alone.

The A/B on the fix itself was the last nail: `V2_SPEC_NOISE_INLINE=1`, which
flips `V2_NOISE_ATTR` back to `always_inline`, made things **worse** — 5/6 seeds
diverging rather than 4/6, and with *larger* deltas (seed 99 went from off-by-1
to off-by-3, 1721 / 1718). That is consistent with perturbing a race's timing,
and inconsistent with the attribute being the load-bearing fix. It also shows
the trap in the original diagnosis: had the A/B not been run, the attribute
would have looked like a partial fix rather than no fix at all, because both
arms diverge on *some* seeds and the seed set is small.

**What was kept and why.** `V2_NOISE_ATTR` stayed, but its justification was
rewritten in the source to say what is actually true:

> The attribute is kept because it measurably helps register pressure and code
> size on large circuits, **NOT** because it is load-bearing for correctness.

That is the honest outcome: a change introduced for a wrong reason that happens
to pay for itself for a different, measurable reason. The comment at
`v2_ops_body.inc:409-427` is labelled `HISTORICAL NOTE` and states the refutation
inline, so the next reader cannot re-derive the dead theory from the code.

### 11.2 The real bug: a barrier that ordered execution but not memory

`v2_barrier()` was a bare `__builtin_amdgcn_s_barrier()`.

On AMDGCN that is an **execution** barrier only. LLVM declares the intrinsic
`IntrNoMem`, so the backend emits no `s_waitcnt lgkmcnt(0)` ahead of it and the
scheduler is free to sink or hoist LDS accesses across it. A wave can retire the
barrier with a `ds_write` still in flight while a peer wave reads the stale word.

Measured on the pre-fix gfx950 ISA — barriers with a `ds_*` op in flight and no
intervening `s_waitcnt lgkmcnt(0)`. Every number below was re-derived for this
report by disassembling the archived binaries with `llvm-objdump` and by
rebuilding the interpreter twice from the same source with only the barrier
body swapped:

| kernel | barriers | unfenced | how measured |
|---|---|---|---|
| interpreter (`clifft_v2_coop`) | 80 | **50 (62.5 %)** | A/B rebuild, `s_barrier` vs release/acquire, `-O2` asm |
| **specialized** (`coop_r10_n1720`) | 1,509 | **1,400 (92.8 %)** | `llvm-objdump -d` on the archived pre-fix `.hsaco` |

**Both kernels were wrong.** The specializer has **~28× more** exposed barriers
because it straight-lines every op, which is precisely why only it failed the
correctness gate — the interpreter was equally incorrect in principle and simply
lost the race far less often.

> **Correction.** Commit `150d09f`'s message, and the header comment in
> `V2_performance/scratch/d5_fence.sh`, both record the specialized figure as
> **1439 / 1509 (95.4 %)**. That number does not reproduce. Disassembling the
> archived pre-fix `coop_r10_n1720_977e1e830813621d.hsaco` gives **1,400** under
> every reasonable definition of "in flight" — scanning back 24, 32, 48, 64 or
> unbounded instructions to the previous barrier all converge on 1,400, and
> loosening `lgkmcnt(0)` to any `lgkmcnt` does not move it either. The only
> definitions that yield anything else are degenerate: counting barriers not
> *immediately* preceded by a wait gives 1,509, and a 3-instruction window gives
> 1,460. The interpreter figure is likewise 50 / 80, not 52 / 81 — the archived
> count was taken on a `.hsaco` linked against `ocml`/`ockl`, which contributes
> one extra barrier and one extra unfenced site from device-library code that is
> not V2's. **None of this changes the conclusion**; the corrected ratio (28×) is
> marginally *larger* than the one originally claimed.

This also explains why the SVM and Hybrid backends never saw it: HIP's
`__syncthreads()` already expands to the fenced sequence. **V2 hand-rolled the
barrier to avoid HIP and lost the fence along with it** — a direct, and
expensive, cost of the no-HIP policy (§13).

The fix (`150d09f`) wraps the barrier in a release/acquire pair:

```c
// s_barrier alone is an EXECUTION barrier only: LLVM models the intrinsic as
// IntrNoMem, so it neither emits `s_waitcnt lgkmcnt(0)` nor stops the scheduler
// from sinking/hoisting LDS accesses across it. A wave can therefore retire the
// barrier with a ds_write still in flight while a peer wave reads the stale
// word. The release/acquire fence pair is what makes it a MEMORY barrier: the
// release forces the waitcnt before s_barrier, the acquire keeps later loads
// from being hoisted above it. HIP's __syncthreads() expands to exactly this,
// which is why the SVM backend never saw the race.
static inline void v2_barrier(void) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
```

`coop_reduce2`'s three raw barriers were converted too — each guards a
read-after-write on `lds_red0`/`lds_red1`, so a lost partial sum perturbs the
reduction total and flips `sample_branch`. **That is the causal path by which a
missing fence became a wrong measurement outcome**, and it is why the symptom
looked numerical: the corrupted value really was a probability.

The cost was measured directly, by compiling `coop_interpreter.c` twice — once
with the bare `s_barrier`, once with the fence pair, same source, same flags
(`-O2 -ffp-contract=off -mcpu=gfx950`), disassembling both:

| build | instructions | barriers | fenced | `ds_*` in flight, unfenced |
|---|---|---|---|---|
| pre-fix interpreter | 5,629 | 80 | 1 (1.2 %) | **50 (62.5 %)** |
| post-fix interpreter | 5,848 | 80 | 74 (92.5 %) | **0 (0.0 %)** |
| pre/post register tier | 4,457 | 0 | — | — |

Two things fall out. **The fix is complete**: the unfenced-with-`ds`-in-flight
count goes to exactly zero — the 6 barriers that remain unfenced have no LDS
write pending, so the wait is correctly elided rather than missing. And the
price is **+3.89 % static instructions** (5,629 → 5,848), matching the +3.88 %
recorded at the time. The register tier is unaffected in both directions: with
`-DV2_REGISTER` there are no barriers at all, so the two builds are
byte-identical at 4,457 instructions — **the fix costs the register tier
nothing**, which is why §9's register-tier numbers need no restatement.

The same audit on the archived specialized binaries:

| `.hsaco` | barriers | fenced (`lgkmcnt(0)` within 3) | `ds_*` in flight, unfenced |
|---|---|---|---|
| pre-fix `coop_r10_n1720` | 1,509 | **49 (3.2 %)** | **1,400 (92.8 %)** |
| post-fix `coop_r10_n1720` | 1,509 | **1,404 (93.0 %)** | **0 (0.0 %)** |

Note the two columns measure different things and both are worth reporting: the
pre-fix kernel has 49 barriers that *happen* to sit behind a wait emitted for an
unrelated reason — accidentally correct, not correct by construction. Post-fix,
the in-flight column is zero, which is the property that actually matters.

With the fence in place the gate **passes** on `coop_r10_n1720`, and the
specializer is selected for all six circuits. Kernel time, median of 5, 10,000
shots, both arms back-to-back on one node (job 50389, gfx950):

| circuit | interpreter | specialized | speedup |
|---|---|---|---|
| circuit_d5_p0.0005 | 14.76 ms | 8.56 ms | 1.72× |
| circuit_d5_p0.001 | 14.89 ms | 8.56 ms | 1.74× |
| circuit_d5_p0.002 | 15.09 ms | 8.63 ms | 1.75× |
| circuit_d5_p0.003 | 15.30 ms | 8.68 ms | 1.76× |
| circuit_d5_p0.005 | 15.62 ms | 8.87 ms | 1.76× |
| circuit_d3_p0.001 | 0.624 ms | 0.379 ms | 1.65× |

<figure>
<img src="diagrams/barrier-race.svg" alt="Execution-only barrier vs release/acquire fenced barrier" width="100%">
<figcaption><b>Figure 11.1</b> — Why an execution-only barrier loses a partial
sum. Wave 0 retires <code>s_barrier</code> with its <code>ds_write</code> to
<code>lds_red0</code> still in flight; wave 1 passes the same barrier and reads
the stale word. The reduction total is wrong, so <code>sample_branch</code>
compares against a corrupted probability and takes the other side — consuming a
different number of PRNG draws from that shot onward.</figcaption>
</figure>

### 11.3 The threshold that was calibrated for the wrong precision

Covered in full in §12, but it belongs in this list because it has the same
shape: `V2_DUST_EPS` was `1e-18`, a sensible clamp for f64 amplitudes, compared
against **f32** amplitudes. The clamp never fired on the GPU where it fired on
the CPU; `sample_branch` returns *without drawing a PRNG value* when a branch is
dust, so one side consumed a random number the other did not, and the streams
desynchronized permanently.

The lesson is not "f32 is imprecise." It is that **a constant outlived the
representation it was chosen for**, and nothing in the type system noticed.

### 11.4 The pitfall that nearly reached this report: a stale specialization cache

This one was found while writing this chapter, and it is the most instructive of
the four because it corrupted *the measurements themselves* rather than the code.

**The mechanism.** `compile_specialized()` computes a content hash to decide
whether a cached `.hsaco` can be reused:

```cpp
// before
std::string ident = csrc + "|" + llvm_bin("clang") + "|" + arch() + "|" + bitcode_dir();
size_t h = std::hash<std::string>{}(ident);
```

`csrc` is the generated C. But the generated C is only *half* the translation
unit — it is a list of `v2_op_*(...)` calls plus `#include "v2_ops.h"`. Every op
body, `v2_barrier()`, and every tunable constant lives in the headers, and **the
headers are not in the hash.** Change `v2_ops.h`, and the key does not move: the
day-old `.hsaco` is reused, *and so is its `<hsaco>.gate` verdict*, which is
keyed off the same path.

**How far it got.** The benchmark run submitted specifically to guarantee that
"every performance claim in the upcoming report reflects shipped, correct code"
dispatched pre-fix kernels for every specialized circuit. Verified on the
binaries with `llvm-objdump`, on two independent markers:

| marker | build cache (what the run used) | fresh cache |
|---|---|---|
| barriers with `ds_*` in flight and no `lgkmcnt(0)` (the `150d09f` fence) | **1400 / 1509 (92.8 %)** | **0 / 1509 (0.0 %)** |
| `s_barrier` preceded by `lgkmcnt(0)` within 3 instructions | **49 / 1509 (3.2 %)** | 1404 / 1509 (93.0 %) |
| `V2_DUST_EPS` constant baked into the binary | **`0x3c32725d…` = `1e-18`** (pre-`2a015fd`) | `0x3da5fd7f…` = `1e-11` (fixed) |

The dust marker is the cleaner of the two because it is a bit pattern, not a
heuristic: the IEEE-754 double `1e-18` has high word `0x3c32725d` and `1e-11`
has `0x3da5fd7f`, and each appears exactly twice in its respective binary and
zero times in the other. There is no interpretation involved — the stale kernel
provably contains the pre-fix constant.

Both markers agree with the timestamps: 32 of the 36 cached kernels are dated
2026-07-25 and the remaining 4 are from 07-26 00:00–01:48 — all of them before
the barrier fix at 07-26 06:33, and well before the dust fix at 07-26 13:31.
Every kernel the run dispatched predates both fixes.

**The consequence for the in-tree conclusions.** The stale run recorded the
`coop_r10_n1720` gate as *failing*, which is a verdict on the **pre-fence**
binary — and the ledger built an entire section around "the specializer is still
incorrect on this shape," listing it as *"the single largest open correctness
item in V2."* It was already fixed. The claim survived only because the cache
kept handing back the binary from before the fix.

**The near-miss that makes this worth reporting.** The hazard was *known*.
`d5_fence.sh` documents it exactly:

> The `.gate` verdict is keyed on the generated-C hash, which does **NOT** change
> when `v2_ops.h` changes — a stale "0" would hide the fix. Each arm uses a fresh
> `V2_SPEC_CACHE_DIR` so every verdict below is recomputed from scratch.

Every `d5_*.sh` diagnostic script therefore sets its own `V2_SPEC_CACHE_DIR` and
got correct results. **The workaround was applied per-script instead of fixing
the cache**, so the one path that did *not* set the variable — the build tree's
default cache, used by the benchmark sweep — silently kept serving stale
kernels. A known bug, correctly documented, routed around rather than fixed, in
the one place it mattered most.

The fix folds a content hash of the three device headers into the cache identity
(`009df59`):

```cpp
// Identity of the DEVICE HEADERS the emitted C includes. The generated .c is
// only half the translation unit: every v2_op_* body, v2_barrier(), and every
// tunable constant (V2_DUST_EPS, ...) lives in these headers. [...]
// Content, not mtime: mtimes change on every checkout and would defeat the
// cache for no reason.
std::string ident = csrc + "|" + llvm_bin("clang") + "|" + arch() + "|" +
                    bitcode_dir() + "|" + device_header_ident();
```

The stale cache was **moved, not deleted**, to
`V2_performance/history/stale_spec_cache_20260725/` — it is the evidence for
this section.

### 11.5 The common thread

| # | plausible story | what it actually was | what settled it |
|---|---|---|---|
| 11.1 | `-O2` reassociates FP across inlined noise ops | a data race between workgroups | kernel disagreed **with itself** (`V2_GATE_SELFTEST`) |
| 11.2 | (see above) | `s_barrier` orders execution, not memory | ISA audit: 92.8 % of barriers unfenced |
| 11.3 | f32 is less precise than f64 | a threshold constant calibrated for f64, compared against f32 | two-arm A/B on the constant |
| 11.4 | the benchmark measures current code | the cache served day-old binaries | `llvm-objdump` on the dispatched `.hsaco` |

Three of the four were settled by an experiment whose outcome the wrong theory
*could not* produce — a self-comparison, a static audit, a controlled A/B. None
were settled by reasoning harder about the plausible story. And the fourth was
caught only because this report's ground rule (*trust data, not text*) required
re-deriving an in-tree claim from the artifact instead of quoting it.

**A fifth, smaller instance belongs here, because it happened while writing this
chapter.** The 95.4 % figure above was quoted from `150d09f`'s commit message
and from a comment in the script that produced it. Re-running the measurement on
the archived binary gave 92.8 %, under every definition tried. The original
number is not recoverable and the script no longer carries the exact counting
code that produced it — the audit trail ends at prose. This is precisely the
failure mode the ground rule exists for, and it is worth noting that it caught a
number in a *correct* section, written by someone who had the artifact in hand at
the time. The conclusion did not change; a number that had been repeated three
times did.

---
