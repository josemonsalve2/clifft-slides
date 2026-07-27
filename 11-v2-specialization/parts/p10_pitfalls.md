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

   | comparison | seeds diverging |
   |---|---|
   | interpreter vs itself | 0 / 6 |
   | **specialized vs itself** | **4 / 6** |

   No rounding model explains a binary disagreeing with itself. This is a race.

`V2_GATE_BISECT` then localized it: **zero divergences at one workgroup.** The
bug needed *concurrent workgroups* — which also rules out any single shot's
arithmetic.

The A/B on the fix itself was the last nail: `V2_SPEC_NOISE_INLINE=1`, which
flips `V2_NOISE_ATTR` back to `always_inline`, made things **worse** (5/6 seeds
diverging rather than 4/6) — consistent with perturbing a race's timing, and
inconsistent with the attribute being the load-bearing fix.

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
preceding `lgkm` wait:

| kernel | unfenced barriers |
|---|---|
| interpreter | 52 / 81 (64.2 %) |
| **specialized** | **1439 / 1509 (95.4 %)** |

**Both kernels were wrong.** The specializer is ~27× more exposed because it
straight-lines every op, which is precisely why only it failed the correctness
gate — the interpreter was equally incorrect in principle and simply lost the
race far less often.

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

Post-fix the interpreter goes 0 % → 91.2 % fenced (the residual 7 are barriers
with no LDS write pending, where the wait is correctly elided). Cost: **+3.88 %
static instructions** — the entire price of correctness here.

Verified independently on the artifacts for this report:

| `.hsaco` | barriers | fenced |
|---|---|---|
| pre-fix `coop_r10_n1720` | 1,509 | **0 (0.0 %)** |
| post-fix `coop_r10_n1720` | 1,509 | **1,400 (92.8 %)** |

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
| `s_barrier` preceded by `lgkmcnt(0)` (the `150d09f` fence) | **0 / 1509 (0.0 %)** | 1400 / 1509 (92.8 %) |
| `V2_DUST_EPS` constant baked into the binary | **`1e-18`** (pre-`2a015fd`) | `1e-11` (fixed) |

Both markers agree with the timestamps: all 36 cached kernels are dated
2026-07-25, while the barrier fix landed 07-26 06:33 and the dust fix 07-26
13:34.

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
| 11.2 | (see above) | `s_barrier` orders execution, not memory | ISA audit: 95.4 % of barriers unfenced |
| 11.3 | f32 is less precise than f64 | a threshold constant calibrated for f64, compared against f32 | two-arm A/B on the constant |
| 11.4 | the benchmark measures current code | the cache served day-old binaries | `llvm-objdump` on the dispatched `.hsaco` |

Three of the four were settled by an experiment whose outcome the wrong theory
*could not* produce — a self-comparison, a static audit, a controlled A/B. None
were settled by reasoning harder about the plausible story. And the fourth was
caught only because this report's ground rule (*trust data, not text*) required
re-deriving an in-tree claim from the artifact instead of quoting it.

---
