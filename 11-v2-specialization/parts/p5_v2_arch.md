## 6. V2: the architecture

### 6.1 The one rule, stated precisely

V1's disease was unrolling the operand **sequence**. The naive lesson — "don't
unroll anything" — is wrong and would give up the entire specialization
opportunity. V2's design document (`docs/v2/V2.md:104-132`, §3.1) is precise
about three *different* loops with *different* rules:

| loop | rule | why |
|---|---|---|
| **over shots** | **ALWAYS a runtime loop** | shots are data; a persistent kernel with work-stealing pulls them |
| **over the bytecode** | **ALWAYS a runtime loop*** | the operand sequence is data; unrolling it is the disease |
| **over amplitude groups** | **stays a loop**, but bounds/constants may specialize | at rank 10 that is 256–512 groups; flattening reinstates the bloat |
| **the 2×2 / 4×4 butterfly** | **fully unrolled** | a constant 4–16 multiply-adds; no size risk |

*The asterisk on the bytecode loop is the whole invention. V2 does not *unroll*
the bytecode loop — it **removes** it, by emitting one **call** per instruction
instead of one **block**. The distinction sounds like hair-splitting and is the
entire result: a call site is O(1) IR regardless of what the callee does. §6.3
measures it.

<figure>
<img src="diagrams/three-loops.svg" alt="The three loops and their rules" width="100%">
<figcaption><b>Figure 6.1</b> — The three nested loops. V1 flattened the middle
one into code. V2 replaces it with a call sequence whose <em>operands</em> are
constants but whose <em>bodies</em> are shared and still contain their own
loops.</figcaption>
</figure>

### 6.2 One operand library, compiled two ways

The substrate is `src/clifft/gpu/mlir/v2/v2_ops.h` + `v2_ops_body.inc`: every
opcode body as a `static inline` function taking its operands **and the pre-op
`active_k`** by value.

```c
static inline void v2_op_array_cnot(V2State* st, CV2Complex* v,
                                    u32 active_k, u32 a1, u32 a2);
static inline void v2_op_meas_dormant_static(V2State* st, u32 axis,
                                             u32 slot, u32 flags);
```

That signature is the whole trick. The **same source** serves two consumers:

- the **interpreter** (`coop_interpreter.c`) passes runtime values —
  `v2_op_array_cnot(st, v, st->active_k, in.axis_1, in.axis_2)`;
- the **specializer** (`v2_specializer.cc`) emits the same call with literals —
  `v2_op_array_cnot(st, v, 8u, 2u, 5u)`.

**Byte-exactness is by construction, not by testing.** The bodies are the same
bodies. This directly retires V1's duplication tax (§5.6).

The tier dimension is handled by preprocessor parameterization of *cooperation*,
not of arithmetic (`v2_ops.h:128-152`, quoted verbatim with the barrier body
elided to §11.2):

```c
#ifdef V2_REGISTER
#  define V2_STRIDE 1u
static inline u32 v2_tid(void)  { return 0u; }
static inline void v2_barrier(void) {}            // no cooperation, no barrier
#  define V2_REDUCE2(T, L0, L1, O0, O1) do { *(O0) = (L0); *(O1) = (L1); } while (0)
#  define IS_OWNER 1
#else
#  define V2_STRIDE 256u
static inline u32 v2_tid(void)  { return __builtin_amdgcn_workitem_id_x(); }
static inline void v2_barrier(void) { /* release fence; s_barrier; acquire fence — §11.2 */ }
#  define V2_REDUCE2(T, L0, L1, O0, O1) coop_reduce2((T), (L0), (L1), (O0), (O1))
#  define IS_OWNER (t == 0)
#endif
```

`V2_REDUCE2` in the register tier is not a stub for "reduction not needed" — it
is an assignment. A stride-1 loop over all amplitudes has already accumulated
both partial sums into `L0`/`L1`, so the cross-lane reduction degenerates to a
copy. Same call site, same argument list, two different meanings, zero runtime
branch.

**The same opcode arithmetic compiles two ways from one source.** Register tier:
1 shot/thread, private state, stride 1, reduction is the identity because a
stride-1 loop has already summed everything. Coop/global: 256 threads/shot,
stride 256, fenced barriers, butterfly reduction.

### 6.3 What the specializer emits

`v2_specializer.cc:26-97` is a `switch` over opcodes that prints exactly one
line each — 35 of the 41 opcodes are handled; the six that are not (the five
`*_FORCED` variants and `OP_EXP_VAL`) `return false`, which aborts emission and
falls the whole circuit back to the interpreter (`v2_specializer.cc:92-93`).

The register-tier `circuit_d3` output (`lowering/v2_src/reg_circuit_d3.c`, 383
lines for 344 instructions) is reproduced verbatim below, lines 1–3 and 15–25
and 42–49 and 360–361, with `…` marking the elisions:

```c
#define V2_REGISTER 1
#define V2_NOISE_ATTR __attribute__((noinline))
#include "clifft/gpu/mlir/v2/v2_ops.h"
…
    v2_shot_init(st, v, amp_capacity, shot_id, num_observables, seed, noise_hazards, num_noise_sites);
    v2_op_meas_dormant_static(st, 6u, 33u, 0u);
    v2_op_meas_dormant_static(st, 8u, 32u, 0u);
    v2_op_meas_dormant_static(st, 11u, 31u, 0u);
    v2_op_meas_dormant_random(st, 13u, 30u, 0u);
    v2_op_meas_dormant_random(st, 5u, 29u, 0u);
    v2_op_meas_dormant_random(st, 0u, 28u, 0u);
…
    v2_op_noise_block(st, noise_sites, noise_channels, noise_hazards, num_noise_sites, 0u, 131u);
    v2_op_frame_cnot(st, 3u, 10u);
    v2_op_frame_cnot(st, 3u, 12u);
    v2_op_frame_cz(st, 3u, 6u);
    v2_op_frame_cz(st, 3u, 8u);
    v2_op_frame_swap(st, 3u, 0u);
    v2_op_frame_h(st, 0u);
    v2_op_expand_t(st, v, 0u, 0u, 1);          // <- active_k = 0, tracked statically
…
    v2_shot_aggregate(st, block_counts, num_observables, expected_obs_mask);
}
```

**Count check.** Lines 16–360 of that file contain exactly **344** `v2_op_*`
call statements, one per bytecode instruction, no more and no fewer. The 39
non-call lines are the 14-line `spec_body` signature, the `v2_shot_init` /
`v2_shot_aggregate` bracket, and the 21-line kernel wrapper at 363–383.

Every operand is a literal. Note `v2_op_expand_t(st, v, **0u**, 0u, 1)`: the
third argument is `active_k`, which the specializer tracks statically as it
walks the program. The mechanism is visible in the emitter itself — every
rank-changing case both *prints* the current `*k` and *updates* it
(`v2_specializer.cc:42-56`):

```cpp
    case Opcode::OP_EXPAND:
        o << "v2_op_expand(st, v, " << *k << "u);"; ++*k; break;
    case Opcode::OP_EXPAND_T:
        o << "v2_op_expand_t(st, v, " << *k << "u, " << a1 << "u, 0);"; ++*k; break;
    …
    case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
        o << "v2_op_meas_active_diagonal(st, v, " << *k << "u, " << a1 << "u, "
          << a << "u, " << flags << "u);";
        if (*k) --*k; break;
```

**The interpreter must load `st->active_k` after every rank-changing op; the
specializer knows it at emission time.** §7.3 measures what that alone is worth.

Two details in that switch are worth noticing because they are free wins that
fall out of the "one call per instruction" shape:

- `OP_FRAME_S` and `OP_FRAME_S_DAG` emit the *same* call
  (`v2_specializer.cc:35-36`); so do `OP_ARRAY_S`/`OP_ARRAY_S_DAG` and
  `OP_ARRAY_T`/`OP_ARRAY_T_DAG`, differing only in a literal `0`/`1` dagger flag.
  Six opcodes collapse to three call sites with a constant argument — which the
  optimizer then resolves to two distinct specializations of each body.
- `OP_DETECTOR` emits `v2_op_detector();` — a call taking *no arguments at all*.
  The detector's work is entirely compile-time in the specialized path.
- `OP_POSTSELECT` is the one case that emits control flow rather than a
  statement: `if (v2_op_postselect(...)) return;`. A shot that fails
  post-selection exits `spec_body` directly, with no flag to test at the next
  instruction.

**The measured invariant.** `V2_performance/lowering/ir_density.csv`:

| circuit | instrs | v2 C lines | **lines/instr** |
|---|---|---|---|
| reg_frame_h | 4 | 43 | 10.75 |
| reg_four_t | 11 | 49 | 4.45 |
| reg_circuit_d3 | 344 | 383 | **1.11** |
| coop_qv10 | 140 | 179 | **1.27** |
| coop_circuit_d5 | 1,720 | 1,760 | **1.02** |
| coop_surface_d7_t10 | 4,134 | 4,174 | **1.00** |
| glob_surface_d7_t19 | 4,296 | 4,346 | **1.01** |

**One line of source per bytecode instruction, plus a fixed ~40-line kernel
wrapper.** The two register-tier outliers are entirely that wrapper amortized
over 4 and 11 instructions; by 344 it has vanished into the noise.

Against V1's 55–2,407 lines/instr and Hybrid's 15–56, this is a different
*asymptotic class* of code generator, not a tuning improvement.

### 6.4 The compile pipeline — no MLIR, no HIP

The directory is named `mlir/v2/`. **That name is the only MLIR left in V2.**
`v2_specializer.cc:127` emits a C translation unit whose first include is
`"clifft/gpu/mlir/v2/v2_ops.h"`, and `v2_compile_cache.cc:151-176` compiles it
with the raw LLVM toolchain:

```cpp
// 1) C -> amdgcn bitcode. Flags are IDENTICAL to the build-time interpreter
// pipeline (ClifftAmdgcn.cmake) so the shared v2_op_*() bodies compile to the
// same code -> byte-exact. -ffp-contract=off with no fast-math is what makes
// that hold regardless of inlining decisions: without reassociation or
// contraction the optimizer cannot change an FP result.
clang --target=amdgcn-amd-amdhsa -mcpu=gfx950 \
      -ffreestanding -nostdlib -nogpulib -std=c23 -O2 -ffp-contract=off \
      -emit-llvm -c -o spec.bc spec.c

// 2) link ocml + oclc controls (identical set to ClifftAmdgcn.cmake)
llvm-link -o linked.bc spec.bc ocml.bc ockl.bc \
      oclc_wavefrontsize64_on.bc oclc_daz_opt_off.bc oclc_finite_only_off.bc \
      oclc_unsafe_math_off.bc oclc_correctly_rounded_sqrt_on.bc \
      oclc_abi_version_500.bc oclc_isa_version_950.bc

opt -O2 -o linked.bc linked.bc                                    // 3
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
    -mattr=+wavefrontsize64 -filetype=obj -O2 -o spec.o linked.bc  // 4
ld.lld -shared -o spec.hsaco spec.o                                // 5
```

Three things about this deserve emphasis.

**No HIP anywhere.** Not as a source language, not as a runtime, not as a
packaging format. Contrast Hybrid, which had to emit HIP, produce a Clang
Offload Bundle, then unbundle it to get an ELF that HSA would accept (§4.2). V2
targets `amdgcn-amd-amdhsa` directly and gets a plain ELF from `ld.lld`. §13
covers why this matters beyond aesthetics.

**The flags are identical to the build-time interpreter's**, deliberately and
with a comment that reads as an instruction to future maintainers
(`v2_compile_cache.cc:155-156`):

> `Do NOT add vectorize/unroll/fast-math flags that would diverge from the interpreter.`

`-ffp-contract=off`, no fast-math: without reassociation or FMA contraction,
**the optimizer cannot legally change a floating-point result**, so the
specialized kernel and the interpreter agree bit for bit no matter what inlining
decisions differ between them. Hybrid did the opposite — `kernel_cache.cc:115`
compiles its HIP with `-O2 -ffast-math`. V2 cannot, and the reason is §2.4: a
1-ULP difference flips a sampling branch and desynchronizes the PRNG forever.
Hybrid got away with it because Hybrid was never byte-compared against the CPU
reference.

Note also `-mattr=+wavefrontsize64` at step 4 and `oclc_wavefrontsize64_on.bc`
at step 2. gfx950 supports wave32; V2 pins wave64 in both the device-library
control set and the backend because the coop tier's `V2_STRIDE 256u` and its
`V2_RED_WARPS` reduction geometry are written for 64-lane waves.

**Compilation is cached and never on the sampling path**, and the cache key is
the part that has already gone wrong once. `v2_compile_cache.cc:125-127`:

```cpp
std::string ident = csrc + "|" + llvm_bin("clang") + "|" + arch() + "|" +
                    bitcode_dir() + "|" + device_header_ident();
size_t h = std::hash<std::string>{}(ident);
```

That last term is a repair. The key originally hashed only `csrc` — the
*generated* C — which is barely half the translation unit: every `v2_op_*` body,
`v2_barrier()`, and every tunable constant lives in the included headers. A
header fix therefore produced the same key, and the stale `.hsaco` **plus its
stale `.gate` verdict** silently won. §11.1 documents the benchmark run this
destroyed. The fix hashes the contents of all three device headers
(`v2_ops.h`, `v2_ops_body.inc`, `device_abi.h`) into the key — content, not
mtime, since mtimes change on every checkout and would defeat the cache for no
reason.

### 6.5 The correctness gate

Specialization is a compiler, and compilers have bugs. V2 does not ask you to
trust it: `v2_kernel.cc:370-398` validates each freshly compiled `.hsaco`
against the interpreter on a shot sample, and **persists the verdict to disk**:

```cpp
// The verdict is written to "<spath>.gate" so it is computed ONCE ever (not
// per process) and NEVER re-run during a profiled sample dispatch -- the
// gate's own validation dispatches would otherwise pollute rocprofv3 traces.
std::string gate_path = spath + ".gate";
if (std::FILE* gf = std::fopen(gate_path.c_str(), "r")) {   // disk cache hit
    int v = 0; if (std::fscanf(gf, "%d", &v) != 1) v = 0;
    validated = (v == 1);
} else {                                                     // compute once
    validated = specialized_matches_interpreter(program, spath, spec_sym);
    ...
}
if (validated) { load_path = spath; sym = spec_sym.c_str(); }
else spec_sym.clear();                                       // -> interpreter
```

On failure the circuit silently falls back to the interpreter — and so does a
thrown exception anywhere in emission or compilation (`v2_kernel.cc:399-403`),
which is what makes the six unsupported opcodes safe rather than fatal. **This
guarantees byte-exactness for every circuit, not just the ones that happened to
be tested.**

Three design details are load-bearing and easy to miss:

1. **The verdict caches in three places, not one** — an in-process
   `static std::map` keyed by circuit shape, then the on-disk `.gate` file,
   then the `.hsaco` itself. The comment explains the third one:
   *"NEVER re-run during a profiled sample dispatch — the gate's own validation
   dispatches would otherwise pollute rocprofv3 kernel traces."* A correctness
   check that perturbs the measurement is a correctness check you stop running.
2. **The key is the circuit *shape*, not its name** — `tname + "_r" + peak_rank
   + "_n" + num_instrs` (`v2_kernel.cc:366-367`). `circuit_d5_p0.001` and
   `cultivation_d5` compile to the same `coop_r10_n1720` key when their
   bytecode has the same rank and length, so they share one gate verdict and one
   `.hsaco`. That is why §11.1's failure took out a whole *family* at once.
3. **The gate does not tell you *what* is wrong**, only that something is. Both
   diagnostics that exist (`V2_GATE_VERBOSE`, `V2_GATE_BISECT`) had to be added
   during §11.2's investigation.

**Verdicts on disk (2026-07-27, `build-v2-nohip/v2_spec_cache/`): 16 of 16
pass.**

```
$ for f in build-v2-nohip/v2_spec_cache/*.gate; do echo "$(cat $f) $(basename $f .hsaco.gate)"; done
1 coop_r10_n140_7b3194ac269d23a9
1 coop_r10_n140_ad02701c918c6f1b
1 coop_r10_n1720_48e8389e3cb74acb      <- the shape that used to fail
1 coop_r10_n1720_847f167402847a8b      <- and its sibling hash
1 coop_r10_n4371_e333cefd1e929cb
1 coop_r7_n16374_b38b86d5770ce20a
1 coop_r7_n4134_197aaf5bcd540fb1
1 coop_r7_n4134_60919e543cb4cced
1 coop_r7_n8859_74c4fcdbba5a0374
1 reg_r0_n11_dc388d7dcd2a8cf0
1 reg_r0_n4_47d470a8aeec1399
1 reg_r0_n4_833175afc86f6aea
1 reg_r3_n4151_3a83fd16f93a5eea
1 reg_r3_n4151_f11fad7b253aa1df
1 reg_r4_n344_7378b9a76010d9be
1 reg_r4_n344_bcf40c29ff41f709
```

Note that shapes now appear in pairs under two different hashes. The shape key
(`coop_r10_n1720`) identifies the *circuit*; the trailing hash covers the
toolchain, arch, bitcode directory and device-header contents, so a header edit
legitimately produces a second entry for the same circuit. That is
`device_header_ident` doing exactly the job §6.4 describes.

Read that against the quarantined cache the report's earlier benchmark actually
ran (`V2_performance/history/stale_spec_cache_20260725/`, 36 verdicts):

| cache | verdicts | failures |
|---|---|---|
| `history/stale_spec_cache_20260725/` (pre-fence, pre-dust) | 36 | **2** — `coop_r10_n1720_977e1e83…`, `coop_r10_n1720_cadfdf19…` |
| `build-v2-nohip/v2_spec_cache/` (current HEAD) | 16 | **0** |

Two things follow, and both matter for how the rest of this report is read.
**First, the gate failure is fixed**: the shape that produced the only two zeros
in the corpus now produces a one. §11.2 is the story of what was actually wrong.
**Second, the two caches are not comparable on count** — 36 versus 16 is not a
shrinking corpus, it is a cache that was rebuilt from scratch after `009df59`
changed the key, and has since only been populated by the circuits that have
been re-run. It was at 6 entries earlier on the day this section was written and
is at 16 now, growing as §15's full-corpus run repopulates it.

### 6.6 The specializer is opt-in

`getenv("V2_SPECIALIZE")` (`v2_kernel.cc:358`). Unset, V2 runs its bytecode
interpreter.

This is a real hazard when reading any V2 benchmark. **With `V2_SPECIALIZE`
unset, a benchmark measures V2's interpreter against SVM's tuned interpreter and
reports what looks like a 2–3× regression.** Every number in this report was
produced with it set; §11.1 shows what happens when the gate un-sets it for you.

### 6.7 The three tier wrappers

One `spec_body`, three wrappers (`v2_specializer.cc:172-225`). `spec_body` takes
21 parameters; the first five are the tier-dependent ones (state, amplitudes,
scratch, capacity, shot id) and the remaining 16 are identical across tiers —
emitted once into a `const char* fwd` at `v2_specializer.cc:166-170` and reused
verbatim by each branch, so the tier choice cannot accidentally change what the
body receives.

Every wrapper opens with the same line:

```c
    (void)peak_rank; (void)num_instrs; (void)instrs; (void)total_meas_slots;
```

Those four kernel arguments are what the *interpreter* runs on: the bytecode
pointer, its length, the rank, and the measurement-slot count. The specialized
kernel takes them because the kernarg layout must stay identical (one dispatch
path, two kernels), and then **discards all four**. That single line is the
specialization thesis in the ABI: the program is no longer an input.

**Register** — one shot per thread, statevector in VGPRs:

```c
#define V2_REG_MAX_AMP 16
__attribute__((amdgpu_kernel, visibility("default")))
void clifft_v2_spec(...) {
    u64 shot_id = shot_offset + ((u64)v2_bid() * 256ul + __builtin_amdgcn_workitem_id_x());
    if (shot_id >= shots) return;
    V2State st; CV2Complex vloc[V2_REG_MAX_AMP]; CV2Complex sloc[V2_REG_MAX_AMP/2];
    spec_body(&st, vloc, sloc, V2_REG_MAX_AMP, shot_id, ...);
}
```

**Coop** — one workgroup per shot, everything in LDS:

```c
extern __attribute__((address_space(3))) CV2Complex lds_v[V2_MAX_AMP];
extern __attribute__((address_space(3))) CV2Complex lds_red_scratch[V2_SCRATCH_AMP];
extern __attribute__((address_space(3))) V2State lds_state;
void clifft_v2_spec(...) {
    u64 shot_id = shot_offset + (u64)v2_bid();
    if (shot_id >= shots) return;
    spec_body((V2State*)&lds_state, (CV2Complex*)lds_v, (CV2Complex*)lds_red_scratch, ...);
}
```

**Global** — a persistent pool draining shots from an atomic counter, with the
amplitude capacity folded to a literal:

```c
void clifft_v2_spec(..., CV2Complex* global_v, CV2Complex* global_scratch, u64* work_counter) {
    u32 t = v2_tid(), slot = v2_bid();
    const u64 amp_capacity = 8192ull;                    // <- 1<<peak_rank, a constant
    CV2Complex* v       = global_v       + (u64)slot * amp_capacity;
    CV2Complex* scratch = global_scratch + (u64)slot * (amp_capacity >> 1);
    for (;;) {
        if (t == 0) lds_shot = __atomic_fetch_add(&work_counter[0], 1UL, __ATOMIC_RELAXED);
        v2_barrier();
        u64 batch_shot = lds_shot;
        if (batch_shot >= shots) return;
        spec_body((V2State*)&lds_state, v, scratch, (u32)amp_capacity, shot_offset + batch_shot, ...);
        v2_barrier();
    }
}
```

<figure>
<img src="diagrams/persistent-kernel.svg" alt="Persistent kernel with work-stealing" width="100%">
<figcaption><b>Figure 6.2</b> — The global tier's persistent kernel. A fixed
pool of resident workgroups, each owning an HBM amplitude slice, drains shots
from an atomic counter. The grid does not scale with shot count.</figcaption>
</figure>

The resident pool is sized from an **HBM budget** rather than from the rank cap
(`v2_kernel.cc:436-445`):

```cpp
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

`GpuComplex` is `{float re; float im;}` — 8 bytes (`gpu_types.h:38-41`) — so
`bytes_per_wg` is a slice plus a half-size scratch, i.e. **12 bytes per
amplitude**. Three clauses, each doing distinct work: the budget divide, a floor
of 1 so rank 26 is *possible* rather than *fast*, a ceiling of 2,048, and a
round-down to a multiple of `kNumXCDs = 8` so the resident pool spreads evenly
across the device's eight compute dies.

Evaluating that arithmetic gives the resident pool at every global-tier rank:

| peak rank | MiB / workgroup | resident workgroups | regime |
|---|---|---|---|
| 11–19 | 0.02 – 6.00 | **2048** | ceiling-limited |
| 20 | 12.00 | **2048** | ceiling-limited (last one) |
| 21 | 24.00 | **1360** | budget-limited |
| 22 | 48.00 | **680** | budget-limited |
| 23 | 96.00 | **336** | budget-limited |
| 24 | 192.00 | **168** | budget-limited |
| 25 | 384.00 | 80 | budget-limited |
| 26 | 768.00 | 40 | budget-limited |

**Rank 20 is the knee.** Below it the pool is flat at the 2,048 ceiling and rank
costs nothing in occupancy; from 21 up, every added qubit halves the resident
pool. §10 shows this table's five measurable rows (20–24) reproducing exactly in
the observed `Grid_Size_X ÷ 256`, and §15 shows the performance decay that
follows from it — a geometric loss that is *by design*, not a bug, and is why
§10 covers what made rank > 19 possible at all rather than what made it fast.

---
