## 6. V2: the architecture

### 6.1 The one rule, stated precisely

V1's disease was unrolling the operand **sequence**. The naive lesson — "don't
unroll anything" — is wrong and would give up the entire specialization
opportunity. V2's design document (`docs/v2/V2.md:104-133`) is precise about
three *different* loops with *different* rules:

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
not of arithmetic (`v2_ops.h:116-139`):

```c
#ifdef V2_REGISTER
#  define V2_STRIDE 1u
   static inline u32 v2_tid(void) { return 0u; }
   static inline void v2_barrier(void) {}          // no cooperation, no barrier
#  define V2_REDUCE2(T,L0,L1,O0,O1) /* identity: stride-1 loop already summed */
#  define IS_OWNER 1
#else
#  define V2_STRIDE 256u
   static inline u32 v2_tid(void) { return __builtin_amdgcn_workitem_id_x(); }
   static inline void v2_barrier(void) { /* fenced s_barrier, see §11.2 */ }
#  define V2_REDUCE2(T,L0,L1,O0,O1) coop_reduce2((T),(L0),(L1),(O0),(O1))
#  define IS_OWNER (t == 0)
#endif
```

**The same opcode arithmetic compiles two ways from one source.** Register tier:
1 shot/thread, private state, stride 1, reduction is the identity because a
stride-1 loop has already summed everything. Coop/global: 256 threads/shot,
stride 256, fenced barriers, butterfly reduction.

### 6.3 What the specializer emits

`v2_specializer.cc:26-90` is a `switch` over opcodes that prints one line each.
The register-tier `circuit_d3` output (`lowering/v2_src/reg_circuit_d3.c`, 383
lines for 344 instructions) begins:

```c
#define V2_REGISTER 1
#define V2_NOISE_ATTR __attribute__((noinline))
#include "clifft/gpu/mlir/v2/v2_ops.h"

static void spec_body(V2State* st, CV2Complex* v, CV2Complex* scratch, ...) {
    v2_shot_init(st, v, amp_capacity, shot_id, num_observables, seed, ...);
    v2_op_meas_dormant_static(st, 6u, 33u, 0u);
    v2_op_meas_dormant_static(st, 8u, 32u, 0u);
    v2_op_meas_dormant_random(st, 13u, 30u, 0u);
    v2_op_apply_pauli(st, pauli_masks, 0u, 21u);
    v2_op_noise_block(st, noise_sites, noise_channels, noise_hazards,
                      num_noise_sites, 0u, 131u);
    v2_op_frame_cnot(st, 3u, 10u);
    v2_op_frame_cz(st, 3u, 6u);
    v2_op_frame_swap(st, 3u, 0u);
    v2_op_frame_h(st, 0u);
    v2_op_expand_t(st, v, 0u, 0u, 1);          // <- active_k = 0, tracked statically
    ...
    v2_shot_aggregate(st, block_counts, num_observables, expected_obs_mask);
}
```

Every operand is a literal. Note `v2_op_expand_t(st, v, **0u**, 0u, 1)`: the
third argument is `active_k`, which the specializer tracks statically as it
walks the program (`v2_specializer.cc:43-79` — `++*k` on every `EXPAND*`,
`--*k` on every rank-reducing measurement). **The interpreter must load
`st->active_k` after every rank-changing op; the specializer knows it.** §7.3
measures what that alone is worth.

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
`"clifft/gpu/mlir/v2/v2_ops.h"`, and `v2_compile_cache.cc:122-141` compiles it
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
with a comment saying so. `-ffp-contract=off`, no fast-math: without
reassociation or FMA contraction, **the optimizer cannot legally change a
floating-point result**, so the specialized kernel and the interpreter agree bit
for bit no matter what inlining decisions differ between them. Hybrid used
`-ffast-math`; V2 cannot, and the reason is §2.4 — a 1-ULP difference flips a
sampling branch and desynchronizes the PRNG forever.

**Compilation is cached and never on the sampling path.** The emitted C is
content-hashed; a cache hit skips straight to `hsa_load_kernel`.

### 6.5 The correctness gate

Specialization is a compiler, and compilers have bugs. V2 does not ask you to
trust it: `v2_kernel.cc:369-400` validates each freshly compiled `.hsaco`
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

On failure the circuit silently falls back to the interpreter. **This guarantees
byte-exactness for every circuit, not just the ones that happened to be
tested.** It is also why one benchmark family in this report measures something
other than what its label suggests — the gate fired, and §11.1 is the
consequence.

Current verdicts on disk: **34 pass, 2 fail** out of 36.

### 6.6 The specializer is opt-in

`getenv("V2_SPECIALIZE")` (`v2_kernel.cc:358`). Unset, V2 runs its bytecode
interpreter.

This is a real hazard when reading any V2 benchmark. **With `V2_SPECIALIZE`
unset, a benchmark measures V2's interpreter against SVM's tuned interpreter and
reports what looks like a 2–3× regression.** Every number in this report was
produced with it set; §11.1 shows what happens when the gate un-sets it for you.

### 6.7 The three tier wrappers

One `spec_body`, three wrappers (`v2_specializer.cc:172-227`).

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

The resident pool is sized from an HBM budget rather than from the rank cap
(`v2_kernel.cc:436-446`) — 32 GB, `bytes_per_wg = 2^rank × 12`, capped at 2,048
workgroups and rounded down to a multiple of the 8 XCDs so the pool spreads
evenly across the device. §10 covers why that sizing rule is what made rank > 19
possible at all.

---
