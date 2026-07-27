## 3. The original GPU SVM: one interpreter, three tiers

The first GPU backend is a direct port of the interpreter: one kernel, written
in HIP, compiled once at build time, that walks the same bytecode the CPU walks.
It is the baseline every later generation is measured against, and — three
rewrites later — it is still the thing to beat on one circuit family.

### 3.1 The kernel is a `for(pc) switch`

`src/clifft/gpu/sampler/hip_sampler.hip:728-731`:

```cpp
__device__ void execute_shot(const GpuProgram& program, Rng& rng, ShotState& st) {
    for (uint32_t pc = 0; pc < program.num_instrs; ++pc) {
        const GpuInstr& instr = program.instrs[pc];
        switch (instr.opcode) {
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_CNOT):
                frame_cnot(st, instr.axis_1, instr.axis_2);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_CZ):
                frame_cz(st, instr.axis_1, instr.axis_2);
                break;
            ...
```

Note what the CPU's computed-goto trick has become: a plain `switch`. There is
no indirect-branch predictor on a GPU to reward threading the dispatch. What
this costs shows up directly in the scalar unit, and §14.2 quantifies it.

Two properties of this shape are worth naming now, because V2 preserves one and
destroys the other:

- **Every operand is a runtime load.** `instr.axis_1`, `instr.flags`,
  `instr.mask` — all read from memory, every shot, every instruction.
- **`st.active_k` is a runtime variable.** Every amplitude sweep computes its
  trip count as `1u << (st.active_k - 1)` at runtime, and every scatter index is
  recomputed per amplitude.

V2 keeps the shot loop and kills both of these (§6, §7).

### 3.2 Three tiers, because the working set spans 10^7

A statevector of 1 amplitude and a statevector of 16.7 million cannot use the
same parallelization. The SVM backend therefore ships three kernels, selected by
compiled `peak_rank`:

| tier | rank | topology | amplitude storage | source |
|---|---|---|---|---|
| **register** | ≤ 4 | 1 shot per **thread** | `GpuComplex v[16]` in VGPRs | `hip_sampler.hip:153,1778` |
| **coop** | 5–10 | 1 shot per **workgroup** (256 threads) | `__shared__ GpuComplex v[1024]` | `hip_sampler.hip:1861` |
| **global** | 11–26 | persistent workgroup pool, work-stealing | HBM slice per workgroup | `hip_sampler.hip:1937` |

Constants: `kThreadMaxPeakRank = 4`, `kSharedMaxPeakRank = 10`,
`kGlobalMaxPeakRank = 26` (`gpu_types.h:8-18`).

<figure>
<img src="diagrams/memory-hierarchy-tiers.svg" alt="Three-tier memory hierarchy" width="100%">
<figcaption><b>Figure 3.1</b> — The three tiers map rank onto the memory
hierarchy: VGPRs → LDS → HBM. The tier boundary is where the statevector stops
fitting in the level above.</figcaption>
</figure>

**Register tier** (`hip_sampler.hip:1778`). One thread = one shot. The
statevector is `GpuComplex v[kThreadMaxAmplitudes]` = 16 complex floats, held in
registers. No LDS, no barriers, no cooperation. Block-level reduction of the
per-shot counters happens once at the end via shared memory.

**Coop tier** (`hip_sampler.hip:1861-1885`). One workgroup = one shot; 256
threads cooperate on the amplitude sweeps. The entire shot state moves to LDS:

```cpp
__shared__ GpuComplex v[kSharedMaxAmplitudes];        // 1024 amps  =  8 KB
__shared__ GpuComplex scratch[kSharedMaxAmplitudes/2];//  512 amps  =  4 KB
__shared__ uint8_t    meas[kMaxMeas];                 // 4096 slots =  4 KB
__shared__ uint64_t   px[kPauliWords], pz[kPauliWords];
__shared__ double     red0[256], red1[256];           //            =  4 KB
__shared__ uint32_t   scatter_lut[kSharedMaxAmplitudes/2];
```

That LDS budget is the coop tier's occupancy constraint, and it is where V2's
first two optimizations went (§9.1). Note `scatter_lut`: the SVM caches computed
scatter indices in LDS, keyed by `(axis1, axis2, k, mode)`. This is a *runtime*
answer to the same problem V2 answers at *compile* time (§7.5) — and the SVM
leaves this LUT **off** for the global tier, which is precisely why V2's global
specialization wins so decisively (§9.3).

**Global tier** (`hip_sampler.hip:1937`). The statevector no longer fits in LDS,
so each resident workgroup owns an HBM slice and shots are pulled from per-XCD
work counters. The grid is a fixed *pool* of persistent workgroups, not one
workgroup per shot.

### 3.3 The cooperative reduction

Active-qubit measurement needs `sum |v_i|^2` over each half of the array — a
reduction across all 256 threads. Both backends implement the identical
two-phase pattern: an intra-wavefront butterfly using `ds_bpermute` (64 lanes,
6 steps), then an inter-wavefront combine through LDS
(`hip_sampler.hip:948`, mirrored in `v2_ops.h:224-262`, `coop_reduce2`).

<figure>
<img src="diagrams/warp-shuffle-reduction.svg" alt="Two-phase cooperative reduction" width="100%">
<figcaption><b>Figure 3.2</b> — Phase 1: 6-step XOR butterfly within each
64-lane wavefront via <code>ds_bpermute</code>. Phase 2: the four wavefront
partials combine through LDS. The summation <em>order</em> is part of the ABI —
change it and f64 rounding diverges at measurement branch points.</figcaption>
</figure>

The comment above V2's copy is emphatic and explains a constraint that recurs
throughout this report (`v2_ops.h:235-236`):

> `// MUST reproduce SVM coop_reduce2's exact summation order or f64 rounding`
> `// diverges at measurement branch points.`

A reduction is not associative in floating point. Two implementations that sum
the same values in different orders produce different totals in the last bits,
and a measurement whose probability sits near the sampling threshold will then
branch differently — which, per §2.4, desynchronizes the PRNG forever. **The
summation order is part of the ABI.** This single constraint rules out most of
the obvious "optimizations" one would reach for.

### 3.4 What the SVM backend does well

It is worth stating plainly, because two successor generations failed to beat it:

- **One kernel, compiled once.** No per-circuit compile latency, no cache, no
  gate. Ship it and it runs.
- **The dispatch loop keeps IR small.** The bytecode is *data*. A 16,000-op
  circuit and a 4-op circuit compile to the same kernel. V1 forgot this and paid
  221-second compiles for it (§5.3).
- **It is heavily tuned.** The tag history (`svm-opt-1-noinline-extended`,
  `svm-opt-2-noinline-coop-sweeps`, `svm-opt-3-geak-warp-shuffle`,
  `svm-opt-4-scatter-lut`) records four rounds of optimization: selective
  `noinline` to control register pressure, GEAK-derived warp-shuffle reduction,
  and the scatter LUT. **This is not a strawman baseline.**

### 3.5 What it leaves on the table

Everything the interpreter re-derives at runtime that was already known when the
circuit was compiled:

| known at compile time | paid at runtime, every shot |
|---|---|
| the opcode at each `pc` | a `switch` — scalar branch-target computation |
| `axis_1`, `axis_2`, `flags`, `mask` index | a 32-byte instruction load + field extraction |
| `active_k` before each instruction | a load, then `1u << (k-1)` for every trip count |
| the scatter-index pattern for fixed axes | `scatter_bits_2()` per amplitude — or an LDS LUT with a cache-validity check |
| which of a measurement's two paths is live | both paths emitted, one predicated |

That table is the entire specialization opportunity, and §7 turns each row into
a measured experiment.

---

## 4. Hybrid: compiling the circuit into HIP, and why it stalled

### 4.1 The idea

If the operands are known when the circuit is compiled, generate a bespoke
kernel per circuit. The Hybrid backend (`src/clifft/gpu/codegen/`) does exactly
that: it walks the bytecode, emits HIP source with the operands as literals, and
compiles it ahead of the sampling run.

The emitter is structured around a **used-function analysis**
(`kernel_codegen.h:17-50`): a `UsedFunctions` struct with one boolean per opcode
family, populated by scanning the bytecode, used to emit only the device
functions the circuit actually needs. A pure-Clifford circuit does not carry the
`ARRAY_U4` code at all.

### 4.2 The compile path — and the first appearance of an HSA/HIP split

`kernel_cache.cc:105-170`:

```cpp
// Step 1: Compile HIP source → offload bundle (.bundle)
// --offload-device-only produces a Clang Offload Bundle, not a raw ELF.
// HSA needs a raw AMDGCN ELF, so we unbundle in step 2.
cmd << clangpp << " -x hip"
    << " --offload-arch=" << gpu_arch
    << " -O2 -ffast-math"
    << " --offload-device-only"
    << " -o " << bundle_path << " " << tmp_src;
```

then `clang-offload-bundler --unbundle` to extract the raw ELF, then
`hsa_load_kernel(hsaco_path, kernel_func_name, 0)` (`kernel_cache.cc:191`).

**Hybrid already dispatches through HSA, not `hipModuleLoad`.** It uses HIP only
as a *source language*, and then has to fight its way back out of HIP's
packaging to get an object HSA will accept. That awkwardness — writing HIP to
get a device binary, then unwrapping the HIP container to hand it to HSA — is
the seed of V2's decision to drop HIP entirely (§13).

Note also `-ffast-math`. Hybrid was allowed reassociation; V2 explicitly is not
(`-ffp-contract=off`, no fast-math, §6.4), because V2's correctness contract is
byte-exactness against the interpreter and fast-math would break it.

### 4.3 The result: correct, competitive, never ahead

The v1-era measurement (`docs/v2/reference_v1.md:270-278`, 10,000 shots,
MI350X, `sample_seconds`) shows Hybrid at or slightly better than SVM:

| circuit | SVM | **Hybrid** | MLIR (v1) |
|---|---|---|---|
| frame_h | 0.066 | **0.053** | 0.271 |
| qv10 | 0.074 | **0.058** | 3.54 |
| circuit_d5 | 0.078 | **0.065** | 1.22 |
| surface_d9_t10 | 0.090 | 0.112 | 5.50 |

> **[unverified, v1-era]** — these are `sample_seconds` (host wall time
> including setup), not isolated kernel time, from a run whose node identity was
> not recorded. They are quoted for their *shape*, not their magnitude: Hybrid
> tracks SVM within ±25%, and MLIR is off the chart. That shape is corroborated
> by every later measurement.

Hybrid wins by 10–20% on three of four and loses by 25% on the fourth. After
per-circuit code generation, an entire compile-cache subsystem, and an
offload-bundle unwrapping dance, the answer is **noise-level**.

### 4.4 Why it stalled — the density measurement

The cleanest diagnosis comes from an artifact built for this report:
`V2_performance/lowering/ir_density.csv`, which counts lines of emitted source
per bytecode instruction.

| circuit | instrs | Hybrid HIP lines | **lines/instr** |
|---|---|---|---|
| reg_circuit_d3 | 344 | 5,301 | 15.4 |
| coop_qv10 | 140 | 7,805 | 55.8 |
| coop_circuit_d5 | 1,720 | 33,508 | 19.5 |
| coop_surface_d7_t10 | 4,134 | 75,409 | 18.2 |
| glob_surface_d7_t19 | 4,296 | 75,794 | 17.6 |

Hybrid emits a **block** per instruction — 15 to 56 lines of HIP each. It is
linear in circuit size with a large constant. That is not pathological (V1's is,
§5), but it means:

- the compiler's problem grows with the circuit, so compile time does too;
- register pressure grows with the circuit, because the body is one long
  straight-line region;
- the win from constant operands is diluted by the sheer volume of code the
  optimizer must chew through.

**The lesson Hybrid teaches, and the reason it is in this report:** per-circuit
compilation is not, by itself, a performance strategy. What matters is *what you
emit per instruction*. Hybrid emitted a block. V2 emits a call (§6.3), and the
same idea that netted Hybrid ~0 % nets V2 **1.3–4.0×**.

Note the shape of the density column, because it is the whole argument in one
number: Hybrid's lines-per-instruction is 15.4–55.8 and does not improve with
scale, while V2's converges *downward* to 1.00–1.27 on every circuit large
enough to matter — 1.00–1.11 on the three Clifford-dominated ones, and 1.27 on
`qv10`, whose fused unitaries are the one gate family V2 still emits more than
one line for. Same input, same target, same compiler — a 14–44× difference in
how much code the optimizer is handed.

---
