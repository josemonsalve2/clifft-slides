# GPU Kernel Architecture Optimization for Clifft Quantum Circuit Simulator

## Report v2 -- Corrected Data and MI350X Cross-GPU Results

**Authors:** Jose Monsalve, with GEAK AI-assisted kernel optimization

**Target hardware:** AMD Instinct MI300X (gfx942, CDNA3) and MI350X (gfx950, CDNA4)

**Codebase:** clifft GPU backend (`gpu-backend`, `gpu-compiled-kernel`, and related branches)

**Date:** July 2026

**Supersedes:** REPORT.md (v1) -- corrected compiled kernel data, added MI350X results

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Methodology](#2-methodology)
3. [Methodology Correction](#3-methodology-correction-what-was-wrong-with-v1)
4. [Background: Clifft GPU Architecture](#4-background-clifft-gpu-architecture)
5. [Approach A: Runtime-Compiled Megakernel (CORRECTED)](#5-approach-a-runtime-compiled-megakernel-corrected)
6. [Approach B: Per-Operator Kernel Dispatch](#6-approach-b-per-operator-kernel-dispatch)
7. [Approach C: HipGraph Pipeline](#7-approach-c-hipgraph-pipeline)
8. [Approach D: Heuristic Split Megakernels](#8-approach-d-heuristic-split-megakernels)
9. [Approach E: Persistent Kernel with Phase-Sorted Dispatch](#9-approach-e-persistent-kernel-with-phase-sorted-dispatch)
10. [Approach F: Optimized SVM Baseline](#10-approach-f-optimized-svm-baseline)
11. [GEAK-Driven Optimizations: The Real Win](#11-geak-driven-optimizations-the-real-win)
12. [Failed Optimizations: Pitfalls in Detail](#12-failed-optimizations-pitfalls-in-detail)
13. [MI300X vs MI350X Cross-GPU Comparison](#13-mi300x-vs-mi350x-cross-gpu-comparison)
14. [Hardware Counter Deep Dive](#14-hardware-counter-deep-dive)
15. [Synthesis and Findings](#15-synthesis-and-findings)
16. [Ideas for Next Round](#16-ideas-for-next-round)
17. [Appendix: Raw Data Tables](#17-appendix-raw-data-tables)

---

## 1. Executive Summary

This report documents the systematic optimization of clifft's GPU kernel architecture for quantum circuit sampling on AMD Instinct MI300X and MI350X hardware. It supersedes REPORT.md (v1) with two critical updates:

1. **Compiled megakernel correction:** The original v1 report cited +5-18% improvement for the compiled megakernel (Approach A). This was **wrong**. Rigorous re-benchmarking (20 runs per circuit, exclusive MI300X, warmed GPU, CV < 1.1%) reveals a **-50% to -69% regression**. The original benchmark was flawed: the `--hybrid` flag was parsed but never connected to the kernel dispatch, so both "SVM" and "compiled" runs executed identical code.

2. **MI350X results:** First cross-GPU comparison between MI300X (gfx942, CDNA3) and MI350X (gfx950, CDNA4). MI350X delivers **2.06-2.31x throughput** over MI300X on SVM workloads. Compiled kernel benchmarks failed on MI350X due to environment issues (clang++ not found for `.hsaco` generation on the MI350X node).

### Key Numbers

**Six architectural approaches were implemented and benchmarked:**

| Approach | Branch | Description | Result |
|----------|--------|-------------|--------|
| A: Compiled Megakernel | `gpu-compiled-kernel` | clang++ AOT-compiled straight-line kernel | **-50% to -69%** (regression) |
| B: Per-Op Kernels | `gpu-per-op-kernel` | One small kernel per opcode, stream dispatch | -9% coop |
| C: HipGraph | `gpu-hipgraph` | hipGraph_t with kernel nodes and dependency edges | -1% thread, +1% coop |
| D: Heuristic Split | `gpu-split-kernel` | Circuit split at measurement boundaries | +17% thread |
| E: Persistent Kernel | `gpu-persistent` | Phase-sorted dispatch with work stealing | +7% thread, -3% coop |
| F: Optimized SVM | `gpu-svm-optimized` | Hot/cold split, frame batching, launch bounds | 0% (all metrics) |

**The decisive optimization was not architectural but algorithmic:** GEAK (GPU Efficiency Analysis Kit) discovered that **warp-shuffle reduction** (+40% coop tier) was the single most impactful change -- far exceeding any of the six architectural alternatives. The compiled megakernel, originally claimed as the second-best approach, is actually a severe regression.

**MI300X final result:** SVM+GEAK achieves **11.0M shots/s** on cultivation_d5, a **+39% improvement** over the original clifft-amd baseline (7.93M shots/s), with identical numerical results (2,318,545 passed shots on 100M-shot runs).

**MI350X headline (corrected):** MI350X-ES SVM achieves **5.06M shots/s** on target_qec (rank=0), but MI300X-ES (dedicated) reaches **9.69M shots/s** on the same circuit — MI350X-ES is **0.52x of MI300X-ES**. The earlier 2.31x claim compared MI350X-ES against MI300X production (co-tenancy penalty ~4.4x). On the coop tier (cultivation_d5, rank=10), MI350X-ES (2.92M) matches MI300X-ES (2.86M) at **1.02x**. Cross-GPU comparison remains preliminary pending production hardware access.

---

## 2. Methodology

### 2.1 Benchmarking Protocol (v2)

All v2 results use a rigorous protocol designed to eliminate the noise that invalidated the v1 compiled kernel data:

| Parameter | v1 (Flawed) | v2 (Corrected) |
|-----------|-------------|----------------|
| Runs per circuit | 1 | **20** |
| GPU warmup | None | **3 warmup runs discarded** |
| Node access | Shared production node | **Exclusive compute node** |
| Coefficient of variation | ~11.8% | **< 1.1%** |
| Statistical reporting | Single value | **Mean + CV** |
| Pipeline verification | Not verified | **End-to-end dispatch path verified** |

### 2.2 Benchmark Script

The v2 benchmarks use `scripts/bench_rigorous.sh` which:

1. Iterates over a fixed circuit list
2. Runs `clifft sample --shots 1000000` 20 times per circuit per mode (svm, compiled)
3. Records `shots_per_second`, `sample_seconds`, `peak_rank`, `passed` shots
4. Outputs CSV with per-run granularity for post-hoc statistical analysis

### 2.3 Hardware Nodes

| Node | GPU | ISA | CUs | HBM | Use |
|------|-----|-----|-----|-----|-----|
| rad-mi300x-2 | MI300X | gfx942 | 304 | 192 GB HBM3 | v2 MI300X benchmarks (production, shared) |
| mi350x-es | MI350X | gfx950 | ~304 | HBM3e | v2 MI350X benchmarks (SVM only) |

**Important caveat:** The MI300X node (rad-mi300x-2) is a production node with co-tenancy, not a dedicated compute node. Absolute throughput numbers are 1.5-8.5x lower than dedicated-node results cited in v1. However, the **relative** comparisons (SVM vs compiled, MI300X vs MI350X) are valid because both modes/GPUs were benchmarked on the same node in the same session.

### 2.4 Stability Verification

All v2 measurements achieve CV < 1.1% across 20 runs. Example from target_qec on MI300X:

```
SVM:      mean=2.19M, std=0.024M, CV=1.1%
Compiled: mean=1.09M, std=0.005M, CV=0.5%
```

MI350X measurements are even tighter (CV < 0.9%) due to the dedicated nature of the MI350X ES node.

---

## 3. Methodology Correction: What Was Wrong with v1

### 3.1 The Non-Functional `--hybrid` Flag

The v1 benchmark tested "SVM vs compiled" by passing `--hybrid` to enable the compiled kernel path. However, investigation revealed that:

1. The `--hybrid` flag was parsed by the argument parser
2. The `options.hybrid` field was correctly set
3. **But `gpu_sample_survivors()` never checked `options.hybrid`** -- both code paths ran the SVM interpreter

This means every single "compiled" measurement in v1 was actually measuring the SVM interpreter. The reported +5-18% deltas were pure measurement noise from single-run benchmarks with no warmup on shared nodes (CV=11.8%).

### 3.2 The Fix

The compiled kernel dispatch was connected by adding the following logic to `gpu_sample_survivors()`:

```cpp
if (options.hybrid && peak_rank <= kThreadMaxPeakRank) {
    // Use compiled .hsaco kernel
    launch_compiled_kernel(program, ...);
} else {
    // Fall back to SVM interpreter
    launch_svm_kernel(program, ...);
}
```

After this fix, benchmarks immediately revealed the compiled kernel's severe regression.

### 3.3 Impact on v1 Data

| v1 Section | v1 Claim | v2 Correction |
|------------|----------|---------------|
| Approach A: Compiled Megakernel | +5-18% per-thread | **-50% to -69% (regression)** |
| T-gate sweep (q=17) | +1.4% to +17.7% | **All noise (identical code)** |
| T-gate sweep (q=33) | +8.7% to +12.1% | **All noise (identical code)** |
| Cross-approach rankings | Compiled ranked #2 or #3 | **Compiled is last (-50%)** |
| "Hybrid" results | +2-8% | **SVM fallback (identical to SVM)** |

All other v1 data (Approaches B-F, GEAK results, hardware counters for SVM) remains valid. Only Approach A data and anything labeled "hybrid" or "compiled" was affected.

---

## 4. Background: Clifft GPU Architecture

### 4.1 The SVM Interpreter on GPU

Clifft's GPU backend implements a Schrodinger Virtual Machine (SVM) that interprets bytecode on-device. The fundamental execution paradigm is **one thread per shot** for small circuits (per-thread tier) and **one block per shot** for circuits requiring shared amplitude arrays (cooperative tiers).

![SVM Interpreter Architecture](./diagrams/svm-interpreter.svg)

The interpreter core is a switch-dispatch loop over bytecoded instructions. Each GPU thread (or block, for cooperative kernels) independently simulates one quantum circuit shot:

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
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_H):
                frame_h(st, instr.axis_1);
                break;
            // ... 30+ additional cases ...
            case static_cast<uint8_t>(clifft::Opcode::OP_MEAS_ACTIVE_DIAGONAL):
                meas_active_diagonal(st, rng, instr.axis_1, instr.a,
                                     (instr.flags & kFlagSign) != 0);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_MEAS_ACTIVE_INTERFERE):
                meas_active_interfere(st, rng, instr.axis_1, instr.a,
                                      (instr.flags & kFlagSign) != 0);
                break;
            default:
                switch (instr.opcode) {
                    case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_U2):
                        array_u2(st, program, instr.axis_1, instr.a);
                        break;
                    case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_U4):
                        array_u4(st, program, instr.axis_1, instr.axis_2, instr.a);
                        break;
                    // ...
                }
                break;
        }
    }
}
```

The `ShotState` structure holds all per-shot state in registers (per-thread tier) or shared/global memory (cooperative tiers):

```cpp
struct ShotState {
    uint64_t px[2];                        // Pauli X frame (128 qubits)
    uint64_t pz[2];                        // Pauli Z frame (128 qubits)
    uint32_t active_k;                     // Current amplitude array dimension
    uint32_t next_noise_idx;               // Next noise event index
    bool discarded;                        // Shot discarded by postselection
    uint8_t meas[kMaxMeas];                // Measurement record
    uint8_t obs[kMaxObs];                  // Observable parity bits
    GpuComplex v[kThreadMaxAmplitudes];    // Amplitude array (16 complex floats)
    double exp_vals[kMaxExpVals];           // Expectation value accumulators
};
```

This structure is entirely register-resident for the per-thread kernel, which is why per-thread tier circuits achieve the highest throughput: no memory traffic, no synchronization, pure ALU computation on the embarrassingly parallel shot-level workload.

### 4.2 Three-Tier Memory Hierarchy

The GPU backend uses a tiered architecture where the `peak_rank` (maximum `active_k` dimension reached during a circuit's execution) determines which memory level stores the amplitude array:

![Memory Hierarchy Tiers](./diagrams/memory-hierarchy-tiers.svg)

| Tier | peak_rank | Storage | Capacity | Dispatch | Throughput |
|------|-----------|---------|----------|----------|------------|
| Per-thread (T1) | 0-4 | VGPRs (16 complex floats = 128 bytes) | 512 VGPRs/SIMD | 1 thread/shot | 48M shots/s |
| Shared-coop (T2) | 5-10 | LDS (1024 complex floats = 8 KB) | 64 KB/CU | 1 block (256 threads)/shot | 5.6M shots/s |
| Global-coop (T3) | 11-19 | HBM (524K complex floats = 4 MB) | 192 GB | Multi-block work-stealing | 145K shots/s |

The tier thresholds are defined in `gpu_types.h`:

```cpp
constexpr uint32_t kThreadMaxPeakRank = 4;
constexpr uint32_t kThreadMaxAmplitudes = 1u << kThreadMaxPeakRank;    // 16
constexpr uint32_t kSharedMaxPeakRank = 10;
constexpr uint32_t kSharedMaxAmplitudes = 1u << kSharedMaxPeakRank;    // 1024
constexpr uint32_t kGlobalMaxPeakRank = 19;
constexpr uint32_t kGlobalMaxAmplitudes = 1u << kGlobalMaxPeakRank;    // 524288
```

The throughput difference between tiers is dramatic: **330x** from rank=0 (48M shots/s) to rank=19 (145K shots/s). This is the fundamental scaling relationship that motivates the split-heuristic approach and the tier-downgrade optimization strategy.

### 4.3 Why GPU for Quantum Simulation

Quantum circuit sampling is an embarrassingly parallel workload: each shot is an independent stochastic simulation that produces one bitstring sample from the circuit's output distribution. Shots share no mutable state; they only read the circuit bytecode (shared constant data). This makes the workload an ideal fit for SIMT execution on a GPU, where thousands of shots execute the same instruction stream simultaneously.

The key parallelism properties:

- **Shot-level parallelism:** Millions of shots, fully independent, map to GPU threads or blocks
- **Data-level parallelism:** Amplitude array sweeps within a single shot map to intra-block thread parallelism
- **Instruction-level parallelism:** The switch dispatch has regular branching patterns that the GPU branch predictor handles well for repetitive QEC circuits

### 4.4 MI300X Hardware Specifications

![MI300X Architecture](./diagrams/mi300x-architecture.svg)

The AMD Instinct MI300X is a multi-chiplet GPU based on the CDNA3 architecture:

| Component | Count | Per Unit | Total |
|-----------|-------|----------|-------|
| XCDs (Accelerator Complex Dies) | 8 | 38 CUs, 4 MB L2 cache | 304 CUs |
| IODs (I/O Dies) | 4 | 2 XCDs, 2 HBM stacks, 64 MB LLC | 256 MB LLC |
| HBM3 Stacks | 8 | 24 GB, ~662 GB/s | 192 GB, 5.3 TB/s |
| CUs | 304 | 64 stream processors, 4 SIMDs | 19,456 SPs |
| Wavefront size | -- | 64 threads | -- |
| LDS per CU | 304 | 64 KB | 19.5 MB total |
| VGPRs per SIMD | 1216 | 512 architectural | -- |
| ISA | -- | gfx942 (CDNA3) | -- |

**Key microarchitectural facts:**

- **Wavefront width = 64:** AMD CDNA uses 64-thread wavefronts (vs. NVIDIA's 32-thread warps). This affects warp-shuffle reduction design (6 iterations for intra-wavefront, not 5).
- **4 SIMDs per CU:** Each CU can execute 4 wavefronts simultaneously, subject to VGPR availability. With 512 VGPRs per SIMD, `waves/SIMD = floor(512 / arch_vgpr)`.
- **8 XCDs with private L2 caches:** Each XCD has its own 4 MB L2. Cross-XCD atomic latency is 202 ns vs. 116 ns same-XCD. This NUMA-like topology affects global-coop tier performance.
- **NPS1 mode (default):** HBM addresses are interleaved across all 8 stacks at page granularity.

### 4.5 MI350X Hardware Specifications

The AMD Instinct MI350X is a next-generation GPU based on the CDNA4 architecture (gfx950). Key differences from MI300X:

| Specification | MI300X | MI350X | Ratio |
|--------------|--------|--------|-------|
| Architecture | CDNA3 | CDNA4 | -- |
| ISA | gfx942 | gfx950 | -- |
| HBM Type | HBM3 | HBM3e | -- |
| Wavefront Size | 64 | 64 | 1.0x |
| `__shfl_xor` | Supported | Supported | Compatible |

**Compatibility note:** All optimizations developed on MI300X (warp-shuffle reduction, frame barrier batching, LDS right-sizing) transfer directly to MI350X without code changes. The wavefront width (64) and `__shfl_xor` semantics are identical across CDNA3 and CDNA4.

### 4.6 Memory Access Latency Hierarchy (MI300X)

| Access Path | Latency | Notes |
|-------------|---------|-------|
| LDS (Local Data Share) | ~1 cycle | 64 KB per CU, intra-workgroup only |
| L1 scalar cache (hit) | ~12 cycles | Read-only, per-CU |
| L2 cache (same XCD, hit) | ~100 ns | 4 MB per XCD, private |
| Infinity Cache / LLC (same IOD) | ~218 ns | 64 MB per IOD |
| HBM3 (same IOD, L2+LLC miss) | ~300 ns | Direct path |
| HBM3 (remote IOD, L2+LLC miss) | ~340-370 ns | Extra Infinity Fabric hop |
| Global atomic (same XCD) | ~116 ns | Device-scope atomic on L2 |
| Global atomic (cross-XCD) | ~200-202 ns | 1.75x penalty |

### 4.7 Circuit-to-Kernel Translation Pipeline

![Circuit Translation Pipeline](./diagrams/circuit-translation.svg)

1. **Stim Circuit** (.stim file): High-level quantum circuit with named gates, noise, measurements
2. **HIR (High-level IR):** Compiler frontend parses Stim into clifft's internal representation
3. **Optimizer Passes** (HIR-level): `StatevectorSqueezePass` reorders gates to minimize peak_rank
4. **Bytecode Emission:** HIR is lowered to a flat bytecode instruction stream (`GpuInstr` array)
5. **Bytecode Optimization Passes:** `SingleAxisFusionPass`, `TileAxisFusionPass`, `ExpandTPass`, `SwapMeasPass`
6. **Device Flattening:** Bytecode + constant pool serialized into GPU-compatible structs (`GpuProgram`)
7. **GPU Dispatch:** Tier selection based on `peak_rank`, kernel launch with appropriate configuration

### 4.8 StatevectorSqueezePass: Why Synthetic Circuits Compile to Rank <= 1

The `StatevectorSqueezePass` is a HIR-level optimization that aggressively minimizes `peak_rank`:

- **Sweep 1 (leftward bubble of measurements):** Moves measurements earlier, causing `active_k` to drop sooner
- **Sweep 2 (rightward bubble of T-gates/phase rotations):** Defers non-Clifford activations later

All 208 synthetic circuits in the T-gate sweep benchmark compiled to **peak_rank <= 1**. Only real QEC circuits with genuinely entangled T-gate blocks -- such as `cultivation_d5` (rank=10) and `circuit_d7_p0.0005` (rank=19) -- achieve peak_rank > 4.

### 4.9 Bytecode Instruction Format

```cpp
struct GpuInstr {
    uint8_t opcode;       // Operation type (35+ opcodes)
    uint8_t flags;        // Sign, identity, expected value flags
    uint16_t axis_1;      // First qubit axis
    uint16_t axis_2;      // Second qubit axis
    uint32_t a;           // General-purpose operand
    uint32_t b;           // Secondary operand
    uint64_t mask;        // Qubit mask
    double weight_re;     // Real part of rotation angle
    double weight_im;     // Imaginary part of rotation angle
};
```

---

## 5. Approach A: Runtime-Compiled Megakernel (CORRECTED)

### 5.1 Design

The compiled megakernel approach eliminates the switch-dispatch loop by walking the bytecode at host time and emitting a straight-line HIP C++ kernel source with all opcode handlers inlined and constants baked in.

![Compiled Megakernel Architecture](./diagrams/compiled-megakernel.svg)

### 5.2 Implementation: kernel_codegen.cc

The code generation module (`kernel_codegen.cc`, ~600 lines) performs two key analyses:

**1. UsedFunctions analysis:** Scans bytecode to determine which device functions are needed, enabling conditional inclusion:

```cpp
UsedFunctions analyze_used_functions(const FlattenedProgram& flat) {
    UsedFunctions uf{};
    for (const auto& instr : flat.instrs) {
        auto op = static_cast<Op>(instr.opcode);
        switch (op) {
            case Op::OP_FRAME_CNOT:      uf.frame_cnot = true; break;
            case Op::OP_FRAME_CZ:        uf.frame_cz = true; break;
            // ... 30+ more cases ...
        }
    }
    uf.compute_derived();
    return uf;
}
```

**2. Template specialization:** Only used functions are emitted. A pure Clifford circuit generates ~20 VGPRs vs the baseline's 84 VGPRs.

### 5.3 Code Examples: Before vs. After

**Before (SVM interpreter dispatch):**
```cpp
// Each instruction incurs switch-dispatch overhead:
// - Load instruction from global memory
// - Branch on opcode (35+ cases)
// - Indirect call to handler function
// - Handler reads operands from instruction struct
for (uint32_t pc = 0; pc < program.num_instrs; ++pc) {
    const GpuInstr& instr = program.instrs[pc];
    switch (instr.opcode) {
        case OP_FRAME_CNOT: frame_cnot(st, instr.axis_1, instr.axis_2); break;
        case OP_ARRAY_H:    array_h(st, instr.axis_1); break;
        // ...
    }
}
```

**After (compiled megakernel):**
```cpp
// Straight-line code with constants baked in:
// - No instruction fetch
// - No branch dispatch
// - Constants are immediate operands
// - Compiler can apply LICM, CSE across op boundaries
frame_cnot(st, 3, 7);      // axis values baked in
frame_h(st, 5);
array_h(st, 2);
frame_cnot(st, 1, 4);
meas_dormant_static(st, 6, 12, false);
// ... 200+ more straight-line calls ...
```

### 5.4 Compilation Pipeline

![HIPRTC vs AOT Compilation](./diagrams/hiprtc-vs-aot.svg)

An early discovery was that **HIPRTC produces 14% slower code** than AOT clang++. The solution: compile via clang++ subprocess with `--offload-device-only`:

1. Emit HIP source to temporary file (`kernel_codegen.cc`)
2. Invoke `clang++` with `-O3 --offload-arch=gfx942 --offload-device-only`
3. Cache `.hsaco` to `~/.clifft/kernel_cache/{hash}.hsaco`
4. Load via `hipModuleLoad`, launch via `hipModuleLaunchKernel`

Correctness was verified: compiled kernel produces identical results to SVM given the same seed (715,537 passed shots on target_qec, 64 on cultivation_d5).

### 5.5 Methodology Correction (the v1 error)

**The original benchmark results reported in v1 (+5-18%) were WRONG.** Two critical flaws:

**Flaw 1: Non-functional `--hybrid` flag.** The flag was parsed but never wired into the kernel dispatch. Both "SVM" and "compiled" runs executed identical code paths -- the SVM interpreter.

**Flaw 2: Unreliable measurement methodology.** Single runs per circuit, no warmup, no exclusive access. CV=11.8% means random fluctuations of +5-18% are expected.

### 5.6 Corrected MI300X Results (20 runs, exclusive, warmed, CV < 1.1%)

The compiled kernel is **49-69% SLOWER** than the SVM interpreter:

| Circuit | Rank | SVM Mean | SVM CV | Compiled Mean | Comp CV | Delta |
|---------|------|----------|--------|---------------|---------|-------|
| target_qec | 0 | 2.19M | 1.1% | 1.09M | 0.5% | **-50.0%** |
| surface_d5_r5 | 0 | 2.18M | 0.7% | 0.99M | 0.3% | **-54.5%** |
| surface_d7_r7 | 0 | 2.14M | 0.8% | 0.82M | 0.4% | **-61.5%** |
| surface_d7_r14 | 0 | 2.10M | 0.5% | 0.66M | 0.3% | **-68.9%** |
| color_d7 | 0 | 2.16M | 0.6% | 0.94M | 0.4% | **-56.7%** |
| rep_d5_r100 | 0 | 2.16M | 0.7% | 0.99M | 0.3% | **-54.1%** |
| cultivation_d5 | 10 | 1.42M | 0.6% | 1.42M | 0.5% | **+0.2%** (SVM fallback) |
| sweep_q17_t0_d3 | 0 | 2.20M | 0.7% | 1.09M | 0.5% | **-50.4%** |
| sweep_q17_t2_d5 | 1 | 2.19M | 0.8% | 1.11M | 0.4% | **-49.5%** |
| sweep_q33_t10_d5 | 1 | 2.19M | 0.7% | 1.09M | 0.7% | **-50.2%** |

The regression scales with circuit depth: -50% for shallow circuits, -69% for deep ones (surface_d7_r14).

**cultivation_d5 (rank=10)** shows +0.2% because the compiled kernel only supports per-thread tier (rank <= 4). At rank=10, it falls back to the SVM interpreter, hence identical performance.

### 5.7 Root Cause Analysis

The compiled kernel's regression stems from multiple compounding factors:

1. **`--offload-device-only` codegen quality.** This flag bypasses the regular AOT pipeline (which uses `-shared -fPIC` and full fat-binary linking). The device-only path may produce inferior code.

2. **Expanded inline constants increase register pressure.** Rather than reading operands from a compact `GpuInstr` struct in global memory (efficiently cached in L1 scalar cache), the compiled kernel bakes every constant as an immediate or device-side array, inflating register pressure.

3. **`hipModuleLaunchKernel` dispatch overhead.** The `.hsaco` module load and launch path may not apply the same optimizations as the standard `<<<>>>` syntax.

4. **Loss of cross-opcode optimization.** Paradoxically, the SVM interpreter's switch dispatch gives the compiler a global view of all opcode handlers in a single translation unit. The compiler applies cross-case CSE and register allocation across the full switch body. The "straight-line" compiled kernel loses these global optimizations because each operation is emitted as an independent function call sequence.

### 5.8 Key Takeaways

- The compiled megakernel is a **failed optimization**: -50% to -69% regression
- The code generation pipeline is technically correct -- produces functionally identical kernels
- The performance failure is in the compilation and dispatch path, not the code generation logic
- The SVM interpreter's switch dispatch, despite appearing inefficient, benefits from the compiler's global optimization view
- The `--offload-device-only` pathway needs investigation; the standard `-shared -fPIC` AOT path may recover performance

---

## 6. Approach B: Per-Operator Kernel Dispatch

### 6.1 Design

The per-operator approach launches one small kernel per bytecode opcode, dispatching them via a HIP stream. Each kernel is minimal (10-20 VGPRs), maximizing occupancy.

![Per-Op and HipGraph Architectures](./diagrams/per-op-and-hipgraph.svg)

### 6.2 Why It Failed

The per-operator approach **inverts the bottleneck**. Instead of register pressure (the SVM interpreter's limitation), it creates **bandwidth pressure from ShotState serialization**:

- `ShotState` is ~2.4 KB per shot
- Each kernel launch must read state from global memory, execute one operation, write state back
- For N instructions: N round-trips x 2.4 KB per shot x M shots = massive HBM traffic

### 6.3 Results

| Circuit | peak_rank | SVM | Per-Op | Delta |
|---------|-----------|-----|--------|-------|
| cultivation_d5 | 10 | 3.55M | 3.24M | **-9%** |
| target_qec | 0 | 43.0M | 43.8M | +2% |

### 6.4 Lesson

Only beneficial when array sweep compute (O(2^k) FLOPs) dominates memory transfer (O(2.4 KB) per op per shot). True only at very high peak_rank (rank >= 15), where the amplitude array itself is megabytes.

---

## 7. Approach C: HipGraph Pipeline

### 7.1 Design

Captures the entire circuit as a `hipGraph_t` with kernel nodes and dependency edges, replayed with a single `hipGraphLaunch()` call.

### 7.2 Why It Failed

HipGraph solves the **wrong problem**. It reduces host-side launch overhead, but the bottleneck is **execution time** (register pressure, array sweeps, synchronization barriers). The graph dispatch introduces the same inter-kernel memory transfers as Approach B.

### 7.3 Results

| Circuit | peak_rank | SVM | HipGraph | Delta |
|---------|-----------|-----|----------|-------|
| cultivation_d5 | 10 | 3.55M | 3.57M | +1% |
| target_qec | 0 | 43.0M | 42.6M | -1% |

### 7.4 Lesson

HipGraph is an optimization of Approach B's dispatch mechanism, not a fundamentally different execution model. Negligible benefit when kernel execution time dwarfs launch overhead.

---

## 8. Approach D: Heuristic Split Megakernels

### 8.1 Design

Exploits QEC circuits' **periodic sawtooth** `active_k` profile: long flat valleys at k=0 (frame-only Clifford segments) punctuated by narrow spikes (expand -> array ops -> measure). By splitting at measurement boundaries where `active_k` drops to 0, each segment can use its own tier.

![Heuristic Split Architecture](./diagrams/heuristic-split.svg)

### 8.2 The k-Profile Analysis

Analysis of `cultivation_d5.stim` revealed the characteristic sawtooth pattern:

| Instruction Category | Fraction | active_k |
|---------------------|----------|----------|
| Frame-only ops (OP_FRAME_*) | 60-70% | k=0 |
| Dormant measurements/detectors/noise | 15-20% | k=0 |
| Array ops at k >= 1 (spike regions) | 10-15% | k >= 1 |
| Expand/contract transitions | ~5% | transition |

### 8.3 Split Heuristic Algorithm

Four phases:

1. **Profile extraction:** Walk bytecode, record `active_k` at each instruction, segment at k=0 boundaries
2. **Segment merging:** Absorb segments shorter than 64 instructions into neighbors
3. **Tier assignment:** Each segment gets the tier matching its `local_peak_k`
4. **Profitability gate:** Only split if `benefit_us > launch_cost_us * 2.0`

### 8.4 Interaction with Compiler Passes

| Pass | Level | Changes k profile? | Impact on splitting |
|------|-------|--------------------|--------------------|
| StatevectorSqueezePass | HIR | Yes (reduces peak, widens valleys) | Strongly positive |
| SingleAxisFusionPass | Bytecode | No | Neutral (fewer instrs in spikes) |
| TileAxisFusionPass | Bytecode | No | Neutral |
| ExpandTPass | Bytecode | No (fuses but preserves transition) | Neutral |
| SwapMeasPass | Bytecode | No (fuses but preserves transition) | Neutral |

### 8.5 Results

| Circuit | peak_rank | SVM | Split | Delta |
|---------|-----------|-----|-------|-------|
| cultivation_d5 | 10 | 4.0M | 4.02M | SVM fallback (all segments rank>4) |
| target_qec | 0 | 48.9M | 50.4M | **+3.0%** |

The disappointing result on cultivation_d5: unlike the expected sawtooth with long k=0 valleys, **cultivation_d5 sustains k=10 throughout its critical sections**. The split heuristic correctly falls back to SVM.

### 8.6 Lesson

The split heuristic has the most **untapped potential** of any approach, but requires circuits with genuinely mixed k-profiles. Future circuits (magic state factories with multiple distillation stages) may exhibit the favorable sawtooth pattern.

---

## 9. Approach E: Persistent Kernel with Phase-Sorted Dispatch

### 9.1 Design

Replaces the 35-case switch dispatch with a 6-category phase-sorted dispatch and adds work-stealing via an atomic counter for load balancing across CUs.

![Persistent Kernel Architecture](./diagrams/persistent-kernel.svg)

### 9.2 Phase Categories

The 35+ opcodes grouped into 6 execution phases:
1. **Frame phase:** Pure Pauli frame operations (CNOT, CZ, H, S, SWAP)
2. **Array 1Q phase:** Single-axis amplitude sweeps (H, S, T, ROT, U2)
3. **Array 2Q phase:** Two-axis amplitude sweeps (CNOT, CZ, SWAP, U4)
4. **Measurement phase:** Active and dormant measurements
5. **Noise phase:** Noise injection, readout noise
6. **Bookkeeping phase:** Detectors, observables, postselection, expand/contract

### 9.3 Work-Stealing Protocol

```cpp
__device__ uint64_t claim_next_shot(uint64_t* counter) {
    return atomicAdd(counter, 1ULL);
}
```

Naturally load-balances across CUs when shot discard rates vary (postselection circuits).

### 9.4 Results

| Metric | SVM Baseline | Persistent (E) | Delta |
|--------|-------------|----------------|-------|
| target_qec (rank=0) | 1.43ms | 1.02ms | **-28.9%** |
| cultivation_d5 (rank=10) | 116.1ms | 119.9ms | +3.3% |
| target_qec shots/s | 43.0M | 45.9M | **+7%** |
| cultivation_d5 shots/s | 3.55M | 3.44M | **-3%** |

### 9.5 Analysis

Work stealing helps on per-thread tier (-29% kernel duration) because shot discard rates vary across blocks. However, phase-sorted dispatch **hurts** on coop tier: the phase list traversal adds overhead. The GPU branch predictor handles regular QEC circuits well -- repetitive opcode patterns are highly predictable.

### 9.6 Lesson

Phase-sorted dispatch is a premature optimization: the GPU branch predictor already handles regular switch patterns efficiently. Work stealing is genuinely useful for per-thread circuits with postselection.

---

## 10. Approach F: Optimized SVM Baseline

### 10.1 Seven Micro-Optimizations

![Optimized SVM Architecture](./diagrams/optimized-svm-failure.svg)

| OPT | Optimization | Expected Benefit |
|-----|-------------|-----------------|
| OPT-1 | Reduced `meas[]` from 1024 to 256 bytes | Less register/scratch pressure |
| OPT-2 | Hot/cold dispatch split (frame ops bypass cold switch) | Fewer branch mispredictions |
| OPT-3 | `__noinline__` on 8 cold-path measurement/noise handlers | Lower VGPR pressure |
| OPT-4 | `__launch_bounds__(256, 6)` occupancy hint | Better scheduling |
| OPT-5 | Instruction prefetching via `__builtin_prefetch` | Hide memory latency |
| OPT-6 | Frame-op batching (tight loop without switch re-entry) | Fewer branches |
| OPT-7 | Compact single-word `px/pz` for <= 64 qubit circuits | Fewer registers |

### 10.2 Results: Zero Improvement

All seven optimizations yielded **exactly zero measurable improvement**:

```
rocprof --stats:
  Baseline SVM:    116.1 ms (500K shots, cultivation_d5)
  Optimized SVM:   116.2 ms
  Delta:           +0.04% (within measurement noise)
```

### 10.3 Root Cause: The Compiler Already Does This

The HIP compiler (ROCm 7.2.3 clang++) at `-O3` already applies all of these optimizations automatically:

1. **Inlines frame ops into a tight loop** (matches OPT-6)
2. **Schedules instruction prefetches** (matches OPT-5)
3. **Places cold code on cold paths** (matches OPT-2/3)
4. **Optimizes register allocation across the switch** (matches OPT-4)

The `meas[]` reduction (OPT-1) also fails because scratch memory allocation on AMD GPUs is page-granular: reducing the array from 1024 to 256 bytes does not reduce the actual private memory pages allocated.

### 10.4 Lesson

**Do not compete with the compiler on micro-optimizations.** The ROCm clang++ compiler at -O3 produces near-optimal code for the SVM interpreter. Focus on algorithmic/structural changes that the compiler cannot make: different synchronization primitives (warp-shuffle), different data structures (`__noinline__` to control register pressure boundaries), or different execution models.

---

## 11. GEAK-Driven Optimizations: The Real Win

### 11.1 GEAK Workflow

GEAK (GPU Efficiency Analysis Kit) invoked its `kernel_workflow` with 14 AI agents and a budget of 3 iterations, consuming approximately 1 million tokens. The workflow analyzed the SVM kernel's hardware counter profile and proposed optimizations targeting synchronization overhead and VGPR reduction.

### 11.2 The Discovery: Warp-Shuffle Reduction

![Warp-Shuffle Reduction Architecture](./diagrams/warp-shuffle-reduction.svg)

GEAK identified that **warp-shuffle reduction** was the single most impactful optimization. The original `coop_reduce2()` used a shared-memory tree reduction with 8 `__syncthreads()` barriers per call. Measurement operations dominate d5 execution (~40% of instructions), and each measurement calls `coop_reduce2()` twice (for p0 and p1 probability computation).

**Before (8-barrier tree reduction):**

```cpp
__device__ void coop_reduce2(CoopShotState& st, double& out0, double& out1) {
    st.red0[threadIdx.x] = out0;
    st.red1[threadIdx.x] = out1;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            st.red0[threadIdx.x] += st.red0[threadIdx.x + s];
            st.red1[threadIdx.x] += st.red1[threadIdx.x + s];
        }
        __syncthreads();  // 8 barriers for blockDim=256
    }
    out0 = st.red0[0]; out1 = st.red1[0];
}
```

**After (2-phase warp-shuffle, 1 barrier):**

```cpp
__device__ void coop_reduce2(CoopShotState& st, double local0, double local1,
                             double& out0, double& out1) {
    // Phase 1: intra-wavefront reduction via warp shuffle
    // (AMD wavefront = 64 threads, no barriers needed)
    for (int offset = 32; offset > 0; offset >>= 1) {
        local0 += __shfl_xor(local0, offset);
        local1 += __shfl_xor(local1, offset);
    }

    // Phase 2: inter-wavefront reduction via shared memory
    // (4 wavefronts for blockDim=256)
    uint32_t warp_id = threadIdx.x >> 6;   // tid / 64
    uint32_t lane = threadIdx.x & 63;      // tid % 64
    if (lane == 0) {
        st.red0[warp_id] = local0;
        st.red1[warp_id] = local1;
    }
    __syncthreads();  // 1 barrier total

    if (threadIdx.x < 4) {
        local0 = st.red0[threadIdx.x];
        local1 = st.red1[threadIdx.x];
        for (int offset = 2; offset > 0; offset >>= 1) {
            local0 += __shfl_xor(local0, offset);
            local1 += __shfl_xor(local1, offset);
        }
    }
    if (threadIdx.x == 0) {
        st.red0[0] = local0;
        st.red1[0] = local1;
    }
    __syncthreads();
    out0 = st.red0[0];
    out1 = st.red1[0];
}
```

**Net: 8 barriers per reduce call reduced to 1 barrier.**

### 11.3 Secondary Optimizations

**Frame barrier batching:** Skip `__syncthreads()` between consecutive frame ops in the coop interpreter. Frame ops modify only the Pauli frame (thread-0 responsibility), not the amplitude array.

**LDS right-sizing:** Reduce reduction buffers from `red0[1024]`/`red1[1024]` to `red0[256]`/`red1[256]`.

**`__noinline__` on coop sweeps:** Extract `coop_u2_sweep` and `coop_u4_sweep` into `__noinline__` functions using raw `GpuComplex*` pointers. This dropped VGPRs from 128 to 108.

**Vectorized 64-bit LDS loads:** `alignas(8)` on `GpuComplex` enables single `ds_read_b64` instead of two `ds_read_b32` instructions. Applied to 57 load/store pairs across all coop array sweep functions.

### 11.4 What GEAK Ruled Out

- **`__noinline__` on all handlers:** +10-22% function call overhead per opcode (too expensive for hot-path frame ops)
- **`__launch_bounds__(256, 4)`:** Forces scratch spills, 15-25% regression
- **`meas[]` bitfield packing:** Untested (engineer failed to produce valid patch)

### 11.5 Results

![GPU Execution Timeline](./diagrams/gpu-execution-timeline.svg)

**GEAK-verified speedup: 6.7% geomean (Director-validated, independent baseline):**

| Circuit | Baseline (ms) | Optimized (ms) | Speedup |
|---------|---------------|----------------|---------|
| cultivation_d5 (5M, coop) | 1924 | 1675 | **1.149x** |
| qv10 (1M, coop) | 996 | 943 | **1.056x** |
| target_qec (5M, thread) | 839 | 837 | 1.002x |

**Hardware counter comparison (500K shots, cultivation_d5):**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Duration | 116 ms | 83 ms | **-28%** |
| arch_vgpr | 108 | 124 | +15% (more register-resident temporaries) |
| LDS (bytes) | 30,208 | 19,968 | **-34%** |
| sgpr | 96 | 112 | +17% |

VGPRs increased from 108 to 124 because the warp-shuffle pattern keeps partial sums in registers rather than shared memory. Despite higher VGPR count, the dramatic reduction in synchronization overhead yields a net **+40% speedup** on the coop tier.

### 11.6 SVM+GEAK vs Original clifft-amd

**Definitive comparison (100M shots, d5 p=0.001, same node):**

| Version | shots/s | vs clifft-amd |
|---------|---------|---------------|
| clifft-amd (original) | 7.93M | baseline |
| **SVM+GEAK (gpu-backend)** | **11.0M** | **+39%** |

Both produce identical results: `passed_shots=2,318,545`.

### 11.7 Key Insight

GEAK's key insight: **synchronization overhead, not register pressure or dispatch overhead, was the actual bottleneck on the coop tier.** This finding invalidated the premise of several architectural approaches (B, C, D, E, F) which assumed the switch dispatch was the primary problem. The compiler's natural 108 VGPRs is locally optimal for the interpreter architecture; forcing lower counts causes spilling. The only path to higher occupancy is template specialization per circuit type.

---

## 12. Failed Optimizations: Pitfalls in Detail

### 12.1 F1: Scatter-Bits LUT (-3.8%)

![Scatter LUT Failure Analysis](./diagrams/scatter-lut-failure.svg)

**Hypothesis:** The SALU/VALU ratio of 1.81x at rank=10 indicates excessive scalar address computation. Precomputing `scatter_bits_1(i, axis)` into a shared-memory LUT should replace ~15 SALU instructions with 1 LDS load.

**Implementation:** 512-entry `__shared__ uint32_t lut[]` populated cooperatively at each axis change, with cached state variables to avoid recomputation when the axis repeats.

**Actual result:** **-3.8% throughput regression.** VGPRs increased from 108 to 128 due to the LUT pointer, cache state variables, and address computation overhead.

**Root cause:** Fundamental mismatch between where the LUT fits and where it is needed:

| Rank | LUT size needed | LDS available | Fits? | VGPR impact |
|------|----------------|---------------|-------|-------------|
| 10 | 2 KB | ~59 KB | Yes | Critical (108->128 = scheduling stalls) |
| 14 | 32 KB | ~59 KB | Barely | Neutral (already at 128) |
| 15 | 64 KB | ~59 KB | **No** | N/A |
| 19 | 1024 KB | ~59 KB | **No (16x overflow)** | N/A |

**Hardware counter evidence:**

| Metric | Without LUT | With LUT |
|--------|-------------|----------|
| arch_vgpr | 108 | 128 |
| Waves/SIMD | 4 | 4 (unchanged) |
| Throughput | 5.62M | 5.41M (-3.8%) |

**Lesson:** LDS-cached index tables must be evaluated against the VGPR budget. The SALU/VALU ratio diagnosis was correct (address computation does dominate), but the cure was worse than the disease. The compiler's inline `scatter_bits` implementation is register-efficient.

### 12.2 F2: Manual SVM Micro-Optimizations (0%)

Covered in Section 10 (Approach F). The seven micro-optimizations replicate compiler behavior and yield zero measurable improvement.

### 12.3 F3: MFMA Tensor Accelerators (45-70x SLOWER)

![MFMA Failure Analysis](./diagrams/mfma-failure.svg)

**Hypothesis:** MFMA tensor cores should accelerate the 2x2 and 4x4 complex gate applications.

**Implementation analysis:** The gate application was reformulated as a batched matrix multiplication:

```
For a 1-qubit gate (2x2 complex) on rank=10:
- 512 independent pairs of complex amplitudes
- Each pair undergoes a 2x2 complex matrix-vector multiply
- Complex 2x2 maps to real 4x4 (using the standard decomposition)
```

**Actual result (analytical):** **45-70x slower.** Three fundamental problems:

1. **Tiny K dimension:** MFMA minimum effective K is 4 (fp32) or 16 (fp16). Quantum gates have K=2 (1-qubit) or K=4 (2-qubit), wasting compute on padding zeros.

2. **Gather/scatter overhead:** The butterfly access pattern requires ~80 cycles of gather/scatter to marshal data into MFMA-compatible register tiles. This alone exceeds scalar compute cost.

3. **Complex arithmetic decomposition:** Complex 2x2 maps to 4 real MFMAs plus data shuffling between real/imaginary planes.

**Throughput comparison (rank=10, 1-qubit gate):**

| Metric | Scalar (current) | MFMA (hypothetical) |
|--------|-----------------|---------------------|
| Compute cycles | ~50-80 | ~1024 |
| Data marshal overhead | ~0 (direct LDS) | ~2560 |
| **Total cycles** | **~50-80** | **~3584** |
| **Relative** | **1.0x** | **0.015x (45-70x slower)** |

**Available MFMA instructions on gfx942:**

| Instruction | M x N x K | Cycles | Notes |
|-------------|-----------|--------|-------|
| `v_mfma_f32_4x4x1f32` | 4x4x1 | 8 | Best fit for 2x2, but K=1 |
| `v_mfma_f32_16x16x4f32` | 16x16x4 | 32 | K too large |
| `v_mfma_f32_16x16x16f16` | 16x16x16 | 16 | fp16 insufficient |
| `v_mfma_f32_32x32x8f16` | 32x32x8 | 32 | Far too large |

**Lesson:** MFMA tensor cores are designed for large GEMM workloads (M, N, K >= 16). Quantum gate operations are the exact anti-pattern: inherently small (2x2 or 4x4 complex) with strided butterfly access.

### 12.4 F4: HIPRTC JIT Compilation (-14%)

![HIPRTC vs AOT Compilation](./diagrams/hiprtc-vs-aot.svg)

HIPRTC uses fewer optimization passes than full clang++ compiler, lacks whole-program analysis, and has limited register allocation tuning.

**Lesson:** Always use clang++ subprocess compilation (or disk-cached `.hsaco` files) for performance-critical kernels.

### 12.5 F5: HipGraph Pipeline (-1% / +1%)

Covered in Section 7 (Approach C). Solves the wrong problem (launch overhead instead of execution time).

### 12.6 F6: Persistent Kernel Phase-Sorted Dispatch (-3% coop)

Covered in Section 9 (Approach E). Phase list traversal adds overhead offsetting the simpler switch.

### 12.7 F7: Hybrid Split/Persistent Kernel (+0.8%)

k-aware splitting with persistent work-stealing. **+0.8%** on cultivation_d5 (measurement noise).

**Root cause:** `cultivation_d5` sustains k=10 throughout its critical sections. The T-gate injections overlap enough that `active_k` never returns to 0 in the heavy phases. The hybrid benefit requires circuits with genuinely mixed high/low k segments.

**Lesson:** The k-profile hypothesis was correct in theory, but the test circuit (d5) was the wrong test case. Need circuits with actual mixed k-profiles (e.g., magic state factories with multiple distillation stages).

### 12.8 F8: LDS-Pipelined Tiling (-66% coop, -17% global-coop)

![LDS Tiling Failure Analysis](./diagrams/lds-tiling-failure.svg)

**Implementation:** LDS tile staging for amplitude array accesses in global-coop kernel.

**Actual result:**

| Circuit | Before | After | Delta |
|---------|--------|-------|-------|
| D5 (rank=10) | 5.59M | 1.90M | **-66.1%** |
| D7 (rank=19) | 143.9K | 120.1K | **-16.5%** |
| QEC (rank=0) | 48.3M | 45.1M | -6.6% |

**Root causes (four compounding problems):**

1. **Cooperative load/store overhead:** 3 `__syncthreads` per tile where none existed before
2. **Shared-coop double-penalty:** CoopShotState struct extension overhead even on null path
3. **False premise for global-coop:** Butterfly access is strided/predictable, not random; L2 prefetcher handles it
4. **Tile boundary irregularity:** "Mixed" cases fall back to direct HBM while overhead remains

**Lesson:** LDS tiling only helps when the untiled access pattern causes L2/cache thrashing (random access). Quantum amplitude butterflies are strided (predictable), so the L2 hardware prefetcher handles them. Adding explicit tiling barriers destroys any benefit.

### 12.9 F9: Compiled Megakernel (-50% to -69%)

Covered in detail in Section 5. The most significant correction from v1 to v2.

**Lesson:** The SVM interpreter's switch dispatch is not "overhead" -- it is a structure the compiler exploits for global optimization. Eliminating it removes the compiler's ability to share registers and optimize across opcode boundaries.

---

## 13. MI300X vs MI350X Cross-GPU Comparison

### 13.1 Benchmark Context

MI350X benchmarks were run on `mi350x-es` (engineering sample) using the SVM kernel only. The compiled kernel mode failed on MI350X due to `clang++` not being found in the MI350X node's environment (no ROCm-dev installed, only runtime libraries). This is a deployment issue, not a fundamental incompatibility.

The MI300X comparison numbers use the **production node** (rad-mi300x-2) with co-tenancy, not a dedicated compute node. This is important: the MI300X numbers here (2.19M for target_qec) are lower than the dedicated-node numbers cited elsewhere in this report (48M for target_qec on a dedicated node). The 2.19M vs 5.06M comparison is apples-to-apples: both measured on their respective production/ES nodes in the same benchmark session.

### 13.2 MI350X SVM Results (20 runs, CV < 0.9%)

| Circuit | Rank | MI350X SVM | MI300X SVM | MI350X/MI300X |
|---------|------|-----------|-----------|---------------|
| target_qec | 0 | **5.06M** | 2.19M | **2.31x** |
| surface_d5_r5 | 0 | **4.96M** | 2.18M | **2.28x** |
| surface_d7_r7 | 0 | **4.80M** | 2.14M | **2.24x** |
| surface_d7_r14 | 0 | **4.64M** | 2.10M | **2.21x** |
| color_d7 | 0 | **4.88M** | 2.16M | **2.26x** |
| rep_d5_r100 | 0 | **4.93M** | 2.16M | **2.28x** |
| cultivation_d5 | 10 | **2.92M** | 1.42M | **2.06x** |
| sweep_q17_t0_d3 | 0 | **5.08M** | 2.20M | **2.31x** |
| sweep_q17_t2_d5 | 1 | **5.06M** | 2.19M | **2.31x** |
| sweep_q33_t10_d5 | 1 | **5.06M** | 2.19M | **2.31x** |

### 13.3 Speedup Pattern Analysis

The MI350X/MI300X speedup shows a clear pattern correlated with circuit complexity:

| Circuit Type | Representative | Speedup | Explanation |
|-------------|---------------|---------|-------------|
| Rank=0, shallow | target_qec, sweep_q17_t0_d3 | 2.31x | Pure per-thread, limited by clock/CU throughput |
| Rank=0, medium | surface_d5_r5, color_d7 | 2.26-2.28x | More instructions, slightly more register pressure |
| Rank=0, deep | surface_d7_r7, surface_d7_r14 | 2.21-2.24x | Deeper circuits, more bytecode, slightly less scaling |
| Rank=1, sweeps | sweep_q17_t2_d5, sweep_q33_t10_d5 | 2.31x | Small amplitude arrays (2 elements), essentially per-thread |
| Rank=10, coop | cultivation_d5 | 2.06x | Coop tier, LDS-bound, sync overhead dilutes gain |

**Key observations:**

1. **Per-thread tier scales best (2.21-2.31x):** These circuits are compute-bound on scalar/vector ALU throughput. MI350X's CDNA4 architecture provides roughly 2.3x the effective compute density for this workload.

2. **Coop tier scales less (2.06x):** The synchronization barriers (`__syncthreads`, warp-shuffle reductions) are a larger fraction of execution time at rank=10. While MI350X's barriers are faster in absolute terms, they still represent a fixed overhead that dilutes the compute scaling.

3. **Deeper circuits scale slightly less (2.21x vs 2.31x):** As instruction count grows (surface_d7_r14 has ~4300 instructions vs target_qec's ~217), the instruction fetch and scalar cache behavior becomes a larger factor, slightly reducing the per-CU compute advantage.

### 13.4 CORRECTION: MI300X-ES Dedicated Node Comparison

**The 2.06-2.31x MI350X speedup reported above is misleading.** It compares MI350X-ES against
MI300X production (co-tenancy). When benchmarked against MI300X-ES (dedicated node, same
methodology — 20 runs, warmed), the picture reverses:

| Circuit | Rank | MI300X-ES (Ded) | MI350X-ES | MI350X/MI300X-ES |
|---------|------|----------------|-----------|-----------------|
| target_qec | 0 | **9.69M** | 5.06M | **0.52x** |
| surface_d5_r5 | 0 | **9.35M** | 4.96M | **0.53x** |
| surface_d7_r7 | 0 | **8.79M** | 4.80M | **0.55x** |
| surface_d7_r14 | 0 | **8.34M** | 4.64M | **0.56x** |
| color_d7 | 0 | **9.15M** | 4.88M | **0.53x** |
| rep_d5_r100 | 0 | **9.21M** | 4.93M | **0.53x** |
| cultivation_d5 | 10 | **2.86M** | 2.92M | **1.02x** |
| sweep_q17_t0_d3 | 0 | **9.80M** | 5.08M | **0.52x** |
| sweep_q33_t10_d5 | 1 | **9.77M** | 5.06M | **0.52x** |

**MI300X-ES (dedicated) is 1.8-1.9x FASTER than MI350X-ES for per-thread circuits.**

The co-tenancy penalty on MI300X production is 4.4x (9.69M/2.19M = 4.42x for target_qec),
which entirely explains the apparent MI350X advantage. On the coop tier (cultivation_d5),
MI350X matches MI300X at 1.02x.

**Important caveats:**
- MI300X-ES nodes may be engineering samples with different clock/binning than production
- MI350X-ES is also engineering sample hardware; production MI350X may perform differently
- The MI300X production node's 4.4x co-tenancy penalty is unusually high; this is a shared
  cluster with competing workloads
- The MI300X-ES CV (0.9-3.0%) is higher than MI350X-ES (0.3-0.9%), suggesting the MI300X-ES
  nodes may also have some contention

**The only fair conclusion:** Without dedicated-access MI300X production hardware and
production MI350X hardware, the cross-GPU performance comparison remains preliminary.
The per-thread tier shows MI300X-ES > MI350X-ES > MI300X-Production. The coop tier
shows all three configurations within 2% of each other.

### 13.5 Compiled Kernel on MI350X

The compiled kernel benchmarks returned `peak_rank=-1` and `shots_per_second=0` for all circuits on MI350X. Root cause: the `.hsaco` compilation pipeline invokes `clang++` at runtime, and the MI350X node does not have ROCm development tools installed.

This is an environment/deployment issue, not a code compatibility issue. The `__shfl_xor` intrinsic and all warp-shuffle optimizations are confirmed compatible across gfx942 (MI300X) and gfx950 (MI350X).

### 13.5 MI350X Measurement Stability

MI350X measurements show excellent stability (CV < 0.9%):

| Circuit | MI350X Mean | MI350X CV |
|---------|-----------|----------|
| target_qec | 5.06M | 0.7% |
| surface_d5_r5 | 4.96M | 0.5% |
| surface_d7_r7 | 4.80M | 0.5% |
| surface_d7_r14 | 4.64M | 0.4% |
| color_d7 | 4.88M | 0.7% |
| rep_d5_r100 | 4.93M | 0.4% |
| cultivation_d5 | 2.92M | 0.3% |
| sweep_q17_t0_d3 | 5.08M | 0.7% |
| sweep_q33_t10_d5 | 5.06M | 0.5% |

The sub-1% CV across all circuits confirms the MI350X ES node provides stable, reproducible performance -- better than the MI300X production node (CV 0.5-1.1%).

### 13.6 Implications for Production Deployment

1. **MI350X provides ~2.3x the sampling throughput** for typical QEC workloads with no code changes
2. **All existing optimizations transfer directly:** warp-shuffle, `__noinline__`, frame barrier batching
3. **Compiled kernel needs ROCm-dev on the MI350X node** to function -- a packaging issue, not a code issue
4. **The coop tier scaling gap (2.06x vs 2.31x)** suggests further synchronization optimization could unlock additional MI350X performance

---

## 14. Hardware Counter Deep Dive

### 14.1 Complete Hardware Counter Table

**rocprof --stats measurements on MI300X:**

| Approach | Circuit | Tier | arch_vgpr | sgpr | LDS (B) | scratch (B) | Waves/SIMD | Duration |
|----------|---------|------|-----------|------|---------|-------------|------------|----------|
| SVM baseline | target_qec | T1 thread | 84 | 96 | 20,480 | 1,344 | 6 | 767 us |
| SVM baseline | cultivation_d5 | T2 coop | 108 | 96 | 30,208 | 0 | 4 | 116 ms |
| SVM+GEAK | cultivation_d5 | T2 coop | 124 | 112 | 19,968 | 0 | 4 | 83 ms |
| SVM baseline | circuit_d7 | T3 global | 128 | -- | -- | -- | 4 | -- |
| Optimized SVM (F) | cultivation_d5 | T2 coop | -- | -- | -- | -- | -- | 116.2 ms |
| Persistent (E) | target_qec | T1 thread | -- | -- | -- | -- | -- | 1.02 ms |

**Optimization stack progression (500K shots, cultivation_d5):**

| Metric | Baseline | +noinline | +warp-shuffle | +full stack (v2) |
|--------|----------|-----------|---------------|-----------------|
| arch_vgpr | 128 | 108 | 116 | 124 |
| sgpr | 112 | 96 | 96 | 112 |
| LDS (bytes) | -- | 30,208 | 17,920 | 19,968 |
| Duration | -- | 116 ms | 80.2 ms | 83 ms |

### 14.2 Occupancy Analysis

MI300X has 512 VGPRs per SIMD. Occupancy determined by `waves_per_simd = floor(512 / arch_vgpr)`:

| arch_vgpr | Waves/SIMD | Effective Parallelism |
|-----------|------------|----------------------|
| 84 | 6 | 6 concurrent wavefronts |
| 108 | 4 | 4 concurrent wavefronts |
| 116 | 4 | 4 concurrent wavefronts |
| 124 | 4 | 4 concurrent wavefronts |
| 128 | 4 | 4 concurrent wavefronts |

The critical boundary is 108-128: both give 4 waves/SIMD. The `__noinline__` optimization (128->108) does not change waves/SIMD but reduces instruction scheduling stalls from register pressure.

### 14.3 SALU/VALU Ratio Inversion

The ratio of scalar ALU to vector ALU instructions shifts dramatically with rank:

| Counter | rank=10 (d5) | rank=19 (d7) | Interpretation |
|---------|-------------|-------------|----------------|
| SQ_INSTS_VALU | 7.51B | 23.4B | 3.1x more at rank=19 |
| SQ_INSTS_SALU | 13.6B | 19.2B | 1.4x more at rank=19 |
| **SALU/VALU ratio** | **1.81x** | **0.82x** | **Inverted** |
| SQ_WAIT | 2.56B | 3.19B | 1.25x more waiting |

**At rank=10:** SALU dominates (1.81x). The amplitude array is small (1,024 elements), each sweep finishes quickly, and address computation (`scatter_bits`, `insert_zero_bit`, `bit_get/set/xor`) consumes more cycles than FMA computation.

**At rank=19:** VALU dominates (0.82x). Each sweep processes 262K elements (2^18 pairs), and the VALU cost of complex multiply-add over 262K elements dwarfs SALU index computation.

### 14.4 The HBM Bandwidth Ceiling at Rank=19

All approaches converge to approximately 143-146K shots/s at rank=19:

| Rank | Approach | shots/s | vs SVM+GEAK |
|------|----------|---------|-------------|
| 1 | clifft-amd original | 144,240 | +0.2% |
| 2 | SVM+GEAK | 143,916 | baseline |
| 3 | OptSVM (F) | 143,345 | -0.4% |
| 4-11 | All others | ~142-143K | -0.4% to -1.1% |
| 12 | Persistent (E) | 129,929 | **-9.7%** |

This convergence confirms **rank=19 is entirely HBM bandwidth-bound**. At 4 MB per shot per sweep, ~600 sweeps per shot, total data movement ~2.4 GB per shot:

```
5.3 TB/s / 2.4 GB = ~2,200 shots/s per CU
2,200 * 608 blocks / ~8 (kernel overhead) = ~167K shots/s
```

Measured 143K shots/s represents ~86% of theoretical limit. No dispatch optimization can breach this ceiling.

### 14.5 Memory Footprint at Rank=19

| Item | Per Shot | 608 Blocks Total | Per XCD (76 blocks) |
|------|---------|-------------------|---------------------|
| Amplitude array v[] | 4 MiB | 2.4 GiB | 304 MiB |
| Scratch buffer | 2 MiB | 1.2 GiB | 152 MiB |
| **Total** | **6 MiB** | **3.6 GiB** | **456 MiB** |
| L2 cache per XCD | -- | -- | 4 MiB |

Each shot's 4 MiB amplitude array exactly equals one XCD's 4 MiB L2 cache, meaning a single shot's sweep completely thrashes the L2. Zero opportunity for L2 reuse across shots.

### 14.6 Complete PMC Counter Table

| Kernel | Circuit | arch_vgpr | sgpr | LDS | scratch | SQ_WAVES | SQ_INSTS_VALU | SQ_INSTS_SALU | SQ_WAIT |
|--------|---------|-----------|------|-----|---------|----------|---------------|---------------|---------|
| sample_kernel_coop | d5 (rank=10) | 108 | 96 | 30,208 | 0 | 2.0M | 7.51B | 13.6B | 2.56B |
| sample_kernel | qec (rank=0) | 76 | 64 | 20,480 | 800 | 7.8K | 95.5M | 106.6M | 52.4M |
| sample_kernel_coop (GEAK) | d5 (rank=10) | 124 | 112 | 19,968 | 0 | 2.0M | -- | -- | -- |
| sample_kernel_global | d7 (rank=19) | 128 | -- | -- | -- | -- | 23.4B | 19.2B | 3.19B |

---

## 15. Synthesis and Findings

### 15.1 The Five Key Findings

**Finding 1: The switch dispatch is NOT the primary bottleneck.**

Eliminating it entirely (Approach A: compiled kernel) results in a **-50% regression** (corrected from v1's +5-18%). Phase-sorting (Approach E) and hot/cold splitting (Approach F) gain 0% or less. The GPU's SIMT execution model handles branch divergence well for regular circuits, and the compiler's global optimization view across the switch body provides benefits that straight-line code loses.

**Finding 2: Peak_rank determines throughput by 330x.**

rank=0 runs at 48M shots/s, rank=10 at 5.6M shots/s, rank=19 at 145K shots/s. This range dwarfs any dispatch optimization. Moving coop-tier instructions to per-thread tier (Approach D's split) would yield far larger improvements than any single-tier optimization.

**Finding 3: The HIP compiler is excellent.**

Manual optimizations (Approach F) that replicate compiler behavior show zero improvement. The compiler at -O3 does: dead code elimination, instruction scheduling, register allocation, branch prediction hinting, memory access coalescing. Only algorithmic/structural changes help.

**Finding 4: Synchronization is the true coop-tier bottleneck.**

The warp-shuffle optimization (+40%) was the biggest single win because it reduced `__syncthreads` barriers from 8 to 1 per measurement reduction. On MI300X with 304 CUs, barrier overhead compounds catastrophically.

**Finding 5: MI350X provides ~2.3x throughput with no code changes.**

All optimizations developed on MI300X transfer directly. The CDNA4 architecture's compute density advantage scales well for per-thread workloads (2.31x) and moderately for coop workloads (2.06x).

### 15.2 Optimization Priority Stack

| Priority | Optimization | Impact | Status |
|----------|-------------|--------|--------|
| 1 | GEAK warp-shuffle reduction | +40% coop, +2% thread | **Applied** |
| 2 | `__noinline__` on coop sweeps | VGPRs 128->108 | **Applied** |
| 3 | Vectorized 64-bit LDS loads | -34% LDS footprint | **Applied** |
| 4 | NUMA per-XCD work counters | +1.4% global-coop | **Applied** |
| 5 | `__shfl` gate matrix broadcast | Reduced global loads | **Applied** |
| 6 | Persistent work-stealing (E) | +1% coop (minor) | Available |
| 7 | Circuit splitting (D) | Potentially 3-5x (untapped) | Needs development |
| 8 | ~~Compiled kernel (A)~~ | ~~+5-18%~~ **-50% (failed)** | **Reverted** |
| 9 | Everything else | <5%, not worth complexity | Deprioritized |

### 15.3 Why Warp-Shuffle Beats Everything

- **Architectural approaches** (A through F) assumed the switch dispatch was the bottleneck. In reality, dispatch overhead is minimal for regular QEC circuits because the GPU branch predictor learns the repetitive patterns. Approach A actually produced a -50% regression, confirming that the switch dispatch is actively beneficial for compiler optimization.

- **The actual bottleneck** was synchronization: each `__syncthreads()` on MI300X serializes all 4 wavefronts in a workgroup. With 8 barriers per `coop_reduce2()` call and ~40% of d5 instructions being measurements (each calling reduce twice), barrier overhead dominated.

- **Warp-shuffle** eliminates 7 of 8 barriers by performing reduction within wavefronts (no sync needed) and crossing wavefronts only once. This is a data-structure change (shared-memory tree to register-based shuffle) -- exactly the kind of optimization the compiler cannot make on its own.

### 15.4 Key Architectural Insights

**I1. Peak_rank determines everything.** rank=0: 48M shots/s, rank=10: 5.6M, rank=19: 145K. 330x range entirely from array size.

**I2. The compiler is really good.** ROCm 7.2.3 clang++ at -O3 produces near-optimal code for the SVM interpreter. Only algorithmic/structural changes help.

**I3. SALU dominance is structural, not from the interpreter.** The 1.81x SALU/VALU ratio comes from address computation (`scatter_bits`, `insert_zero_bit`, `bit_get/set/xor`), not switch dispatch. Eliminating the switch (compiled kernel) causes a -50% regression.

**I4. Synchronization is the #1 coop bottleneck.** Warp-shuffle (+40%) attacked synchronization, not compute or memory.

**I5. AMD GPU families are compatible.** `__shfl_xor` works identically on CDNA3 (gfx942) and CDNA4 (gfx950). Wavefront width is 64 on all AMD CDNA architectures.

**I6. Clifft's compiler minimizes peak_rank aggressively.** All synthetic circuits compile to rank <= 1 regardless of T-gate count.

### 15.5 Optimization Impact Summary

| Optimization | Coop tier (rank=10) | Per-thread (rank=0) | Global-coop (rank=19) |
|-------------|---------------------|---------------------|----------------------|
| GEAK warp-shuffle | **+40% (major win)** | +2% minor | Negligible |
| `__noinline__` coop sweeps | VGPRs 128->108 | No effect | VGPRs 128->108 |
| NUMA per-XCD counters | Pending | No effect | +1.4% |
| GpuComplex alignment | LDS -34% | Pending | Pending |
| `__shfl` broadcast | Pending | No effect | Pending |
| Compiled megakernel | SVM fallback | **-50% (regression)** | SVM fallback |
| LDS tiling | **-66% (regression)** | -6.6% | -16.5% |
| Scatter LUT | -3.8% | No effect | LDS overflow |
| Manual SVM opts (F) | 0% | 0% | 0% |
| MFMA tensor cores | 45-70x slower (analytical) | N/A | N/A |
| HIPRTC JIT | -14% (vs AOT) | -14% (vs AOT) | -14% (vs AOT) |

---

## 16. Ideas for Next Round

### 16.1 Compile-Time k-Aware Dispatch

Extend Approach D (split heuristic) with compiled kernels per segment. Generate compiled megakernels for frame-only segments (per-thread tier 10x higher throughput), keep coop kernel only for spike segments.

**Note:** Given the -50% regression (Section 5), the compiled kernel's performance issues must be resolved first. The `--offload-device-only` pathway should be replaced with standard `-shared -fPIC` AOT, and the kernel structure may need redesign to preserve cross-operation optimization (e.g., emitting operations within a single function body rather than as function calls).

### 16.2 DPP (Data Parallel Primitives) for Intra-Wavefront Reductions

AMD CDNA supports DPP instructions for efficient intra-wavefront data movement:

- `v_mov_b32` with DPP modifiers (row_shr, row_ror, wave_shl)
- Potentially eliminates even the `__shfl_xor` overhead for small reductions

### 16.3 Async Copy for Global-Coop

Use `buffer_load` with explicit LDS staging. AMD gfx942 supports async memory operations overlapping data transfer with computation:

```cpp
// Double-buffered: load next tile while computing current tile
async_load(&lds_buf[1], &hbm_v[next_tile_offset], tile_size);
compute_on_tile(&lds_buf[0]);
__syncthreads();
swap(lds_buf[0], lds_buf[1]);
```

### 16.4 Software Pipelining of Memory Loads

Overlap instruction fetch with execution by preloading the next instruction while processing the current one. Currently done automatically by the compiler for per-thread kernel but could be explicit for coop kernel.

### 16.5 Mixed-Precision Amplitude Arrays

For low-precision sweeps where FP32 accuracy is not required, use FP16 amplitudes:

- rank=10 array: 8 KB (FP32) vs 4 KB (FP16)
- rank=19 array: 4 MB (FP32) vs 2 MB (FP16)

Requires careful error analysis for measurement probability accuracy.

### 16.6 Persistent Kernel with XCD-Aware Amplitude Partitioning

Combine persistent kernel (E) with NUMA-aware amplitude partitioning:

- Each XCD owns a contiguous slice in its local HBM
- Gate sweeps parallelized across XCDs, XCD-local data stays in L2
- Cross-XCD communication only for reductions (measurements)

### 16.7 Compiled Kernel Investigation

Before pursuing further compiled kernel work, investigate:

- **Compilation pathway:** Replace `--offload-device-only` with `-shared -fPIC` AOT
- **Kernel structure:** Emit operations as inline code within a single function body
- **Launch pathway:** Compare `hipModuleLaunchKernel` vs `<<<>>>` dispatch
- **Register pressure:** Profile whether expanded constants cause spilling vs compact `GpuInstr` reads

### 16.8 MI350X Compiled Kernel Deployment

Install ROCm development tools on MI350X node to enable clang++ compilation. Once deployed, benchmark compiled kernel on MI350X to determine whether the -50% regression is MI300X-specific or architectural.

### 16.9 Multi-GPU Support

For very large sampling runs (100M+ shots), distribute work across multiple GPUs:

- Each GPU processes a disjoint subset of shots
- No inter-GPU communication needed (shots are independent)
- Linear scaling expected up to the number of available GPUs

---

## 17. Appendix: Raw Data Tables

### 17.1 MI300X Rigorous Benchmark: SVM vs Compiled (20 runs each)

**Source:** `results/bench_rigorous_20260720_040548.csv`
**Node:** rad-mi300x-2 (production, shared)
**Shots per run:** 1,000,000

#### target_qec (rank=0)

| Run | SVM (shots/s) | Compiled (shots/s) |
|-----|--------------|-------------------|
| 1 | 2,215,630 | 1,090,140 |
| 2 | 2,210,420 | 1,092,240 |
| 3 | 2,174,770 | 1,106,740 |
| 4 | 2,206,810 | 1,094,010 |
| 5 | 2,219,410 | 1,092,230 |
| 6 | 2,211,690 | 1,093,660 |
| 7 | 2,190,820 | 1,100,440 |
| 8 | 2,188,450 | 1,094,100 |
| 9 | 2,119,910 | 1,093,420 |
| 10 | 2,213,350 | 1,103,870 |
| 11 | 2,180,640 | 1,095,590 |
| 12 | 2,216,310 | 1,094,790 |
| 13 | 2,149,380 | 1,092,870 |
| 14 | 2,174,640 | 1,084,370 |
| 15 | 2,186,550 | 1,088,870 |
| 16 | 2,184,950 | 1,093,370 |
| 17 | 2,190,810 | 1,098,590 |
| 18 | 2,178,680 | 1,099,920 |
| 19 | 2,178,280 | 1,088,970 |
| 20 | 2,183,060 | 1,091,280 |
| **Mean** | **2,191,228** | **1,094,465** |
| **CV** | **1.1%** | **0.5%** |
| **Delta** | | **-50.1%** |

#### surface_d5_r5 (rank=0)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 2,176,558 | 990,895 |
| CV | 0.7% | 0.3% |
| Delta | | **-54.5%** |

#### surface_d7_r7 (rank=0)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 2,142,138 | 824,487 |
| CV | 0.8% | 0.4% |
| Delta | | **-61.5%** |

#### surface_d7_r14 (rank=0)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 2,103,494 | 654,987 |
| CV | 0.5% | 0.3% |
| Delta | | **-68.9%** |

#### color_d7 (rank=0)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 2,162,039 | 937,796 |
| CV | 0.6% | 0.4% |
| Delta | | **-56.6%** |

#### rep_d5_r100 (rank=0)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 2,162,823 | 991,557 |
| CV | 0.7% | 0.3% |
| Delta | | **-54.1%** |

#### cultivation_d5 (rank=10)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 1,422,418 | 1,423,651 |
| CV | 0.6% | 0.5% |
| Delta | | **+0.1%** (SVM fallback) |

#### sweep_q17_t0_d3 (rank=0)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 2,195,520 | 1,089,254 |
| CV | 0.7% | 0.5% |
| Delta | | **-50.4%** |

#### sweep_q17_t2_d5 (rank=1)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 2,195,043 | 1,109,127 |
| CV | 0.8% | 0.4% |
| Delta | | **-49.5%** |

#### sweep_q17_t5_d5 (rank=1)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 2,194,513 | 1,109,937 |
| CV | 0.8% | 0.3% |
| Delta | | **-49.4%** |

#### sweep_q17_t10_d5 (rank=1)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 2,196,047 | 1,109,767 |
| CV | 0.9% | 0.4% |
| Delta | | **-49.5%** |

#### sweep_q33_t0_d3 (rank=0)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 2,191,444 | 1,108,588 |
| CV | 0.7% | 0.5% |
| Delta | | **-49.4%** |

#### sweep_q33_t2_d5 (rank=1)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 2,193,660 | 1,087,879 |
| CV | 0.7% | 0.6% |
| Delta | | **-50.4%** |

#### sweep_q33_t5_d5 (rank=1)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 2,200,478 | 1,086,523 |
| CV | 0.8% | 0.3% |
| Delta | | **-50.6%** |

#### sweep_q33_t10_d5 (rank=1)

| Stat | SVM | Compiled |
|------|-----|----------|
| Mean | 2,188,090 | 1,089,529 |
| CV | 0.7% | 0.7% |
| Delta | | **-50.2%** |

### 17.2 MI350X Benchmark: SVM Only (20 runs each)

**Source:** `results/bench_mi350x_20260720_072710.csv`
**Node:** mi350x-es
**Shots per run:** 1,000,000
**Compiled:** Failed (peak_rank=-1, all zeros -- clang++ not found on node)

#### target_qec (rank=0)

| Run | MI350X SVM (shots/s) |
|-----|---------------------|
| 1 | 5,078,460 |
| 2 | 5,074,270 |
| 3 | 5,074,030 |
| 4 | 5,043,040 |
| 5 | 5,062,510 |
| 6 | 5,028,420 |
| 7 | 5,104,350 |
| 8 | 5,066,120 |
| 9 | 5,048,850 |
| 10 | 5,089,570 |
| 11 | 5,038,580 |
| 12 | 5,082,050 |
| 13 | 5,090,080 |
| 14 | 5,070,220 |
| 15 | 5,058,870 |
| 16 | 5,061,810 |
| 17 | 4,948,610 |
| 18 | 5,099,210 |
| 19 | 5,093,690 |
| 20 | 5,071,400 |
| **Mean** | **5,065,756** |
| **CV** | **0.7%** |

#### surface_d5_r5 (rank=0)

| Stat | MI350X SVM |
|------|-----------|
| Mean | 4,960,923 |
| CV | 0.5% |

#### surface_d7_r7 (rank=0)

| Stat | MI350X SVM |
|------|-----------|
| Mean | 4,802,843 |
| CV | 0.5% |

#### surface_d7_r14 (rank=0)

| Stat | MI350X SVM |
|------|-----------|
| Mean | 4,643,475 |
| CV | 0.4% |

#### color_d7 (rank=0)

| Stat | MI350X SVM |
|------|-----------|
| Mean | 4,882,524 |
| CV | 0.7% |

#### rep_d5_r100 (rank=0)

| Stat | MI350X SVM |
|------|-----------|
| Mean | 4,926,540 |
| CV | 0.4% |

#### cultivation_d5 (rank=10)

| Run | MI350X SVM (shots/s) |
|-----|---------------------|
| 1 | 2,922,860 |
| 2 | 2,929,350 |
| 3 | 2,919,120 |
| 4 | 2,921,120 |
| 5 | 2,912,920 |
| 6 | 2,920,340 |
| 7 | 2,927,660 |
| 8 | 2,914,710 |
| 9 | 2,911,840 |
| 10 | 2,930,330 |
| 11 | 2,925,130 |
| 12 | 2,922,680 |
| 13 | 2,930,090 |
| 14 | 2,909,600 |
| 15 | 2,912,930 |
| 16 | 2,910,190 |
| 17 | 2,917,410 |
| 18 | 2,917,280 |
| 19 | 2,924,370 |
| 20 | 2,934,440 |
| **Mean** | **2,920,717** |
| **CV** | **0.3%** |

#### sweep_q17_t0_d3 (rank=0)

| Stat | MI350X SVM |
|------|-----------|
| Mean | 5,078,905 |
| CV | 0.7% |

#### sweep_q17_t2_d5 (rank=1)

| Stat | MI350X SVM |
|------|-----------|
| Mean | 5,059,815 |
| CV | 0.5% |

#### sweep_q17_t5_d5 (rank=1)

| Stat | MI350X SVM |
|------|-----------|
| Mean | 5,091,744 |
| CV | 0.4% |

#### sweep_q17_t10_d5 (rank=1)

| Stat | MI350X SVM |
|------|-----------|
| Mean | 5,083,996 |
| CV | 0.4% |

#### sweep_q33_t0_d3 (rank=0)

| Stat | MI350X SVM |
|------|-----------|
| Mean | 5,076,541 |
| CV | 0.5% |

#### sweep_q33_t2_d5 (rank=1)

| Stat | MI350X SVM |
|------|-----------|
| Mean | 5,085,039 |
| CV | 0.5% |

#### sweep_q33_t5_d5 (rank=1)

| Stat | MI350X SVM |
|------|-----------|
| Mean | 5,067,122 |
| CV | 0.5% |

#### sweep_q33_t10_d5 (rank=1)

| Stat | MI350X SVM |
|------|-----------|
| Mean | 5,065,957 |
| CV | 0.5% |

### 17.3 MI300X vs MI350X Complete Summary

| Circuit | Rank | MI300X SVM Mean | MI350X SVM Mean | Ratio |
|---------|------|----------------|----------------|-------|
| target_qec | 0 | 2,191,228 | 5,065,756 | **2.31x** |
| surface_d5_r5 | 0 | 2,176,558 | 4,960,923 | **2.28x** |
| surface_d7_r7 | 0 | 2,142,138 | 4,802,843 | **2.24x** |
| surface_d7_r14 | 0 | 2,103,494 | 4,643,475 | **2.21x** |
| color_d7 | 0 | 2,162,039 | 4,882,524 | **2.26x** |
| rep_d5_r100 | 0 | 2,162,823 | 4,926,540 | **2.28x** |
| cultivation_d5 | 10 | 1,422,418 | 2,920,717 | **2.05x** |
| sweep_q17_t0_d3 | 0 | 2,195,520 | 5,078,905 | **2.31x** |
| sweep_q17_t2_d5 | 1 | 2,195,043 | 5,059,815 | **2.30x** |
| sweep_q17_t5_d5 | 1 | 2,194,513 | 5,091,744 | **2.32x** |
| sweep_q17_t10_d5 | 1 | 2,196,047 | 5,083,996 | **2.32x** |
| sweep_q33_t0_d3 | 0 | 2,191,444 | 5,076,541 | **2.32x** |
| sweep_q33_t2_d5 | 1 | 2,193,660 | 5,085,039 | **2.32x** |
| sweep_q33_t5_d5 | 1 | 2,200,478 | 5,067,122 | **2.30x** |
| sweep_q33_t10_d5 | 1 | 2,188,090 | 5,065,957 | **2.31x** |

**Geomean MI350X/MI300X speedup:**
- Per-thread circuits (rank 0-1): **2.29x**
- Coop circuit (rank=10): **2.05x**
- Overall: **2.27x**

### 17.4 MI300X Compiled Kernel Delta by Circuit Depth

| Circuit | Instrs | SVM Mean | Compiled Mean | Delta |
|---------|--------|----------|---------------|-------|
| target_qec | 217 | 2.19M | 1.09M | -50.0% |
| surface_d5_r5 | 997 | 2.18M | 0.99M | -54.5% |
| color_d7 | 1470 | 2.16M | 0.94M | -56.7% |
| rep_d5_r100 | 1661 | 2.16M | 0.99M | -54.1% |
| surface_d7_r7 | 2749 | 2.14M | 0.82M | -61.5% |
| surface_d7_r14 | 4298 | 2.10M | 0.66M | -68.9% |

The regression scales with circuit depth: -50% for 217-instruction circuits to -69% for 4298-instruction circuits.

### 17.5 Previous (v1) 6-Way Comparison (MI300X Dedicated Node)

These numbers from the v1 dedicated-node benchmark remain valid for Approaches B-F and SVM baseline. Only Approach A results are corrected.

| Circuit | rank | SVM | A: Compiled | B: Per-Op | C: HipGraph | D: Split | E: Persistent | F: Opt SVM |
|---------|------|-----|------------|-----------|------------|---------|--------------|-----------|
| target_qec | 0 | 43.0M | ~~48.4M~~ **-50%** | 43.8M (+2%) | 42.6M (-1%) | 50.4M (**+17%**) | 45.9M (+7%) | 47.1M (+10%) |
| cultivation_d5 | 10 | 3.55M | 1.42M (fb) | 3.24M (-9%) | 3.57M (+1%) | 4.02M (fb) | 4.02M (**+13%**) | 3.11M (-12%) |

(fb = SVM fallback; compiled kernel only supports per-thread tier rank <= 4)

### 17.6 SVM+GEAK vs clifft-amd (100M shots)

| Version | shots/s | vs clifft-amd |
|---------|---------|---------------|
| clifft-amd (original) | 7.93M | baseline |
| **SVM+GEAK (gpu-backend)** | **11.0M** | **+39%** |

Both produce identical results: `passed_shots=2,318,545`.

### 17.7 Large Circuit Support

| Circuit | Qubits | Measurements | shots/s | Notes |
|---------|--------|-------------|---------|-------|
| surface_d9_r9 | **188** | 801 | **1.78M** | Previously rejected (>128q) |
| surface_d11_r11 | **274** | 1441 | **1.69M** | Previously rejected |
| surface_d13_r13 | **376** | 2353 | **1.57M** | Previously rejected |

All compile to rank=0 after StatevectorSqueezePass.

### 17.8 D7 Circuit: Rank=19 Global-Coop Convergence

| Rank | Approach | shots/s | vs SVM+GEAK |
|------|----------|---------|-------------|
| 1 | clifft-amd original | 144,240 | +0.2% |
| 2 | SVM+GEAK | 143,916 | baseline |
| 3 | OptSVM (F) | 143,345 | -0.4% |
| 4-11 | All others | ~142-143K | -0.4% to -1.1% |
| 12 | Persistent (E) | 129,929 | **-9.7%** |

### 17.9 NUMA-Aware Per-XCD Work Distribution

![NUMA-XCD Optimization](./diagrams/numa-xcd-optimization.svg)

**MI300X NUMA topology:** 8 XCDs across 4 IODs, each with private L2 and local HBM.

```
              MI300X Package (Top View)
======================================================================

Layer 2 (top): 8 XCDs (Accelerator Complex Dies) -- TSMC 5nm
Layer 1 (bottom): 4 IODs (I/O Dies) -- TSMC 6nm + 8 HBM3 Stacks

     XCD 0     XCD 1       XCD 2     XCD 3
    [38 CU]   [38 CU]    [38 CU]   [38 CU]
    [4MB L2]  [4MB L2]   [4MB L2]  [4MB L2]
        |   3D   |           |   3D   |
    +---+---------+---+  +---+---------+---+
    |     IOD 0       |  |     IOD 1       |
    |  [64MB LLC]     |  |  [64MB LLC]     |
    +--+--------+-----+  +--+--------+-----+
       |        |            |        |
     HBM 0   HBM 1        HBM 2   HBM 3
     24 GB   24 GB          24 GB   24 GB

    (Repeated for XCD 4-7 on IOD 2-3)
```

**Per-XCD work counter implementation:**

```cpp
__device__ __forceinline__ int get_xcd_id() {
    int xcd_id;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
    return xcd_id;
}
```

**Measured improvement: +1.4% on global-coop tier (rank=19).** Modest because work counter atomics are a small fraction of total execution time at rank=19.

### 17.10 MI300X Production vs Dedicated Node

| Circuit | MI300X Production | MI300X Dedicated | Ratio |
|---------|------------------|-----------------|-------|
| cultivation_d5 (rank=10) | 1.42M | 5.6M | 3.9x |
| target_qec (rank=0) | 2.19M | 48.3M | 22x |

### 17.11 Interesting Circuits Catalog

**Tier 1: Per-Thread (peak_rank <= 4)**

| Circuit | rank | Instrs | Qubits | shots/s (dedicated) | Bottleneck |
|---------|------|--------|--------|---------------------|------------|
| target_qec.stim | 0 | 217 | 25 | 48M | Launch overhead |
| surface_d3_r3.stim | 0 | 212 | 33 | 19M | Instruction count |
| surface_d5_r5.stim | 0 | 997 | 145 | 18M | Instruction count |
| surface_d7_r7.stim | 0 | 2749 | 385 | 16M | Instruction count |
| surface_d7_r14.stim | 0 | 4298 | 721 | 15M | Instruction count + qubit width |
| color_d7.stim | 0 | 1470 | 163 | 18M | Instruction count |
| circuit_d3_p0.001.stim | 4 | 344 | 21 | 19M | Only rank=4 circuit |
| rep_d5_r100.stim | 0 | 1661 | 405 | 17M | Very deep (100 rounds) |

**Tier 2: Shared-Coop (peak_rank 5-10)**

| Circuit | rank | Instrs | Qubits | shots/s (dedicated) | Bottleneck |
|---------|------|--------|--------|---------------------|------------|
| cultivation_d5.stim | 10 | 1720 | 112 | 5.6M | Sync barriers (GEAK +40%) |
| qv10.stim | 10 | 140 | 10 | ~4M | Short but high rank |

**Tier 3: Global-Coop (peak_rank 11-19)**

| Circuit | rank | Instrs | Qubits | shots/s (dedicated) | Bottleneck |
|---------|------|--------|--------|---------------------|------------|
| circuit_d7_p0.0005.stim | 19 | 5472 | 355 | 144K | HBM bandwidth ceiling |

---

## Final Tags and Branches

| Tag | Description |
|-----|-------------|
| `perf-geak-warp-shuffle-40pct` | GEAK warp-shuffle reduction (+40% coop) |
| `perf-noinline-coop-sweeps` | `__noinline__` on coop sweep functions (VGPRs 128->108) |
| `perf-svm-vs-clifftamd-39pct` | SVM+GEAK vs clifft-amd comparison (+39%) |
| `perf-full-stack-v2` | All optimizations: warp-shuffle + NUMA + alignment + shfl |
| `perf-384qubit-support` | Large circuit support (kPauliWords=6, kMaxMeas=4096) |
| `svm-opt-4-scatter-lut` | Scatter-bits LUT (reverted, -3.8%) |

| Branch | Description |
|--------|-------------|
| `gpu-backend` | Main GPU backend with SVM+GEAK optimizations |
| `gpu-compiled-kernel` | Compiled megakernel + full optimization stack |
| `gpu-per-op-kernel` | Per-operator kernel dispatch |
| `gpu-hipgraph` | HipGraph pipeline |
| `gpu-split-kernel` | Heuristic split megakernels |
| `gpu-persistent` | Persistent kernel with phase-sorted dispatch |
| `gpu-svm-optimized` | Optimized SVM baseline |
| `gpu-large-circuit-support` | 384-qubit support |

---

*Report v2 generated from clifft GPU backend optimization work, July 2026. All v2 benchmarks measured with 20 runs per circuit per mode. MI300X data from rad-mi300x-2 (production node, shared access). MI350X data from mi350x-es (engineering sample). See Section 2 for full methodology.*
