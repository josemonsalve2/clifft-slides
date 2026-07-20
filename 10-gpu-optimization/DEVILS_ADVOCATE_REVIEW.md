# Devil's Advocate Review: Clifft GPU Kernel Optimization

**Reviewer role:** Skeptical peer reviewer challenging every claim with specific data questions.

**Date:** July 20, 2026

---

## 1. Challenging the -50% Compiled Kernel Regression

The report claims the compiled megakernel is 49-69% slower than the SVM interpreter. Before accepting this as a definitive "compiled kernels don't work" conclusion, several alternative explanations must be ruled out:

### 1.1 Is `--offload-device-only` the right compilation flag?

The kernel_cache.cc uses `clang++ --offload-device-only` to produce a standalone `.hsaco`. This flag is designed for producing device-only binaries, but it may invoke a **different compilation pipeline** than the full `hipcc` flow used for the SVM kernel.

**What to check:**
- Compare `rocminfo --show-isa` output from the compiled kernel vs the SVM kernel
- Use `llvm-objdump --disassemble` on both `.hsaco` files to compare ISA quality
- Try `--cuda-device-only` (older flag) or `-fgpu-rdc` (relocatable device code) as alternatives
- Compare VGPR counts: `llvm-objdump -d kernel.hsaco | grep .vgpr_count`

**Risk:** The -50% could be entirely from inferior compilation, not from the approach itself. A properly compiled straight-line kernel might still outperform the SVM.

### 1.2 hipModuleLaunchKernel overhead

The compiled kernel is launched via `hipModuleLaunchKernel()` while the SVM uses `<<<>>>` syntax. These may have different dispatch paths:

- `<<<>>>` goes through the HIP runtime's kernel launch optimization (argument packing, dispatch coalescing)
- `hipModuleLaunchKernel()` is the lower-level HSA-based path

**Experiment needed:** Benchmark a trivially different kernel (e.g., the SVM kernel with one constant changed) launched via both `<<<>>>` and `hipModuleLaunchKernel()` to isolate the dispatch overhead.

### 1.3 Missing compiler flags

The SVM kernel is compiled through cmake with flags like `-ffp-contract=off`, optimization level from `CMAKE_BUILD_TYPE=Release`, and potentially `-march=native`. The `--offload-device-only` path gets `-O3` but may be missing:
- `-ffp-contract=off` (clifft requires precise FP for correctness)
- Target-specific tuning flags
- Link-time optimization passes

**Experiment needed:** Dump the exact compiler command lines for both paths and diff them.

### 1.4 VGPR explosion from expanded constants

The generated kernel bakes all constant pool data (Pauli masks, noise channels, fused U2/U4 matrices) as `__device__` constant arrays. For a large circuit, this could mean:
- Hundreds of constant array elements in global scope
- The compiler may fail to place them in constant memory (read-only cache)
- Constants may spill to VGPRs during address computation

**Experiment needed:** Use `rocprof --stats` to compare VGPR counts between SVM and compiled kernels on the same circuit.

---

## 2. Challenging the MI300X vs MI350X Comparison

### 2.1 Not apples-to-apples

The MI300X numbers (2.19M) are from `rad-mi300x-2`, a **production node with co-tenancy**. The MI350X numbers (5.06M) are from `smci350-rck-g03-d13-21`, which appears to be a **dedicated ES node**.

The RESULTS.md itself documents 1.5-8.5x variation between production and dedicated MI300X nodes:
- MI300X production: 5.63M (target_qec)
- MI300X dedicated: 47.9M (target_qec)
- Ratio: 8.51x

**This means the 2.31x MI350X/MI300X ratio is meaningless** without running on equivalent node types. The MI300X dedicated nodes might show 47.9M for target_qec vs MI350X's 5.06M — making MI300X appear 9.5x *faster*.

**Critical experiment:** Run the same benchmark on `mi300x-es` (dedicated MI300X nodes, 3 idle) to get a fair comparison.

### 2.2 ES hardware caveat

MI350X-ES (Engineering Sample) may not represent final silicon performance. Clock speeds, memory bandwidth, and power limits may differ from production MI350X.

---

## 3. Challenging the GEAK +40% Warp-Shuffle Claim

### 3.1 Methodology consistency

The +40% warp-shuffle speedup was measured in a *previous* session using a different methodology:
- Was it a single run or multiple runs?
- Was the GPU warmed up?
- Was the node exclusive?

The APPLIED_OPTS.md shows: "cultivation_d5: 4.0M → 5.6M shots/s" but doesn't specify:
- Number of runs
- Standard deviation
- Whether this was on the same node as the -50% compiled kernel measurement

**If the +40% was a single-run measurement (like the original flawed T-gate sweep), it may be noise.**

### 3.2 Current data doesn't show +40%

The MI300X rigorous benchmark shows cultivation_d5 SVM at 1.42M shots/s. The original claim was 4.0M → 5.6M. The discrepancy (1.42M vs 4.0M) is 2.8x — consistent with production vs dedicated node difference, but it means we can't verify the +40% from this data.

**To verify:** Run on `mi300x-es` (dedicated) with the GEAK warp-shuffle reverted (revert commit `2e92b47`) and compare against current code. 20 runs each.

---

## 4. Challenging "The Compiler is Excellent"

### 4.1 Self-contradictory evidence

The report claims "ROCm clang++ -O3 produces near-optimal code for the SVM interpreter" (Approach F yielded 0% improvement). But then the compiled kernel (also compiled with clang++ -O3) is -50%.

If the compiler is excellent, why does it produce such bad code for the straight-line kernel? Possible explanations:
- It's not the same compiler pipeline (hipcc vs clang++ --offload-device-only)
- The straight-line code structure is harder to optimize than the switch-dispatch
- The constant pool embedding creates a different optimization landscape

**This suggests the compiler is excellent *for the specific code pattern of the SVM interpreter*, not universally.** The claim should be scoped.

### 4.2 Different optimization surfaces

The switch-dispatch loop has a compact instruction cache footprint — all opcodes fit in I-cache. The straight-line kernel unrolls everything, potentially causing I-cache thrashing for large circuits.

**Experiment needed:** Compare I-cache miss rates (SQ_IFETCH_LEVEL counter in rocprof) between SVM and compiled kernels.

---

## 5. Challenging Methodology

### 5.1 Statistical power

20 runs with CV ~0.7% gives a standard error of mean = 0.7% / sqrt(20) = 0.16%. A 50% difference is ~300 standard errors — definitely significant. But for detecting *small* improvements (e.g., 2-3%), 20 runs may not be sufficient.

**For future work:** Use at least 50 runs when expecting <5% deltas.

### 5.2 Run independence

Were the 20 runs truly independent? The script runs them sequentially in a loop with no GPU reset between runs. GPU state (caches, TLB, thermal throttle point) may carry over.

**Experiment:** Interleave SVM and compiled runs (ABAB pattern) instead of batch (AAAA...BBBB...) to detect order effects.

### 5.3 Shot count sensitivity

All benchmarks use 1M shots. The throughput at 1M shots may differ from 100M shots due to:
- Kernel launch overhead amortization
- GPU occupancy ramp-up time
- Memory allocation patterns

**Experiment:** Run target_qec at 100K, 1M, 10M, 100M shots to check throughput stability.

---

## 6. Missing Experiments (Must-Do List)

1. **MI300X dedicated node benchmark** — Run on `mi300x-es` (splinter nodes) for fair comparison with MI350X
2. **VGPR comparison** — `rocprof --stats` on SVM vs compiled kernel to see register pressure difference
3. **Compiler flag parity** — Dump and diff exact compiler flags between hipcc (SVM) and clang++ --offload-device-only (compiled)
4. **GEAK +40% re-verification** — 20-run rigorous benchmark with warp-shuffle reverted vs current
5. **Alternative compilation path** — Try HIPRTC, `hipcc --genco`, or linking the generated .hip as a proper cmake target instead of --offload-device-only
6. **I-cache pressure** — rocprof SQ_IFETCH_LEVEL counter comparison
7. **Shot count sweep** — 100K to 100M shots on target_qec to verify throughput scaling
8. **Interleaved run order** — ABAB pattern to detect order effects
9. **circuit_d7 (rank=19)** — The most important missing circuit. The d7 benchmark was absent from the MI300X rigorous run (or failed). This is the HBM-bound workload.
10. **Thermal throttle check** — Monitor GPU temperature during the 20-run sweep to verify no throttling

---

## 7. Summary of Verdict

| Claim | Confidence | Issue |
|-------|-----------|-------|
| Compiled kernel is -50% | **Medium** | May be --offload-device-only issue, not inherent to approach |
| MI350X is 2.3x faster than MI300X | **Low** | Not apples-to-apples (production vs dedicated) |
| GEAK +40% warp-shuffle | **Medium** | Not re-verified with rigorous methodology |
| Compiler is excellent | **High for SVM** | Scoped to switch-dispatch pattern, not universal |
| 188% outlier was transient | **High** | Proven by 20-run investigation |

**Bottom line:** The original +5-18% compiled kernel claim was correctly debunked. But the -50% result may also be misleading — it could be a compilation pipeline issue rather than an inherent problem with the compiled kernel approach. Before declaring the approach dead, try a proper compilation path (hipcc --genco or cmake integration).
