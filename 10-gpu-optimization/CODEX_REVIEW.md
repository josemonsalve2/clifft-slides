# Technical Review: Clifft GPU Optimization Report

This review is grounded in the following sources:

- `10-gpu-optimization/REPORT.md` (the main report)
- `clifft/RESULTS.md` (benchmark and counter record)
- `clifft/docs/LESSONS_LEARNED.md` (success and failure analyses)
- `clifft/src/clifft/gpu/hip_sampler.hip` (current SVM, cooperative, and global-cooperative kernels)

Source references below use file names, section names, and kernel/function names so that each observation can be traced without relying on benchmark values not present in the source material. Potential gains in Sections 3 and 6 are estimates, not measured results, unless explicitly identified as measured.

## 1. Technical Accuracy

### Correct or well-supported conclusions

- The report is correct that synchronization reduction is the strongest demonstrated cooperative-tier optimization. `LESSONS_LEARNED.md` reports the warp-shuffle reduction as the largest success, and `RESULTS.md` records the rank-10 throughput change from about 4.0M to 5.62M shots/s in the definitive comparison. The source kernel confirms that measurement paths call `coop_reduce2()` and then perform additional block synchronization (`hip_sampler.hip`, `coop_reduce2`, `coop_meas_active_diagonal`, and `coop_meas_active_interfere`).

- The report is correct that `__noinline__` can be a useful register-lifetime boundary. The current kernel keeps `coop_u2_sweep` and `coop_u4_sweep` out of line and passes raw pointers rather than a `CoopShotState&` (`hip_sampler.hip`, `coop_u2_sweep` and `coop_u4_sweep`). This matches the failure analysis that inlining large matrix temporaries into the interpreter can extend live ranges across the dispatch loop (`LESSONS_LEARNED.md`, Success 2).

- The conclusion that interpreter removal helps primarily in the per-thread tier is supported by the benchmark record. `RESULTS.md` shows measured compiled-kernel improvements on selected rank-1 circuits, while the cooperative and global-cooperative paths generally fall back or converge near the SVM results.

- The rank-dependent bottleneck shift is directionally credible. `RESULTS.md` reports SALU/VALU ratios of 1.81 at rank 10 and 0.82 at rank 19, while `hip_sampler.hip` shows repeated scalar index construction (`scatter_bits_1`, `scatter_bits_2`, bit operations) around vector complex arithmetic. It is reasonable to conclude that index/control overhead has greater relative weight for short rank-10 sweeps than for long rank-19 sweeps.

- The report correctly rejects MFMA as an obvious fit for these kernels. The operations in `array_u2`, `array_u4`, `coop_u2_sweep`, and `coop_u4_sweep` are tiny complex butterflies with irregular index placement, not large reusable matrix tiles. The specific cycle estimates in the report are not independently demonstrated, but the architectural conclusion is sound.

- The report correctly identifies an important negative result for naive LDS tiling. `LESSONS_LEARNED.md` records regressions of 66.1% at rank 10 and 16.5% at rank 19, and the current direct sweep structure confirms why extra tile-level barriers can be expensive.

### Incorrect, internally inconsistent, or overstated claims

- **The VGPR occupancy model is presented too literally and then applied inconsistently.** The report states `waves/SIMD = floor(512 / arch_vgpr)` and gives 84→6 waves, 108→4, 116→4, 124→4, and 128→4 (`REPORT.md`, Sections 2.4 and 14.1). Those integer results match the requested formula. However:

  - The hardware table simultaneously says “VGPRs per SIMD: 1216; 512 architectural,” which is unexplained and mixes physical capacity, architectural addressing, and profiler-reported allocation units.
  - The formula is at best a simplified upper bound. Actual residency is also constrained by allocation granularity, maximum waves per SIMD/CU, SGPRs, LDS per workgroup, workgroup size, and any scratch use.
  - The statement in `LESSONS_LEARNED.md` that 108 and 128 VGPRs both give four waves is consistent with the simplified formula, but it does not prove occupancy is unchanged in practice. The report needs measured occupancy or a complete occupancy calculation, not only integer division.
  - The wording “each CU can execute 4 wavefronts simultaneously on its 4 SIMD units” is misleading. A CU has four SIMDs, but each SIMD can have multiple resident waves; “four simultaneously” conflates SIMD issue resources with resident wave capacity.

- **The warp-shuffle reduction uses two block barriers, not one.** The main report repeatedly calls the optimized `coop_reduce2()` a “1 barrier” reduction and says “8 barriers per reduce call reduced to 1” (`REPORT.md`, Section 10.2). The actual function has one `__syncthreads()` after wave leaders write partials and a second after thread 0 writes the final result (`hip_sampler.hip`, `coop_reduce2`). `RESULTS.md` correctly calls this a “2-barrier inter-wavefront step.” The report should say “tree reduction reduced from approximately eight/nine block-wide synchronization points to two,” with the exact baseline count defined from the actual old implementation.

- **The report says each measurement calls `coop_reduce2()` twice, but the current code calls it once per active measurement operation.** Each call reduces two accumulators simultaneously (`local0` and `local1`). `coop_meas_active_diagonal` makes one call for `(p0,p1)`, and `coop_meas_active_interfere` makes one call for `(p_plus,p_minus)`. The wording likely confuses “two values reduced” with “two function calls.”

- **“ShotState is entirely register-resident” is too strong.** The per-thread kernel has reported scratch allocation—800 bytes in one counter table and 1,344 bytes in another (`RESULTS.md`; `REPORT.md`, Sections 4.5, 14.1, and 17.17). `ShotState` contains large `meas`, `obs`, amplitude, and expectation arrays (`hip_sampler.hip`, `ShotState`). Some values may be scalarized into registers, but the profiler evidence shows private/scratch backing. The report should describe it as “primarily thread-private and compiler-scalarized where possible,” not entirely register-resident.

- **The fixed “ShotState is ~2.4 KB” serialization argument is insufficiently grounded.** The current `ShotState` size depends on constants such as `kMaxMeas`, `kMaxObs`, and `kMaxExpVals`, alignment, and which state a per-op implementation actually materializes. The report should provide `sizeof(ShotState)` from the exact branch/build or a field-by-field calculation. It also should distinguish logical state size from bytes actually transferred after compiler dead-store elimination or a structure-of-arrays redesign.

- **The statement that scratch allocation is page-granular and therefore reducing `meas[]` cannot help is asserted without supporting counters.** `RESULTS.md` and `REPORT.md` repeat this explanation, but no before/after scratch-byte counter is shown for the 1024→256-byte change. Private-memory allocation and scratch traffic are not interchangeable: even if an allocation unit does not change, reduced address range or fewer live accesses can still matter. This should be labeled a hypothesis unless backed by resource-usage and scratch-traffic measurements.

- **The “GPU branch predictor learns the 35-case switch” explanation is speculative and likely uses CPU terminology too casually.** The evidence establishes that phase sorting and hot/cold restructuring did not help. It does not establish a conventional dynamic branch predictor as the reason. On a wavefront machine, uniform control flow, compiler-generated jump tables/branch sequences, instruction-cache behavior, and identical opcode streams across lanes can make dispatch inexpensive without requiring the claimed predictor behavior. This should be rewritten as: “dispatch is largely wave-uniform and regular; measured results show it is not dominant.” Label any branch-prediction mechanism as a hypothesis.

- **The L2-prefetcher claim is not proven.** The report says the L2 hardware prefetcher tracks the butterfly stride (`REPORT.md`, Section 11.8; `LESSONS_LEARNED.md`, F8). The failed LDS-tiling experiment proves only that that implementation’s barrier and state overhead exceeded any benefit. It does not prove an L2 prefetcher handled all strides well. Large power-of-two strides may produce poor sector utilization or partition effects even when accesses are deterministic. L2 hit/miss and cache-request counters are required.

- **The rank-19 “entirely HBM bandwidth-bound” and “86% of theoretical limit” claims are not established by the shown arithmetic.** The report estimates 2.4 GB per shot from 4 MB × ~600 sweeps, then divides peak 5.3 TB/s by that quantity. That direct roofline gives roughly 2,200 shots/s for the whole device, not per CU. The subsequent multiplication by 608 blocks and division by 8 has no justified dimensional basis. Measured 143K shots/s multiplied by 2.4 GB/shot would imply hundreds of TB/s, proving that either “600 full 4 MB sweeps” is not the executed traffic model or cache/reuse/active-rank assumptions are missing. Convergence of implementations suggests a common bottleneck, but HBM saturation must be demonstrated with measured HBM bytes/bandwidth and cache hit rates.

- **“A single 4 MiB shot exactly equals one XCD’s 4 MiB L2, therefore it completely thrashes L2 and has zero reuse” is too absolute.** Capacity equality alone does not prove complete thrashing. Set associativity, line mapping, concurrent blocks, code/data traffic, partial sweeps, and reuse distance determine behavior. The source does show two blocks per CU are requested for global-coop (`global_worker_blocks = CU_count * 2`), so many working sets compete, but the report still needs L2 counters before claiming zero reuse.

- **MI300X specification details need authoritative sourcing and clearer terminology.** The report lists 8 XCDs, 304 CUs, 192 GB HBM3, 5.3 TB/s, 4 MB L2 per XCD, and 64 MB LLC per IOD. These may be intended as hardware facts, but the four reviewed files do not contain a primary hardware citation. The “1216 / 512 architectural” VGPR row is especially unclear. The report should cite the AMD MI300X/CDNA3 architecture or ISA documentation and define whether profiler `arch_vgpr` values are per thread, allocation units, or logical registers.

- **The source and prose disagree about scatter-LUT status.** `LESSONS_LEARNED.md` says the LUT was reverted, but the reviewed `hip_sampler.hip` still declares `scatter_lut`, cache metadata, `ensure_scatter_1/2`, and a 512-entry shared LUT in `sample_kernel_coop`. The `CoopShotState` initializer also appears visually error-prone because it supplies `nullptr` and `scatter_lut` around a long pointer list. The report must identify the exact commit used for each benchmark and reconcile whether the reviewed kernel is pre-revert, post-revert, or a later reintroduction.

- **Several “final” performance statements contradict one another.** Examples include:

  - Approach F is described as 0% everywhere, then as +10% or -12% in early tables, -16.6% in a definitive table, and later as leading one all-workspace D5 comparison.
  - Persistent coop is reported as -3%, +13%, and +1.1% depending on table/baseline.
  - Compiled target-QEC results range from +13% to -2.3% and later +1–2%.
  - Warp-shuffle is described as +14.9%, +28% duration reduction, +40% throughput, and part of a +39% end-to-end comparison against a different baseline.

  These can all arise from different commits, nodes, warmup states, or baselines, but the report currently presents them as one narrative. Every table needs a run ID, commit/tag, node, ROCm version, shot count, warmup policy, and baseline definition.

- **The report’s use of “JIT” for the clang++ subprocess path is imprecise.** It is runtime-generated AOT-quality compilation into an HSACO, not HIPRTC-style in-process JIT. “Runtime code generation with external AOT compilation and disk caching” is clearer.

### SALU/VALU interpretation

- Instruction-count ratios are useful but do not by themselves identify the cycle bottleneck. SALU and VALU have different issue paths, dependencies, and overlap. A SALU/VALU ratio of 1.81 does not mean SALU consumes 1.81× the time or that reducing SALU by 30% saves 30% of cycles.

- The report’s later back-of-the-envelope SALU time calculation assumes instruction count divided by CU clock is elapsed time. That ignores issue width, wave distribution, instruction latency, dual-issue/overlap, and stalls. The conclusion that a hypothetical SALU reduction saves 0.26% should be marked as a rough upper/lower-bound exercise, not a quantitative prediction.

- Better evidence would combine instruction counts with active cycles, issue-utilization counters, dependency/wait counters, and per-pipeline busy percentages.

## 2. Missed Optimizations (be thorough)

The project explored several architectural alternatives, reductions, register-lifetime control, a scatter LUT, vectorized LDS accesses, NUMA-aware counters, and one LDS-tiling design. The following opportunities were not explored, or were mentioned only as future ideas without implementation evidence.

### 2.1 Amplitude-array coalescing by axis class

- The sweep mapping is `i = threadIdx.x + n*blockDim.x`, followed by `scatter_bits_1/2`. Whether neighboring lanes access neighboring `GpuComplex` values depends strongly on `axis`.
- For low axes, lanes can access compact pairs; for high axes, each half of a pair may be far apart while each stream is still contiguous. For two-axis gates, the four streams may have different locality.
- A missing analysis is an axis-bucketed memory study: bytes/request, cache-line or sector utilization, and achieved bandwidth for every axis and gate class.
- **Hypothesis:** Specialized lane-to-index mappings for low-, medium-, and high-axis cases could improve global-coop memory transaction efficiency by 5–20% on poorly coalesced cases, but the average gain may be small if current streams are already coalesced.

### 2.2 Incremental index generation instead of repeated scatter

- Every loop iteration recomputes `scatter_bits_1/2`, even though each thread walks a regular arithmetic sequence.
- A per-thread incremental generator could update the physical index using carry/bit-transition logic, or process short contiguous runs between inserted-bit boundaries.
- This directly targets the reported rank-10 SALU pressure without an LDS LUT or its register/cache metadata.
- **Hypothesis:** 2–10% on index-heavy rank-5–10 sweeps if the incremental state fits in fewer live registers than the LUT implementation.

### 2.3 Instruction-level parallelism in sweep bodies

- Current U2/U4 loops generally load one butterfly, compute it, and store it before the next iteration.
- Processing two independent butterflies per iteration can overlap LDS/global load latency with complex arithmetic and expose more independent FMA chains.
- This must be tuned against VGPR growth; the failed LUT demonstrates that more registers can hurt even without a nominal occupancy step.
- **Hypothesis:** 3–12% on global-coop and compute-heavy U4 paths; potentially negative on rank-10 if register pressure increases scheduler stalls.

### 2.4 Software pipelining of global loads

- For global-coop sweeps, preload butterfly `n+1` into registers while computing butterfly `n`, without staging the whole tile in LDS.
- This avoids the three-barrier failure mode of the prior LDS tiling experiment.
- Separate variants should be tested for U2/U4 and measurement reductions because their arithmetic intensity differs.
- **Hypothesis:** 5–15% where VMEM latency, rather than bandwidth, is limiting.

### 2.5 L1/L2 cache policy experiments

- The report infers cache behavior but does not test cache policy or collect hit rates.
- Candidate experiments include default versus streaming/non-temporal behavior for one-pass stores, cacheable loads for repeated measurement/fold passes, and worker-count changes to reduce simultaneous 4 MiB working sets per XCD.
- The fixed `2 * CU_count` global worker count is especially important: fewer blocks may reduce memory-level parallelism but improve cache residency and reduce working-set contention.
- **Hypothesis:** -5% to +15%; this is highly workload- and rank-dependent.

### 2.6 BUFFER_LOAD/STORE versus FLAT_LOAD/STORE selection

- `global_v` and `global_scratch` are base-plus-index arrays with known bounds, a pattern that can be suitable for buffer instructions.
- The C++ pointer dereferences in `load_complex64` do not guarantee the desired global-memory ISA. The report discusses `buffer_load` only as an async-copy idea and does not show disassembly.
- Compile and inspect variants using AMDGPU builtins or inline assembly where justified, comparing instruction form, address-generation SALU, cache policy, and bounds handling.
- **Hypothesis:** 2–8% if buffer addressing reduces address arithmetic or improves scheduling; 0% if LLVM already selects equivalent instructions.

### 2.7 DPP for intra-wave reductions

- DPP is mentioned but not benchmarked.
- A 64-lane full-wave reduction may require careful row-crossing stages; not every DPP mode communicates across the entire wave. The implementation must account for 16-lane row semantics where applicable.
- Compare DPP, `ds_bpermute`/`ds_permute`, and `__shfl_down`/`__shfl_xor` using generated ISA and cycle counters.
- **Hypothesis:** 0–5% on measurement-heavy rank-10 circuits because only the intra-wave portion changes and two inter-wave barriers remain.

### 2.8 `ds_permute`/`ds_bpermute` for wave communication

- These instructions can move lane values through LDS hardware without a conventional shared-memory array and may be useful for non-XOR communication patterns or matrix broadcasts.
- They are also candidates for lane remapping of small butterflies that fit within a wave, potentially avoiding some LDS round trips.
- **Hypothesis:** 0–8% for selected low-rank cooperative kernels; unlikely to help rank-19 HBM sweeps directly.

### 2.9 Barrier elimination via wave-specialized block layouts

- `coop_reduce2()` still uses two block barriers, and most cooperative operations add barriers before and after thread-0 frame updates.
- A wave-specialized design could assign wave 0 to frame/RNG/bookkeeping and waves 1–3 to array work, using explicit flags/counters and narrower synchronization.
- This is more radical than simply “some threads do frame ops”; it requires proving that stale frame state cannot be observed by array waves.
- **Hypothesis:** 5–20% for instruction streams with many frame/noise/bookkeeping operations, but complexity and correctness risk are high.

### 2.10 Predication versus branch variants

- `array_multi_cnot`, `array_multi_cz`, measurement collapse, and frame updates contain data-dependent conditions.
- For cheap operations with approximately balanced predicates, predicated select/XOR may avoid divergent control. For sparse predicates, branching can avoid unnecessary stores.
- The correct choice should be driven by predicate density measured per circuit.
- **Hypothesis:** 1–6% on affected opcodes; likely negligible for fully wave-uniform branches such as opcode dispatch.

### 2.11 Loop unrolling and gate-specific iteration shapes

- No explicit unroll pragmas or specialized fixed-trip-count variants are visible in the current sweep loops.
- Rank 5–10 gives small, known iteration counts per thread. Partial unrolling may reduce loop/control SALU and expose ILP.
- Rank-19 loops should use modest unrolling to avoid code-size and VGPR growth.
- **Hypothesis:** 2–10% on U2/U4 or measurement loops after per-op tuning.

### 2.12 Register packing and reduced precision

- Packing two FP16/BF16 amplitudes per VGPR is mentioned only as a general mixed-precision idea, not evaluated.
- A viable study needs an error budget for output probabilities and logical-error estimates, plus accumulation in FP32/FP64 where required.
- On rank 19, packed amplitudes could halve HBM traffic if the workload is truly bandwidth-limited; on rank 10, they could halve LDS footprint and potentially permit more resident workgroups.
- **Hypothesis:** 1.3–2.0× for memory-capacity/bandwidth-limited sweeps if accuracy is acceptable; otherwise unusable.

### 2.13 Register-residency specialization inside the cooperative tier

- Ranks 5–10 are all handled by the same 256-thread cooperative shape, although at rank 5 or 6 the state is tiny.
- Hybrid wave-per-shot or multiple-shots-per-block kernels could hold amplitudes in lane registers or a small LDS partition and process several shots concurrently.
- This could increase occupancy and amortize instruction fetch/bookkeeping.
- **Hypothesis:** 1.2–3× for ranks 5–7; diminishing benefit near rank 10.

### 2.14 Axis permutation and compile-time qubit remapping

- The compiler already minimizes peak rank, but the report does not discuss remapping active axes to optimize physical memory stride within a fixed rank.
- Keeping frequently operated axes in low bit positions can improve locality and simplify index generation. Occasional logical/physical axis swaps may be cheaper than repeatedly sweeping high-stride axes.
- This is an algorithm/compiler optimization, not merely a kernel tweak.
- **Hypothesis:** 5–25% on circuits with skewed gate-axis frequency, subject to remapping overhead.

### 2.15 Constant-pool and instruction-fetch layout

- The interpreter loads a relatively large `GpuInstr` even when an opcode needs only a few fields. The current loop reads `program.instrs[pc]` as a structure.
- Hot fields could be split into compact opcode/axis streams, with cold payloads referenced only for U2/U4/noise/measurement operations.
- This may improve scalar-cache and instruction/data-cache utilization without per-op state serialization.
- **Hypothesis:** 3–15% for deep rank-0/1 circuits where bytecode fetch is a meaningful fraction.

### 2.16 Matrix-coefficient broadcast specialization

- The source has `__shfl` broadcast mentioned in the results but the reviewed kernel still reads matrix entries through pointers in U2/U4 sweeps.
- A complete study should compare scalar loads, one-lane load plus shuffle, preloading coefficients into SGPRs, and constant-memory placement.
- **Hypothesis:** 2–8% on U4-heavy circuits if repeated matrix loads are currently significant.

### 2.17 Occupancy versus ILP sweep

- The report treats occupancy mostly as a VGPR threshold problem. It does not benchmark controlled variants at equal semantics with different unroll factors, launch bounds, workgroup sizes, and worker counts.
- A useful experiment would plot throughput against measured active waves, VGPRs, VMEM stalls, VALU utilization, and instruction count.
- This is necessary because the LUT regression suggests scheduler/ILP effects even when the simplified occupancy number remains four waves/SIMD.

### 2.18 AMDGPU-specific intrinsics and ISA validation

- The report proposes hardware-specific operations but does not show a systematic disassembly-driven process.
- Candidate targets include DPP moves, `ds_bpermute`, buffer loads/stores, cache-control modifiers, scalar loads for uniform operands, and explicit packed 64/128-bit loads.
- Each intrinsic should be accepted only if the generated ISA and counters improve; otherwise it risks constraining LLVM unnecessarily.

## 3. Top 5 Suggestions for Next Round

### 1. Build an axis-specialized, incrementally indexed global/cooperative sweep family

- Replace generic `scatter_bits_1/2` in the hottest U2/U4/H/phase loops with specialized low-axis, high-axis, and incremental-index variants.
- Add modest two-butterfly software pipelining where VGPR headroom permits.
- **Estimated gain:** 5–20% on affected rank-10/19 sweeps; likely 2–10% end-to-end depending on opcode mix.
- **Justification:** This attacks the measured SALU-heavy address generation at rank 10 and may improve coalescing at rank 19 without introducing the LDS barriers that caused F8.

### 2. Establish a real memory roofline for rank 19, then tune worker count and cache policy

- Collect HBM bytes/read/write throughput, L2 hit rate, cache-line utilization, VMEM stalls, and memory-channel balance.
- Sweep global worker blocks from below one block/CU through the current two blocks/CU and beyond.
- **Estimated gain:** 0–15%; the immediate value is eliminating uncertainty about whether rank 19 is bandwidth-, latency-, cache-, or synchronization-limited.
- **Justification:** The current 86%-of-HBM argument is dimensionally inconsistent, so optimization priorities at rank 19 are not yet evidence-based.

### 3. Implement wave-per-shot or multi-shot-per-block kernels for ranks 5–7

- Treat the cooperative tier as multiple subtiers rather than one 256-thread design.
- Keep small amplitude arrays in lane registers or compact LDS slices and run several independent shots per workgroup.
- **Estimated gain:** 1.3–3× for ranks 5–7; minimal effect at rank 10.
- **Justification:** Peak rank drives the largest performance discontinuity, and the current tier boundary leaves a broad middle range using a kernel shape designed for rank 10.

### 4. Reduce the remaining synchronization surface, not just the reduction primitive

- Audit every `__syncthreads()` in `execute_shot_coop` and its handlers.
- Fuse frame-state updates into adjacent operations, use narrower wave-level synchronization where valid, and test a wave-specialized control/bookkeeping design.
- **Estimated gain:** 5–20% on measurement/frame-heavy cooperative circuits.
- **Justification:** The strongest measured win came from synchronization reduction, and the current code still has two barriers in `coop_reduce2` plus frequent handler-level barriers.

### 5. Create an accuracy-qualified mixed-precision rank-19 path

- Store amplitudes in FP16 or BF16, compute key operations in FP32, and retain FP64 accumulation for probabilities where needed.
- Validate total variation distance, logical-error estimates, postselection rates, and reproducibility over representative circuits.
- **Estimated gain:** 1.3–2.0× if HBM traffic is dominant and accuracy passes; zero if accuracy requirements reject it.
- **Justification:** This is one of the few ideas capable of changing bytes moved per amplitude rather than shaving dispatch overhead.

## 4. Hardware Counter Gaps

The existing counter set—VGPR, SGPR, LDS, scratch allocation, wave count, SALU/VALU instruction counts, and `SQ_WAIT`—is not enough to prove the report’s bottleneck claims.

- **HBM read/write bytes and achieved bandwidth:** Required to substantiate the rank-19 bandwidth-ceiling claim and to distinguish reads from writeback traffic.

- **Per-memory-channel utilization and imbalance:** Important for power-of-two butterfly strides and the claimed NUMA/XCD effects.

- **L2 request count, hit rate, miss rate, and writeback traffic:** Needed to validate or reject the “prefetcher handles it” and “4 MiB array completely thrashes L2” claims.

- **L1/vector-cache hit rate and bypass behavior:** Especially useful for matrix constants, instruction operands, and repeated measurement passes.

- **Scalar cache and instruction-cache utilization/miss rate:** The interpreter fetches common bytecode across waves; this would quantify whether instruction/data fetch matters and whether compact bytecode could help.

- **VMEM issue rate, outstanding requests, and memory-dependency stalls:** Distinguishes bandwidth saturation from insufficient memory-level parallelism or long dependency chains.

- **LDS bank conflicts and LDS throughput:** `load_complex64`, reduction buffers, and scatter LUTs can alter bank behavior even when total LDS bytes decrease.

- **Barrier count and barrier stall cycles:** The central success claim is synchronization reduction, yet no direct barrier-stall metric is shown.

- **Wavefront occupancy over time, not only static waves/SIMD:** Needed because early exits, postselection, long thread-0 sections, and varying active rank can reduce effective occupancy.

- **Eligible waves / issue occupancy / SIMD utilization:** Clarifies whether more resident waves would actually improve issue utilization.

- **VALU and SALU busy cycles or utilization:** Instruction counts alone do not show which pipeline limits elapsed time.

- **Branch/control-flow counters:** Taken cautiously, these could quantify divergent branches, uniform branches, and instruction-fetch disruption instead of attributing results to an unspecified branch predictor.

- **Scratch read/write transactions and bytes:** Necessary to evaluate `ShotState`, `meas[]` reduction, and per-thread local arrays. Allocation size alone is insufficient.

- **Register spill/fill instructions:** Particularly important for launch-bounds experiments and ILP/unroll variants.

- **Atomic request rate, contention, and latency:** Needed to quantify the +1.4% per-XCD counter result and determine whether output-count atomics matter.

- **Per-XCD traffic/locality counters:** Required before claiming that the hardware XCC-ID work distribution produces local-HBM or local-L2 behavior. The current allocator is still a single device allocation.

- **Power/clock/thermal telemetry:** Benchmark variation of 1.5–8.5× is too large to ignore. Clocks, throttling, and concurrent activity should be recorded with every run.

- **Kernel launch and graph overhead measured separately:** Needed for fair evaluation of per-op and HipGraph designs.

- **Instruction mix below SALU/VALU totals:** Counts of VMEM, LDS/DS, branch, shuffle/DPP, FMA, conversion, and transcendental instructions would make optimization hypotheses much more precise.

## 5. Report Quality Issues

- **There is no single canonical benchmark dataset.** The report includes early comparisons, GEAK comparisons, “final definitive” comparisons, post-iteration comparisons, all-workspace comparisons, and another final benchmark. Results change materially across them. Select one canonical table and move historical tables to a clearly dated appendix.

- **Baseline names are overloaded.** “SVM baseline,” “SVM+GEAK,” “GEAK-only,” “full stack,” “clifft-amd,” and branch names are compared without a consistent lineage. Include a version matrix listing exact commits and optimizations in each binary.

- **Measured, estimated, and expected numbers are mixed.** Examples include 3–5× split gains, 45–70× MFMA slowdown, 5–10% NUMA expectations, and “pending” optimizations appearing in an applied priority table. Use explicit labels: `Measured`, `Analytical estimate`, `Hypothesis`, or `Not benchmarked`.

- **The report overstates causality.** Negative phase-sorting results do not prove a branch predictor explanation; failed LDS tiling does not prove good L2 prefetching; implementation convergence does not prove HBM saturation.

- **The hardware section needs primary citations and definitions.** In particular, clarify XCD/XCC terminology, cache hierarchy, VGPR accounting, maximum resident waves, and the meaning of the “1216 / 512” row.

- **Resource tables conflict.** The same or similar kernels are listed with 76, 84, 108, 116, 124, and 128 VGPRs and different LDS/scratch values. That is plausible across builds, but every row needs a build identifier.

- **The reduction description should match the code.** Correct “one barrier” to two, and correct “called twice” to “reduces two accumulators in one call.”

- **The current source status is not reconciled with the narrative.** The scatter LUT is described as reverted, but it remains in the reviewed kernel. Similarly, `load_complex64` and NUMA logic appear in source while some tables call them pending.

- **The per-op analysis needs a concrete state-layout diagram.** A claimed 2.4 KB round trip per operation is central to the argument but is not derived from the actual branch implementation.

- **Correctness methodology is too thin.** One identical `passed_shots` count is useful but insufficient for mixed precision, RNG changes, split execution, or reordered operations. Report deterministic seeds, observable distributions, expectation-value errors, and statistical confidence.

- **Benchmark uncertainty is missing.** Provide repetitions, median, min/max or confidence intervals, warmup count, and outlier policy. Several tables differ by amounts small enough to be noise.

- **The report needs opcode and active-rank profiles for each representative circuit.** Claims such as “measurement operations are ~40%” and “600 sweeps” should be backed by exact dynamic counts, not only static instruction counts.

- **The phrase “compiler already does memory coalescing” is conceptually imprecise.** Coalescing is primarily a consequence of lane address patterns and hardware transaction formation; the compiler can transform addressing but cannot automatically fix an unfavorable algorithmic mapping in general.

- **The final recommendations are too narrow.** They list only eight ideas and omit several high-value compiler/algorithm changes such as axis remapping, rank-subtier kernels, bytecode compression, and controlled precision.

## 6. Ideas for Future Work

All gains below are hypotheses unless marked as measured from the reviewed sources.

1. **Rank-subtier kernel family (5–7, 8–10, 11–14, 15–19).**
   - Use wave-per-shot or multiple shots per block at low cooperative ranks, conventional block-per-shot at rank 10, and separately tuned global kernels above rank 10.
   - **Expected impact:** 1.3–3× at ranks 5–7; 5–20% elsewhere.

2. **Axis-frequency-aware logical-to-physical qubit remapping.**
   - Keep frequently swept axes in positions that minimize scatter cost and improve locality; account for remapping/swap overhead in the compiler.
   - **Expected impact:** 5–25% on circuits with nonuniform axis use.

3. **Incremental scatter/index generators.**
   - Replace per-iteration bit insertion with run-based or carry-based index updates.
   - **Expected impact:** 2–10% on SALU-heavy rank-5–10 sweeps.

4. **Two-butterfly software pipeline.**
   - Load the next independent butterfly before finishing arithmetic/stores for the current one; tune unroll factor by gate and rank.
   - **Expected impact:** 3–12%, with possible regressions if VGPR pressure is excessive.

5. **BUFFER instruction experiment.**
   - Force or encourage buffer-addressed global loads/stores and compare against generated flat-memory instructions using disassembly and counters.
   - **Expected impact:** 0–8%.

6. **DPP versus shuffle versus `ds_bpermute` reduction bake-off.**
   - Implement equivalent full-wave reductions, verify cross-row behavior, and compare instruction counts and latency.
   - **Expected impact:** 0–5% end-to-end on measurement-heavy cooperative circuits.

7. **Barrier-light wave specialization.**
   - Dedicate one wave to control, RNG, frame, and measurement decisions while worker waves process arrays; communicate through explicit flags.
   - **Expected impact:** 5–20%; high implementation and correctness risk.

8. **Mixed FP16/BF16 storage with FP32 compute and FP64 probability accumulation.**
   - Add a rigorous numerical qualification suite before enabling.
   - **Expected impact:** 1.3–2.0× at bandwidth/capacity-limited ranks if accuracy permits.

9. **Block-floating or per-tile scaled half precision.**
   - Store half-precision mantissas with a shared scale per tile to improve dynamic range over naive FP16.
   - **Expected impact:** 1.2–1.8× at high ranks if conversion overhead remains small.

10. **Compact structure-of-arrays bytecode.**
    - Separate hot opcode/axis/flag streams from cold payloads and constant-pool references.
    - **Expected impact:** 3–15% on deep rank-0/1 interpreter workloads.

11. **Trace/superinstruction compilation.**
    - Fuse common opcode sequences—frame runs, expand–gate–measure motifs, and repeated stabilizer rounds—without compiling the entire circuit.
    - **Expected impact:** 5–25% on repetitive QEC circuits, with lower code-size and compile-time cost than full megakernels.

12. **Dynamic-rank-aware execution within one shot.**
    - Switch kernel strategy when `active_k` crosses thresholds rather than selecting only by global `peak_rank`; preserve state in a compact transferable format.
    - **Expected impact:** 1.2–3× on genuinely mixed-rank circuits; negligible on sustained-rank cultivation_d5.

13. **Reduce global-worker concurrency to improve cache locality.**
    - Sweep blocks/CU and slots/XCD while measuring HBM and L2 counters.
    - **Expected impact:** -10% to +15%; primarily an empirical tuning opportunity.

14. **Measurement fusion.**
    - Where legal, combine probability accumulation, branch selection, collapse/fold, and frame update into fewer passes and synchronization points.
    - **Expected impact:** 5–20% on measurement-heavy cooperative circuits.

15. **Parallel cooperative expectation values.**
    - `coop_exec_exp_val` currently performs the full amplitude loop on thread 0. Parallelize numerator/denominator accumulation across the block and reuse `coop_reduce2` or a three-value reduction.
    - **Expected impact:** potentially large for workloads using expectation values; near zero for circuits that never execute `OP_EXP_VAL`.

16. **Specialize multi-controlled gates by mask density.**
    - Use branchless parity/predication for dense masks and sparse index enumeration for low-popcount masks.
    - **Expected impact:** 2–15% on circuits rich in `OP_ARRAY_MULTI_CNOT/CZ`.

17. **Precompute per-op index plans in compact global/constant memory.**
    - Unlike the failed full LDS LUT, store small run descriptors or axis-specific masks shared across all shots.
    - **Expected impact:** 2–10% if descriptor fetches are cache-resident and reduce SALU without increasing VGPRs materially.

18. **Triton prototype for sweep kernels.**
    - Use Triton primarily as a rapid search vehicle for block sizes, vector widths, and memory layouts; verify whether AMD lowering emits competitive ISA.
    - **Expected impact:** uncertain (0–15%); high value for design-space exploration even if production remains HIP.

19. **IREE/MLIR generated kernel variants.**
    - Express rank/axis/gate shape as compile-time attributes and auto-generate specialized HIP/ROCDL variants.
    - **Expected impact:** 0–20% plus reduced maintenance cost if specialization is automated successfully.

20. **Composable Kernel or custom CK-style templates for butterfly primitives.**
    - Reuse CK’s tuning machinery for vector width, thread clustering, and pipeline depth, while recognizing that the operation is not GEMM.
    - **Expected impact:** uncertain (0–15%); useful as an autotuning framework rather than an MFMA path.

21. **Disassembly-guided compiler intrinsic library.**
    - Wrap DPP, DS permute, buffer loads, cache controls, packed loads, and scalar broadcasts behind tested helpers with ISA checks per gfx target.
    - **Expected impact:** cumulative 2–10% if several small wins compose.

22. **Multi-GPU shot-level scaling.**
    - Partition independent shots across GPUs, aggregate only counters/results, and overlap host aggregation with device execution.
    - **Expected impact:** near-linear throughput scaling until host orchestration, power, or I/O limits; this is the lowest-risk multi-GPU mode.

23. **Multi-GPU state partitioning for ranks beyond local capacity.**
    - Partition amplitude indices across GPUs and use XGMI collectives for gates whose target bit crosses the partition.
    - **Expected impact:** enables larger ranks rather than guaranteeing speedup; communication-heavy gates may scale poorly.

24. **Circuit-batch execution.**
    - Run multiple structurally similar circuits together so they share compiled code and improve occupancy when individual shot counts are small.
    - **Expected impact:** 1.2–2× for small-batch workloads; little effect for already massive shot counts.

25. **Profile-guided code generation.**
    - Feed dynamic opcode counts, axis histograms, active-rank timelines, discard rates, and mask densities into kernel selection and unroll decisions.
    - **Expected impact:** 5–20% across a heterogeneous workload suite.

26. **Statistical early stopping at the algorithm level.**
    - Stop sampling once the desired confidence interval for logical error or observables is reached rather than using a fixed shot count.
    - **Expected impact:** workload-dependent, potentially orders of magnitude in total time-to-answer when the requested statistical precision is modest.

27. **Variance reduction / importance sampling for rare logical errors.**
    - If mathematically compatible with the simulator’s objectives, bias sampling toward rare failure events and reweight estimates.
    - **Expected impact:** potentially much larger than kernel-level gains for rare-event estimation, but requires rigorous unbiasedness and variance analysis.

28. **Compile repeated stabilizer rounds as parameterized loops.**
    - Preserve instruction-cache locality and reduce code size compared with fully unrolled megakernels while eliminating interpreter dispatch inside repeated rounds.
    - **Expected impact:** 5–20% on repetitive surface-code circuits.

29. **RNG/control overlap.**
    - Pre-generate or pipeline random values for known upcoming noise/measurement operations, avoiding serial RNG latency on thread 0 where safe.
    - **Expected impact:** 2–10% on noise- and measurement-heavy circuits.

30. **Output aggregation redesign.**
    - Replace per-shot global atomics in cooperative kernels with per-block or per-XCD aggregation followed by a final reduction.
    - **Expected impact:** 1–10%, larger when many observables are enabled or shots complete quickly.

The most important next step is not another broad architectural rewrite. It is a controlled, counter-driven campaign around three unresolved questions: the true rank-19 memory roofline, the cost of generic scatter/index generation, and how much synchronization remains removable after the successful shuffle reduction.
