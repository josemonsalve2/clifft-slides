## 1. scalar-pipe-deletion -- How specialization removes the interpreter's scalar dependency chain
**Kind:** microarchitecture
**Report section:** §7.3 and §14.3 (lines 1990–2067, 5254–5321)
**What it explains:** Show one wavefront executing a frame op through the interpreter path: `pc * 40` address generation, instruction load, `s_waitcnt`, `v_readfirstlane`, dynamic frame-word address, second dependent load, then switch dispatch. Beside it, show the specialized path issuing literal masks directly on SALU, with no bytecode fetch or cross-pipe operand transfer. This is an EXTENSION of `svm-interpreter.svg` and `spec-classes-gains.svg`: those show the interpreter and the aggregate gains, but not the dependency chain that makes SALU the decisive counter.
**Why prose is not enough:** The key result is causal, not merely numerical: two dependent memory operations and a VGPR-to-SGPR transfer disappear before useful arithmetic begins. A pipe-level before/after makes clear why both VALU and SALU fall, rather than work moving from one to the other.
**DATA:** S1 has `136→47` total instructions, `44→8` VALU, `74→35` SALU, `8→3` VGPR, and `15` `s_waitcnt`s in the interpreter form against `2` in the specialized form. Across the corpus, the SALU ratio is below the VALU ratio on `26 of 26` circuits; the SALU range is `0.110–0.361`, or `2.8× to 9.1× fewer scalar instructions`.
**Strength:** strong — This is the report's primary end-to-end performance mechanism and is not yet illustrated at the GPU issue/dependency level.

## 2. rank22-register-spill-cliff -- The allocator cliff where rank 22 turns registers into HBM traffic
**Kind:** microarchitecture
**Report section:** §10.6 and §14.5 (lines 4056–4189, 5380–5426)
**What it explains:** Draw rank 20–24 as an allocator state transition: ranks 20–21 fit below the VGPR/AGPR caps, while rank 22 jumps to the caps and spills both vector and scalar state into HBM-backed scratch. Connect that extra load/store traffic to a bandwidth-bound global kernel and to the collapse in V2's speedup. This complements `hbm-budget-pool.svg`, which explains resident-pool capacity but not the independent register-allocation mechanism.
**Why prose is not enough:** The report contains two interacting cliffs at the same ranks, and readers can easily conflate pool shrinkage with register spilling. A register-file-to-scratch diagram separates the mechanisms and shows why a shorter generated program can spill more.
**DATA:** Rank 20 `qv20_seed42`: `.vgpr_count 104`, `.agpr_count 40`, `.vgpr_spill_count 0`, `.sgpr_spill_count 0`, scratch `96`, ratio `0.850`. Rank 20 `qv20_L8`: `106`, `42`, `0`, `0`, scratch `112`, ratio `0.732`. Rank 21: `104`, `40`, `0`, `0`, scratch `112`, ratio `0.736`. Rank 22: `128`, `64`, `136`, `762`, scratch `448`, ratio `0.980`. Rank 23: `128`, `64`, `199`, `662`, scratch `576`, ratio `0.990`. Rank 24: `128`, `64`, `152`, `594`, scratch `480`, ratio `0.882`. The three spilling kernels contain only `320–359 instructions`, while a rank-14 kernel contains `16,521 instructions` and spills `4 SGPRs`.
**Strength:** strong — It directly explains why the measured advantage disappears at high rank and targets the report's top open item.

## 3. specialization-cache-identity -- How a missing header hash served stale kernels and stale gate verdicts
**Kind:** runtime-concept
**Report section:** §6.4 and §11.4 (lines 1472–1489, 4437–4516)
**What it explains:** Show the cache identity as a derivation graph from generated C, compiler path, architecture, bitcode directory, and device-header contents to one hash, then to the paired `.hsaco` and `.gate` artifacts. Contrast the broken key, which omitted the included headers, with the fixed content-derived key and trace how one header edit should invalidate both artifacts. No existing diagram covers cache-key derivation or invalidation.
**Why prose is not enough:** The bug arose because the visible generated file was only half the translation unit, while the semantically dominant headers were invisible to the key. A dependency graph makes the omission and its propagation to both code and correctness verdict immediate.
**DATA:** The stale binary had `1400 / 1509 (92.8 %)` barriers with `ds_*` in flight and no `lgkmcnt(0)`, and only `49 / 1509 (3.2 %)` barriers preceded by `lgkmcnt(0)` within 3 instructions. The fresh binary had `0 / 1509 (0.0 %)` unfenced in-flight sites and `1404 / 1509 (93.0 %)` preceded by the wait. The stale binary contained `1e-18` (`0x3c32725d…`); the fresh binary contained `1e-11` (`0x3da5fd7f…`). `32 of the 36` stale cached kernels were dated `2026-07-25`; the remaining `4` were from `07-26 00:00–01:48`.
**Strength:** strong — It is a subtle runtime correctness and benchmarking failure with a clean visual dependency mechanism and exact binary evidence.

## 4. correctness-gate-lifecycle -- Compile once, validate once, then select specialization or fallback
**Kind:** runtime-concept
**Report section:** §6.5 and §9.1 (lines 1491–1580, 3401–3432)
**What it explains:** Diagram the full gate state machine: emit and compile `.hsaco`, run interpreter and specialized validation dispatches, compare byte-exact outputs, persist `<hsaco>.gate`, and select `clifft_v2_spec` or the interpreter on later runs. Include the in-process map, disk verdict, and binary as three distinct caches, plus the profiler-pollution failure when validation ran inside every traced process. No current figure explains how correctness gating changes the runtime dispatch path.
**Why prose is not enough:** The gate is simultaneously a compiler safeguard, a cache, and a source of measurement contamination. A state diagram clarifies which work happens once, which happens on cache hits, and why a correctness mechanism appeared as a performance regression.
**DATA:** Current cache: `16 of 16 pass`; stale cache: `36` verdicts with `2` failures. In the polluted run, `frame_h` dispatched `clifft_v2_register x1 (19.1us) + clifft_v2_spec x2 (28.8us)` and `qv10` dispatched `clifft_v2_coop x1 (1270.2us) + clifft_v2_spec x2 (1667.2us)`. The apparent ratios moved `frame_h 0.612 → 2.859`, `circuit_d3 1.116 → 2.117`, and `qv10 0.252 → 0.675`.
**Strength:** strong — This is a central runtime concept whose interaction with profiling caused one of the report's most misleading intermediate results.

## 5. hsa-aql-dispatch-path -- What is reused, allocated, written, and rung on each HSA dispatch
**Kind:** runtime-concept
**Report section:** §13.1–§13.4 (lines 4843–5014)
**What it explains:** Extend `dispatch-latency.svg` with a mechanism panel showing the AQL queue packet, kernarg segment, completion signal, write index, doorbell, and wait. Compare naive HSA, persistent HSA, batched HSA, and HIP by highlighting which resources are recreated per dispatch and which are retained. The existing figure shows timing bars but not the packet/resource path that produces them.
**Why prose is not enough:** The `31×` naive-path penalty is impossible to understand from a latency bar alone. Readers need to see that memory allocation, GPU access authorization, and signal creation sit on the hot path in one mode and outside it in another.
**DATA:** HSA `naive` steady state `197,935 ns`; HSA `persistent` `6,326 ns`; HSA `batched16` `2,320 ns`; HIP `sync` `10,737 ns`; HIP `stream_sync` `10,997 ns`; HIP `launch_only` `2,093 ns`. Persistent HSA is `1.74× faster` than HIP `stream_sync` (`−4,671 ns`); batched HSA is `4.74× faster` (`−8,677 ns`). The benchmark's kernarg segment is `40` bytes.
**Strength:** strong — It converts a runtime microbenchmark into an explanatory dispatch-path figure and directly satisfies the request for queue/packet structure.

## 6. sroa-to-packed-f32 -- How register promotion unlocks packed vector arithmetic
**Kind:** microarchitecture
**Report section:** §8.5 and §8.6 (lines 2985–3111, 3282–3295)
**What it explains:** Show the compiler transformation from `addrspace(5)` stack slots and escaped locals, through SROA/mem2reg and inlining, to `<2 x float>` operations and final `v_pk_add_f32` / `v_pk_mul_f32` instructions. Preserve the three aggregate allocas that cannot be promoted and show them as the fixed scratch residue. This is an EXTENSION of `lowering-pipelines.svg`, adding the hardware consequence of the `-O0 → -O2` collapse rather than another stage-size chart.
**Why prose is not enough:** The surprising result is that `fmul` and `shufflevector` counts rise while the program becomes faster. A transformation flow makes clear that more optimized IR arithmetic corresponds to fewer packed machine instructions and less memory-resident state.
**DATA:** `alloca 2,271 → 3`; `addrspacecast 2,262 → 8`; `call 1,770 → 72`; `load 4,898 → 1,014`; `store 2,488 → 1,103`; `fmul 12 → 448`; `shufflevector 0 → 459`. Final ISA contains `v_pk_add_f32 431` and `v_pk_mul_f32 416`. V1 has `0` packed f32 ops; V2 has `847`.
**Strength:** strong — It explains a real VALU throughput improvement at the compiler-to-microarchitecture boundary that no existing figure shows.

## 7. branch-erasure-exec-mask -- From eighteen divergent branches to straight scalar selection
**Kind:** microarchitecture
**Report section:** §7.4 (lines 2071–2162)
**What it explains:** Show dormant measurement control flow first with runtime flags, bounds checks, and divergent branch/exec-mask handling, then with constant flags and slots deleting paths and converting the random outcome to `s_cselect_b32`. Include the tradeoff that fewer scalar instructions can require more SGPRs because folded literals stay live. This extends `spec-classes-gains.svg`, whose S2 bar does not explain the control-flow mechanism.
**Why prose is not enough:** Branch count alone hides the GPU cost: divergent scalar branches manipulate `exec` and serialize paths. The picture should distinguish branch deletion, if-conversion, and the separate SGPR-pressure cost.
**DATA:** Branches `18→0`; total instructions `206→70`; VALU `86→19`; SALU `93→45`; VGPR `25→14`; SGPR `13→26`; `s_load 9→4`.
**Strength:** strong — It is the clearest step-by-step example of constant operands changing wavefront control flow, not merely reducing instruction count.

## 8. global-shot-lifecycle -- One persistent workgroup's complete work-stealing loop
**Kind:** runtime-concept
**Report section:** §6.7 (lines 1593–1708)
**What it explains:** Extend `persistent-kernel.svg` with the current V2 implementation rather than the legacy phase-sorted concept: workgroup slot selection, per-slot HBM amplitude and scratch slices, tid-0 atomic shot claim, LDS broadcast, fenced barrier, `spec_body`, completion barrier, and the next claim. Explicitly show that the grid is a resident pool independent of shot count and that the same slot reuses its memory for successive shots.
**Why prose is not enough:** The persistent loop interleaves three address spaces and two kinds of identity (`slot` versus `shot_id`). A lifecycle diagram prevents the common mistaken reading that one global-tier workgroup is launched per shot.
**DATA:** The example wrapper bakes `const u64 amp_capacity = 8192ull`. Each workgroup owns `12 bytes per amplitude`; the HBM budget is `32 GB`; the pool is capped at `2,048` workgroups and aligned to `kNumXCDs = 8`. Rank 21 has `1,360` resident workgroups; rank 22 `680`; rank 23 `336`; rank 24 `168`; rank 25 `80`; rank 26 `40`.
**Strength:** strong — It is the missing runtime-concept view of the production global kernel and corrects the emphasis of the existing legacy figure.

## 9. gfx950-lds-occupancy -- LDS reclamation, the corrected occupancy model, and what the optimization really bought
**Kind:** microarchitecture
**Report section:** §9.3 (lines 3515–3608)
**What it explains:** Show the coop LDS layout before P1, after reduction-buffer right-sizing, and after measurement bit-packing, then place each footprint against both gfx942's and gfx950's occupancy ceilings. Emphasize that gfx950 reaches occupancy 8 after P1a and remains there after P1b, so the measured footprint reduction is real while the originally claimed `2 → 4 wg/CU` mechanism belongs to the previous architecture. This extends `memory-hierarchy-tiers.svg` with actual allocation anatomy and corrected chip-specific occupancy.
**Why prose is not enough:** Two valid measurements and one invalid inference are intertwined in the text. A footprint-to-occupancy diagram makes the architecture mismatch obvious and preserves the useful headroom result.
**DATA:** LDS footprint `25,088 → 16,896 → 13,312 bytes`. `lds_red0` and `lds_red1` shrink `[256] → [8]`; `lds_red_scratch` shrinks `[1024] → [512]`; `u8 lds_meas[4096]` (`4 KB`) becomes `u64 lds_meas[64]` (`512 B`). gfx942 occupancy is `2 → 3 → 4`; gfx950 occupancy is `6 → 8 → 8`. gfx950 has a `163840`-byte LDS limit, and occupancy first drops below 8 at `20,992 bytes`.
**Strength:** strong — It is exactly the kind of occupancy mechanism audit the author requested, with a valuable correction rather than a restated table.

## 10. v1-register-pressure-cascade -- How monolithic IR disables optimization and ends in spills
**Kind:** microarchitecture
**Report section:** §5.3 and §5.5 (lines 895–935, 1099–1165)
**What it explains:** Trace V1's failure from per-instruction block expansion to multi-megabyte IR, size-triggered optimizer detuning, a giant basic-block chain, VGPR saturation, reduced occupancy, and finally scratch spills. Show separately that the famous `115,224 B` private segment belongs to the global tier built with `llc -O0`, while the register tier's failure mode is VGPR saturation at 128. This complements `ir-density.svg` and `v1-mlir-reality.svg`, which stop before the register file.
**Why prose is not enough:** The report corrects a long-lived misattribution, and a cascade diagram can show that IR bloat, optimizer shutdown, and spilling are one failure expressed at three stages. It also prevents readers from repeating the false “115 KB/thread register-tier” claim.
**DATA:** `surface_d11_t5` produces `19,856,511 bytes of LLVM IR`; `glob_surface_d7_t19` compiles in `221.48 s`. Optimizer thresholds are `4 MB` and `16 MB`. Maximum recorded private segments: register `4,192 B`, coop `608 B`, global `115,224 B`. `reg_circuit_d3` uses `128 VGPR`, occupancy `4`, and `156 B` scratch; small register circuits use `16 VGPR`, occupancy `8`, and `0 B` scratch. The global `115,224 B` case came from `9,967,049 bytes of IR` with `llc -O0` and `opt` skipped.
**Strength:** strong — It adds the missing hardware mechanism to the report's central V1 failure analysis.

## 11. register-tier-topology -- Why 256 threads per rank-0 shot was 28× slower
**Kind:** microarchitecture
**Report section:** §9.1–§9.2 (lines 3356–3376, 3461–3511)
**What it explains:** Show a rank-0 shot mapped first onto a full 256-thread workgroup with owner guards and one barrier after each operation, then onto the shot-packed topology where every thread owns a complete shot, barriers compile away, and the reduction degenerates to an assignment. This is an EXTENSION of `memory-hierarchy-tiers.svg` and `optimization-timeline.svg`: the former states the final topology and the latter shows the performance step, but neither shows wasted lanes and barrier issue.
**Why prose is not enough:** “Wrong topology” is abstract until the reader sees 255 lanes participating in synchronization around one lane's scalar work. The visual would make the magnitude of the P0 step credible.
**DATA:** `frame_h` moves from V2/SVM `28.258` at baseline to `1.215` after P0+P1. It is rank `0`, has `four instructions`, and the original topology used `256 threads` per shot with a barrier after each instruction. The commit measurement states `frame_h 30.6x→1.14x`, `circuit_d3 14.1x→1.13x`, and byte-exactness `38/38`.
**Strength:** strong — It explains the largest single optimization in the history with a concrete wave/workgroup utilization mechanism.

## 12. noise-loop-cannot-fold -- Why the most expensive opcode class barely benefits from specialization
**Kind:** microarchitecture
**Report section:** §7.9 and §14.4 (lines 2551–2649, 5323–5338)
**What it explains:** Extend `spec-classes-gains.svg` with a control/data-dependence view of `OP_NOISE_BLOCK`: `next_noise` selects a site, `ocml_log_f64` advances it by a random amount, and the loop trip count therefore cannot be known until the shot runs. Show the only fold—the two-compare range test becoming one unsigned compare—beside the unchanged PRNG, table lookup, and log call. Connect the per-op limit to the d5 instruction mix.
**Why prose is not enough:** The negative result is mechanistic: constants exist, but they are not the values controlling the expensive loop. A dependency diagram makes the boundary of specialization much easier to understand than a nearly-flat bar.
**DATA:** Total instructions `447→407`; VALU `214→210`; SALU `177→141`; branches `23→20`; VGPR `56→56`; SGPR `86→85`; `s_load 9→5`. `circuit_d5` has `1,720` calls, of which noise/readout calls are `329` or `19.1 %`; a noise op has `447` interpreter-form instructions against `136` for a frame op. The d5 family runs at `0.786–0.856`.
**Strength:** strong — It explains why the report's weakest coop family remains weak despite large scalar deletion elsewhere.

## 13. one-library-two-consumers -- Specialization data flow through the shared operand library
**Kind:** runtime-concept
**Report section:** §6.2–§6.3 (lines 1264–1413)
**What it explains:** Draw bytecode and statically tracked rank feeding two call forms into the same `v2_op_*` body: runtime operands for the interpreter, literals for the specializer. Then show tier macros selecting stride, ownership, barrier, and reduction behavior without duplicating arithmetic. This extends `three-loops.svg` by explaining the source-sharing and byte-exactness architecture rather than loop placement.
**Why prose is not enough:** The crucial design claim is “same semantics, different knowledge,” which is naturally a data-flow picture. It also clarifies how V2 can be aggressively specialized without reintroducing V1's duplicated opcode implementations.
**DATA:** The specializer handles `35 of the 41 opcodes`; unsupported opcodes fall back to the interpreter. For `reg_circuit_d3`, `344` bytecode instructions produce exactly `344` `v2_op_*` calls in `383` C lines. V2 source density reaches `1.11`, `1.27`, `1.02`, `1.00`, and `1.01 lines/instr` on the five nontrivial examples. Register cooperation is `V2_STRIDE 1u`; coop/global is `V2_STRIDE 256u`.
**Strength:** strong — This is the report's core architecture and currently lacks a single end-to-end specialization data-flow figure.

## 14. xcd-aligned-pool-underfill -- How rank growth drains work from eight XCDs and eventually leaves CUs idle
**Kind:** microarchitecture
**Report section:** §6.7 and §14.5 (lines 1667–1707, 5340–5378)
**What it explains:** Extend `hbm-budget-pool.svg` and replace the stale MI300X assumptions in `numa-xcd-optimization.svg` with a current gfx950 device map: round the resident pool down to a multiple of eight XCDs, distribute workgroups across the dies, and show the transition from many workgroups per CU to fewer workgroups than CUs. The figure should explain why instruction-count improvements cannot fill the device at rank 24.
**Why prose is not enough:** A descending line chart shows pool size but not spatial underfill across chiplets and CUs. The XCD map makes the hardware consequence of `168` workgroups on `256` CUs immediately visible.
**DATA:** `kNumXCDs = 8`; the pool is capped at `2,048`. Predicted and measured workgroups are rank 20 `2,048`, rank 21 `1,360`, rank 22 `680`, rank 23 `336`, rank 24 `168`. The device has `256 CUs`; at rank 24, `168 workgroups` means “the machine cannot be filled.” Each rank step doubles bytes/workgroup: `12 MB`, `24 MB`, `48 MB`, `96 MB`, `192 MB`.
**Strength:** strong — It directly addresses XCD partitioning and occupancy with current, verified report data rather than legacy architecture artwork.

## 15. kernarg-abi-contract -- Direct HSA dispatch makes struct layout part of correctness
**Kind:** runtime-concept
**Report section:** §9.6 (lines 3786–3830)
**What it explains:** Show host kernarg packing, the AQL packet's raw pointer, and the device kernel's fixed-offset reads, with no runtime type or size check in between. Use the observable-mask fix to show how a new field occupied existing padding without changing total struct size, and place the static-assert layer alongside as the only guardrail. No existing diagram covers this ABI contract.
**Why prose is not enough:** The failure mode is silent byte reinterpretation, not a launch error. An offset diagram makes clear why one four-byte shift can corrupt every later argument and why compile-time layout checks are mandatory under raw HSA.
**DATA:** `sizeof(CV2KernArgs) = 152`; `offsetof num_noise_sites = 112`; `offsetof expected_obs_mask = 116`. The report records `55 static_assert`s. The observed hazard was a `120-vs-116 mismatch`; the opcode checks include `FRAME_CNOT == 0` and `EXPAND == 19`.
**Strength:** strong — This is a subtle runtime concept with a concrete incident and a compact, high-value visual form.

## 16. hybrid-v2-code-object-paths -- From circuit to HSA-loadable ELF, with and without HIP packaging
**Kind:** runtime-concept
**Report section:** §4.2 and §6.4 (lines 683–709, 1415–1450)
**What it explains:** Compare Hybrid's HIP-source path—clang offload bundle, explicit unbundle, raw ELF, HSA load—with V2's direct freestanding C-to-amdgcn bitcode, device-library link, optimization, object, and `ld.lld` path. This is an EXTENSION of `lowering-pipelines.svg`: that figure compares V1 and V2 representation sizes, while this one isolates packaging, device-library linkage, and runtime load boundaries.
**Why prose is not enough:** Both paths ultimately dispatch through HSA, so the phrase “Hybrid uses HIP, V2 does not” is easy to misread as a runtime difference only. A pipeline diagram shows that Hybrid uses HIP as a source and container format, whereas V2 targets the HSA code-object format directly.
**DATA:** qualitative, no numbers.
**Strength:** good — The mechanism is important for understanding the no-HIP policy, though it is less directly tied to the measured kernel speedups than the higher-priority proposals.

## 17. opmix-to-speedup -- Three circuit families, three different limiting mechanisms
**Kind:** conceptual
**Report section:** §7.9, §14.4, and §14.5 (lines 2629–2649, 5323–5344, 5380–5407)
**What it explains:** Build a three-column causal synthesis: surface circuits dominated by foldable frame/dormant-measurement work; d5 circuits dominated in time by data-dependent noise; QV circuits dominated by amplitude arithmetic, shrinking resident pools, and spills. Point each column from instruction mix through the relevant microbenchmark result to the observed end-to-end ratio. This goes beyond `spec-classes-gains.svg` by explaining why whole circuits land in different performance regimes.
**Why prose is not enough:** Readers currently have to mentally join §7's per-op experiments to §14's family results and then remember the independent QV resource cliff. A single synthesis figure would make the report's predictive claim testable at a glance.
**DATA:** Surface family ratios `0.256–0.534`; S1 VALU gain `5.50×`; S2 VALU gain `4.53×` and branches `18→0`. d5 family ratios `0.786–0.856`; noise is `329` calls or `19.1 %`; S7 VALU gain is `1.02×`. QV family ratios `0.732–0.990`; rank-22/23/24 SGPR spills are `762 / 662 / 594`.
**Strength:** strong — It is the best high-level mechanism summary still missing from the report and connects otherwise separated evidence.

## 18. specialized-resource-footprint -- Why knowing the circuit shrinks LDS and scratch allocation
**Kind:** microarchitecture
**Report section:** §14.6 and §15.5 (lines 5455–5484, 5681–5722)
**What it explains:** Compare SVM's worst-case provisioning with V2's circuit-specific resources across register, coop, and global tiers. Show which arrays disappear entirely in the global tier, why register-tier V2 needs less scratch, and how those allocations map to VGPR/LDS/HBM rather than merely listing byte counts. This extends `memory-hierarchy-tiers.svg`, which maps tiers to memory levels but does not compare backend-specific resource footprints.
**Why prose is not enough:** The important distinction is “provision for any opcode” versus “provision for this circuit.” A memory-layout comparison explains why specialization saves storage even when the arithmetic is unchanged.
**DATA:** Coop LDS: V2 `13,312 B`, SVM `23,040 B`, ratio `0.578`. Global LDS: V2 `1,024 B`, SVM `8,704 B`, ratio `0.118`. Register LDS: `0 B` for both. Register scratch: V2 `656–1,040 B`, SVM `4,480 B`, described as `4.3× smaller`. On the register tier V2 executes `zero` LDS instructions against SVM's `25,280`.
**Strength:** good — It explains a broad resource advantage, though the report does not isolate its independent timing contribution.

## 19. d5-three-counter-anomaly -- One unexplained family seen through time, activity, and L2
**Kind:** data
**Report section:** §14.7–§14.8, §15.7, and §16.4 (lines 5486–5516, 5813–5819, 5937–5943)
**What it explains:** Plot the six d5-family circuits as linked residuals: busy-cycle and GRBM activity ratios around `0.62–0.65`, time ratios around `0.79–0.86`, and L2 hit rates around `91 %` against SVM's `98 %`. The figure should explicitly label the mechanism as unknown and show that all three anomalies select the same population. No existing figure covers this open cache/activity result.
**Why prose is not enough:** Three tables contain the evidence, so the exact population match is hard to see. A coordinated residual plot would turn “something is odd” into a precise diagnostic target without pretending the cause is known.
**DATA:** Busy-cycle relative errors for the family are `28.2 %`, `27.2 %`, `26.4 %`, `25.0 %`, `21.2 %`, and `17.9 %`. V2 L2 hit rates are `91.1–92.1 %` for the five `circuit_d5` variants and `90.9 %` for `cultivation_d5`; SVM is `97.6–98.4 %` and `98.9 %`. Time ratios are `0.786–0.856`; both activity counters report approximately `0.62–0.65`.
**Strength:** good — It is the report's largest unexplained result; the figure would guide future profiling, but it cannot yet explain a confirmed mechanism.

## 20. profiler-pass-admissibility -- Which hardware counters may be combined, and which quotient is invalid
**Kind:** conceptual
**Report section:** §14.1, §14.5, and §15.2 (lines 5172–5184, 5428–5440, 5584–5596)
**What it explains:** Draw three separate kernel executions for `pmcA`, `pmcB`, and `pmcC`, with counters attached to their own run. Mark ratios of the same counter between V2 and SVM as valid, same-pass combinations as conditionally valid, and cross-pass quotients such as `SQ_WAVE_CYCLES / SQ_WAVES` as invalid. No current diagram explains the profiling methodology that caused a retracted mechanism claim.
**Why prose is not enough:** Readers naturally treat one counter table as one execution, but the hardware collection model makes that false. A run-separation diagram prevents the exact analytical error the report documents.
**DATA:** There are `three separate profiling passes`. `pmcA` contains `SQ_INSTS_{VALU,SALU,LDS,MFMA}` and `SQ_WAVES`; `pmcB` contains `TCC_HIT_sum` and `TCC_MISS_sum`; `pmcC` contains `GRBM_GUI_ACTIVE`, `SQ_BUSY_CYCLES`, `SQ_WAIT_INST_LDS`, and `SQ_WAVE_CYCLES`. Three identical `512-workgroup`, `256-thread` SVM grids report `8,832`, `10,240`, and `8,000` waves where geometry implies `2,048`.
**Strength:** good — It is a broadly useful methodological figure and explains why one plausible occupancy story had to be removed.

## 21. dust-threshold-calibration -- Choosing a clamp from the fp32 residual floor, not the fp64 constant
**Kind:** conceptual
**Report section:** §12.2–§12.3 (lines 4651–4753)
**What it explains:** Extend `prng-desync.svg` with the calibration step that precedes desynchronization: place fp64 and fp32 residual probabilities on a log axis, show the old `1e-18` threshold below the fp32 floor, and show `1e-11` safely above the conservative tail. Include the rank sweep to show that the floor tightens rather than grows with rank, then the two-arm A/B that changes only the constant.
**Why prose is not enough:** The counterintuitive result is that low rank has the widest tail and one threshold covers ranks 1–26. A scale diagram makes the four-decade miscalibration and the margin of the replacement immediately legible.
**DATA:** Reference case: fp64 `p0 = 2.465e-32`, which clamps at `1e-18`; fp32 `p0 = 8.882e-16`, which does not. Conservative residual model: rank 1 median `9.7e-15`, max `1.2e-13`; rank 4 median `1.3e-14`, max `5.8e-14`; rank 12 median `1.42e-14`, max `1.6e-14`; rank 26 median `1.42e-14`, max `1.5e-14`. New threshold `1e-11`. A/B: d5 changes from `gpu=[0] cpu=[1] MISMATCH` to `gpu=[1] cpu=[1] match`; post-fix same-stream verification is `36/36 exact`.
**Strength:** good — The desynchronization mechanism already has a figure, but the threshold-selection mechanism and its rank dependence do not.

## 22. compiled-rank-census -- The fixture tree collapses into two rank clusters
**Kind:** data
**Report section:** §10.4–§10.5 (lines 3969–4054)
**What it explains:** Show all 353 fixtures as a rank histogram or dot census after `StatevectorSqueezePass`, colored by register, coop, and global tier. Highlight the empty ranks and contrast misleading fixture names with measured compiled rank. This is not a duplicate of `circuit-translation.svg`: that legacy figure sketches one squeeze transformation, while this proposal shows the measured population that determines real tier coverage.
**Why prose is not enough:** The distribution is highly discontinuous, and the table does not convey how overwhelmingly the corpus sits at rank 0–1. A census figure explains why tier benchmarking is effectively a population study rather than a random sample.
**DATA:** Total `353` fixtures. Rank 0: `82`; rank 1: `244`; rank 3: `3`; rank 4: `2`; ranks 0–1 subtotal `326`. Rank 7: `3`; rank 10: `8`; rank 11: `2`; rank 12: `1`; rank 13: `1`; rank 14: `1`; rank 20: `2`; ranks 21, 22, 23, and 24: `1` each; rank ≥5 subtotal `22`. No fixtures compile to ranks `2, 5, 6, 8, 9, 15, 16, 17, 18 or 19`.
**Strength:** good — It makes the report's scope and benchmark representativeness much easier to grasp, though it is primarily contextual rather than a performance mechanism.

## 23. dispatch-cost-relevance -- A fixed 4,671 ns matters only at the short tail
**Kind:** data
**Report section:** §13.4a (lines 5016–5074)
**What it explains:** Extend `dispatch-latency.svg` with a second panel that places the fixed HSA-versus-HIP saving against actual V2 kernel durations on a log scale. Show the same `4,671 ns` block occupying a large fraction of `frame_h` and `four_t`, then disappearing inside millisecond and second kernels. This prevents the dispatch microbenchmark's `1.74×` headline from being mistaken for the source of V2's corpus-wide speedup.
**Why prose is not enough:** Fixed overhead is most intuitive as area relative to total duration, not as percentages in a table. The visual would reconcile “large launch win” with “negligible median effect.”
**DATA:** The fixed saving is `4,671 ns`. `frame_h`: `12.1 µs`, `38.7 %`; `four_t`: `13.1 µs`, `35.7 %`; `circuit_d3_p0.001`: `220.7 µs`, `2.12 %`; `qv10`: `1.386 ms`, `0.337 %`; `circuit_d5_p0.001`: `8.600 ms`, `0.054 %`; `cultivation_d5`: `16.42 ms`, `0.028 %`; `qv24_L4_seed42`: `7.242 s`, `0.00006 %`. The naive HSA path would cost `15–20× the kernel itself` on the two shortest circuits.
**Strength:** good — It is an important qualification to an existing figure, though it explains host overhead rather than the kernel's main performance mechanism.

## 24. butterfly-not-mfma -- Why pairwise amplitude updates do not map to matrix cores
**Kind:** microarchitecture
**Report section:** §2.5 and §14.6 (lines 493–503, 5442–5453)
**What it explains:** Replace the stale MFMA implication in `mi300x-architecture.svg` with a current gfx950 mechanism comparison: a strided 2×2 or 4×4 butterfly over amplitude pairs/quadruples versus the tiled, reusable matrix operands an MFMA instruction expects. Show that each shot follows dependent state updates and reductions rather than accumulating a GEMM tile. The figure should rule out a tempting optimization direction rather than advertise unused hardware.
**Why prose is not enough:** “Not a GEMM” is easy to dismiss as a slogan. Side-by-side dataflow would show the missing reuse and tiling structure that prevents matrix-core mapping.
**DATA:** `SQ_INSTS_MFMA = 0.0` on `all 52 backend×circuit cells` — `26 circuits × 2 backends`, none missing. `ARRAY_U2` applies a `2×2` complex matrix to `2^(k-1)` amplitude pairs; `ARRAY_U4` applies a `4×4` matrix to `2^(k-2)` quadruples.
**Strength:** marginal — It is valuable as a guardrail and corrects a misleading legacy architecture figure, but it explains an absent mechanism rather than a measured speedup.
