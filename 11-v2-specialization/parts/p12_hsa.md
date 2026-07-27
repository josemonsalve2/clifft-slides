## 13. Removing HIP, introducing HSA

V2's dispatch policy is stated as an absolute: **avoid HIP at all cost; HSA-based
dispatch and runtime management.** This section measures what that is worth,
corrects the in-tree justification for it by a factor of ~44, bounds the benefit
honestly against V2's actual workloads, and prices the bill the policy has
already come due for.

The short version, stated up front so the numbers below are read correctly:

- HSA's persistent path beats HIP's synchronized launch by **1.74×** (4,671 ns).
- The pre-existing naive HSA path was **31× worse than either** — the single
  largest dispatch finding here, and the one the in-tree comments got most wrong.
- But V2 issues **one dispatch per run**, so at the median circuit that 1.74×
  is worth **0.03 % of kernel time**. It is decisive only in the short tail.
- The policy's real cost was a correctness bug (§11.2), not a performance one.

### 13.1 What HIP actually does per launch

A HIP kernel launch is not a doorbell write. `hipLaunchKernelGGL` enters the
CLR runtime, which validates arguments, resolves the kernel from a module,
packs the kernarg segment into a runtime-managed pool, builds an AQL packet,
and enqueues it on a stream — and `hipStreamSynchronize` then re-enters the
runtime to wait. Underneath, all of it becomes the same AQL packet on the same
hardware queue that HSA exposes directly.

The question is only what the abstraction costs. Every prior answer in this tree
predates the current hardware and none of them compares the two runtimes.
`hsa_persistent_dispatch.h` documents per-op costs (`alloc_kernarg ~800 ns`,
`allow_gpu_access ~1200 ns`, `signal_create ~600 ns`, "~3.3 µs saved of ~4.5 µs
total") under the label *"measured on MI300X"* — a different chip, a different
ROCm, and no reproducible artifact in the tree. There is no HIP-side number at
all. So the policy that shapes V2's entire host path rested on an unverified
comparison that was never actually run.

### 13.2 The experiment

An empty amdgcn kernel — the smallest thing that can be dispatched — built with
the **exact** flags the V2 specializer uses, so the code object is
representative:

```c
// empty_kernel.c — the smallest possible amdgcn kernel.
//
// The point of an empty kernel is that its GPU execution time is a floor
// (a few hundred ns of wave launch + s_endpgm), so end-to-end launch-to-
// completion latency measured on the host is DOMINATED by the runtime's
// dispatch path. That is exactly the quantity we want to attribute to
// HIP vs raw HSA.
__attribute__((visibility("default")))
__attribute__((amdgpu_kernel))
void bench_empty(unsigned char* out, u64 a, u64 b, u64 c, u64 d) {
    u32 t = __builtin_amdgcn_workitem_id_x();
    if (t == 0u) out[0] = (unsigned char)(a ^ b ^ c ^ d);
}
```

It takes five arguments and writes one byte so the compiler cannot delete the
body and the kernarg segment is non-trivially sized. The benchmark reads the
segment size back out of the loaded code object rather than assuming it, and
prints it in the run header — `kernarg_seg=40` — so the packet being timed is
demonstrably carrying real arguments.

Compiled with the specializer's flags verbatim (`run_bench.sh:28-29`):

```sh
clang --target=amdgcn-amd-amdhsa -mcpu=gfx950 -ffreestanding -nostdlib \
      -nogpulib -std=c23 -O2 -ffp-contract=off -c -emit-llvm -o bench_empty.bc
```

Six modes, three per runtime, all measuring **launch-to-completion wall latency
per dispatch** — the same quantity on both sides, which is what makes them
comparable:

| runtime | mode | what it does |
|---|---|---|
| HSA | `naive` | alloc kernarg + `agents_allow_access` + `signal_create` per dispatch, destroy + free after — what `hsa_dispatch_and_wait` did before `PersistentDispatcher` |
| HSA | `persistent` | kernarg + signal allocated once; packet written per dispatch; signal reset with `store_relaxed` — **V2's hot path** |
| HSA | `batched16` | 16 packets published back-to-back, doorbell rung once, wait once on the last |
| HIP | `sync` | `<<<>>>` + `hipDeviceSynchronize` |
| HIP | `stream_sync` | `<<<>>>` + `hipStreamSynchronize` |
| HIP | `launch_only` | `<<<>>>` only, no sync — HIP's *asynchronous enqueue floor* |

Both runtimes in **one SLURM job on one node** (50507, `smci350-rck-g03-d13-21`,
gfx950, ROCm 7.2.3), 2,000 iterations × 5 reps. The partition is heterogeneous,
so measuring the two arms on different nodes would have been worthless.

### 13.3 The result

ns per dispatch, all five reps:

| mode | rep 0 | rep 1 | rep 2 | rep 3 | rep 4 | steady state |
|---|---|---|---|---|---|---|
| HSA `naive` | 198,920 | 201,622 | 203,587 | 193,931 | 191,615 | **197,935** |
| HSA `persistent` | 6,333 | 6,329 | 6,323 | 6,325 | 6,320 | **6,326** |
| HSA `batched16` | 2,326 | 2,318 | 2,318 | 2,318 | 2,318 | **2,320** |
| HIP `sync` | 13,668 | 10,739 | 10,731 | 10,745 | 10,734 | **10,737** |
| HIP `stream_sync` | 10,993 | 11,002 | 10,992 | 11,000 | 10,997 | **10,997** |
| HIP `launch_only` | 2,106 | 2,088 | 2,090 | 2,085 | 2,095 | **2,093** |

Reproducibility is strong in every mode that matters. Across reps 1–4 the
peak-to-peak spread is **0.15 % for `persistent`, 0.02 % for `batched16`,
0.13 % for HIP `sync` and 0.09 % for `stream_sync`** — the two numbers the
headline rests on are stable to two parts in a thousand. HIP `sync` rep 0
carries a visible first-touch cost (13,668 → 10,739), which is why it is
excluded from its steady-state column.

Two modes are noisier and should be read with that in mind: `launch_only`
spreads 0.44 % and `naive` spreads **6 %** (191.6–203.6 µs). The naive path's
variance is unsurprising — it makes two KFD ioctls and a page-table update per
dispatch, so it inherits kernel-side scheduling jitter. It doesn't matter for
any conclusion drawn here, because the effect being measured is 31×.

**The headline comparison** — same kernel, same node, same completion semantics:

| | ns/dispatch | vs HIP `stream_sync` |
|---|---|---|
| HIP `stream_sync` | 10,997 | 1.00× |
| **HSA `persistent`** (V2's hot path) | **6,326** | **1.74× faster** (−4,671 ns) |
| **HSA `batched16`** (not yet used) | **2,320** | **4.74× faster** (−8,677 ns) |

<figure>
<img src="diagrams/dispatch-latency.svg" alt="Per-dispatch latency, HSA vs HIP, six modes" width="100%">
<figcaption><b>Figure 13.1</b> — Launch-to-completion latency per dispatch on
gfx950 (job 50507). Log scale: the naive HSA path is 31× worse than V2's
persistent path, and batched HSA <em>with</em> a completion wait lands within
11 % of what HIP charges for an <em>unsynchronized</em> enqueue.</figcaption>
</figure>

### 13.4 Reading the result honestly

Three observations, and the third is the one that matters most:

**1. HSA is worth 1.74× on the synchronized path.** 6,326 vs 10,997 ns. That
4,671 ns per dispatch is pure runtime abstraction — identical kernel, identical
hardware queue, identical completion semantics, differing only in who builds the
AQL packet.

**2. Batching is worth more than the runtime choice.** `batched16` at 2,320 ns
is 2.73× better than `persistent` — a larger factor than HSA-vs-HIP's 1.74×.
Amortizing the doorbell and the completion wait across 16 packets beats any
per-dispatch micro-optimization available on either runtime.

The useful comparison is against **HIP's `launch_only` floor of 2,093 ns**, the
cost of enqueueing without ever waiting. Batched HSA reaches 2,320 ns *while
still paying for a completion wait* — 11 % above HIP's fire-and-forget cost, for
strictly stronger semantics. Put the other way: HIP charges nearly as much to
merely hand a packet to a stream as batched HSA charges to dispatch it and
observe it finish.

**3. The in-tree numbers this replaces were wrong by a factor of ~44.**
`hsa_persistent_dispatch.h:36-42` carries a cost model — labelled *"measured on
MI300X"* — that itemizes exactly the operations the `naive` mode performs:

```
/// Eliminated per-dispatch costs (measured on MI300X):
///   - alloc_kernarg:      ~800ns  (pool allocator + bookkeeping)
///   - allow_gpu_access:   ~1200ns (kernel call into KFD for page table update)
///   - signal_create:      ~600ns  (KFD ioctl for doorbell-backed signal)
///   - signal_destroy:     ~400ns  (KFD ioctl)
///   - free_kernarg:       ~300ns  (pool return + bookkeeping)
///   Total saved:          ~3.3us per dispatch (of ~4.5us total overhead)
```

The `naive` mode is a faithful reimplementation of that op sequence, so the two
are directly comparable:

| claim | asserted | measured | error |
|---|---|---|---|
| naive per-dispatch cost | ~4.5 µs total | **~198 µs** | 44× understated |
| saving from persistent resources | ~3.3 µs | **~192 µs** (198,000 − 6,326) | 58× understated |

One precision note in the header's favour: its "~4.5 µs" is labelled *overhead*,
while 198 µs is total launch-to-completion latency. But the two are separated by
only the 6,326 ns `persistent` floor — subtract it and the eliminated cost is
still ~192 µs against an asserted ~3.3 µs. The gap is not an accounting artifact.

The *direction* was right and the *magnitude* was badly wrong. Per-dispatch
`hsa_amd_memory_pool_allocate` + `hsa_amd_agents_allow_access` +
`hsa_signal_create` costs ~198 µs, not ~4.5 µs — `agents_allow_access` in
particular is a KFD call that updates GPU page tables, and paying that per
launch dwarfs everything else in the system. This is the strongest single
argument for `PersistentDispatcher`, and nobody had actually measured it.

It also reframes the priority order. Against a ~10 µs HIP dispatch, a
millisecond-scale kernel is dominated by compute. Against a ~198 µs naive
dispatch, dispatch is a first-order cost for anything short — which is exactly
the regime the v1-era `PERFORMANCE_OPTIONS.md` was written in, and why "host
dispatch overhead" was labelled the dominant bottleneck there.

### 13.4a How much of this does V2 actually collect?

Honesty requires answering the obvious follow-up: V2 does **one dispatch per
run**, not one per shot. The shot loop is inside the kernel (§4's R5 principle),
and `n_dispatches` is `1` for all 26 tier-5+ circuits in the corpus. So the
4,671 ns saving is paid once, and its weight is 4,671 ns divided by the whole
kernel time:

Measured on the canonical run (job 50793), whose `n_dispatches` is **1 for all
26 circuits** — so this is exact, not an assumption:

| circuit | tier | V2 kernel | 4,671 ns as % of it |
|---|---|---:|---:|
| `frame_h` | register | 12.1 µs | **38.7 %** |
| `four_t` | register | 13.1 µs | **35.7 %** |
| `circuit_d3_p0.001` | coop | 220.7 µs | 2.12 % |
| `qv10` | coop | 1.386 ms | 0.337 % |
| `circuit_d5_p0.001` | coop | 8.600 ms | 0.054 % |
| `cultivation_d5` | coop | 16.42 ms | 0.028 % |
| `qv24_L4_seed42` | global | 7.242 s | 0.00006 % |

> **An earlier version of this table was stale in a way worth recording.** It
> quoted `circuit_d5_p0.001` at 14.76 ms and `cultivation_d5` at 30.15 ms from
> the two `f13-21` runs. Those are **interpreter** times: both circuits map to
> `coop_r10_n1720`, whose gate verdict was a stale pre-fence *failure* (§11.4),
> so V2 fell back to the interpreter for exactly that shape. The canonical run
> has the gate passing and the specializer selected, at 8.60 ms and 16.42 ms —
> 1.72× and 1.84× faster. The percentages barely moved (0.032 % → 0.054 %) so no
> conclusion changed, but the absolute times were measuring a different kernel
> than the one the surrounding text describes.

The two register-tier rows carry the effect and deserve a stability note: at
12–13 µs a kernel is close enough to the measurement floor that run-to-run
variation is comparable to the effect. The two earlier `f13-21` runs put
`four_t` at 13.2 and 10.1 µs and `frame_h` at 13.6 and 10.0 µs, spanning
34–47 %. Read the short-tail figure as **~35–47 %**, not as a single number.
Those two circuits are also unaffected by the stale cache in the first place:
they are register tier (LDS = 0, VGPR = 32), and §11.2's A/B rebuild showed the
register-tier binary is **byte-identical by md5** before and after the barrier
fix — 4,457 instructions, zero barriers, nothing to fence. So all three runs
measured the same code for these rows.

The conclusion is unambiguous and cuts against a naive reading of §13.3:
**for V2's production workloads, the HSA-vs-HIP dispatch difference is
negligible.** At the median circuit it is well under 0.1 % of kernel time. The 1.74× is a
real property of the dispatch path and it is not where V2's speedup comes from
— §14 attributes that to the kernel.

Where it *does* matter is the short tail. `four_t` and `frame_h` run for 10–13 µs,
so a single HIP dispatch would add 35–47 % to their cost, and the ~198 µs naive
path would have cost **15–20× the kernel itself** (16.4× and 15.1× on the
canonical run's 12.1 and 13.1 µs). Those two circuits are also
exactly the ones a user iterates on interactively. And the correctness gate
(§9) dispatches per validation, as does any future per-batch structure.

So the defensible framing of the no-HIP policy is not "it makes V2 fast." It is:
*dispatch overhead is a fixed floor that becomes the entire cost at small
problem sizes, and HSA puts that floor 1.74× lower — and 31× lower than the
naive path the code started from.*

### 13.5 What the no-HIP policy has cost

The policy is not free, and §11.2 is the invoice.

`v2_barrier()` was hand-rolled to avoid HIP's `__syncthreads()`. HIP's version
expands to a release/acquire fence pair around `s_barrier`; V2's expanded to a
bare `s_barrier`, which on AMDGCN orders **execution but not memory**. The
result was a workgroup-level data race exposed at 92.8 % of the specialized
kernel's barriers, which corrupted reduction totals, which flipped measurement
outcomes.

> V2 hand-rolled the barrier to avoid HIP and lost the fence with it.

That is the honest shape of the trade. **HIP's abstraction costs 4,671 ns per
dispatch, and it also encodes correctness knowledge that is easy to lose when
you reimplement it.** Combine that with §13.4a — where the saving is worth
0.03 % of a median run — and the accounting is uncomfortable: on the workloads
V2 actually runs, the no-HIP policy bought a fraction of a percent and cost
weeks of debugging a memory-model detail that `__syncthreads()` had been
handling silently.

That is not an argument for reverting. The policy remains right for three
reasons that are not about the 1.74×:

1. **The short tail is real.** At 13 µs, `four_t` pays 35 % for a HIP dispatch.
2. **Self-containment.** V2's whole premise is emitting freestanding amdgcn and
   loading it directly; a `.hsaco` produced by `clang --target=amdgcn-amd-amdhsa`
   has no HIP module to be launched from. HSA is not an optimization here so
   much as the native interface for what V2 already builds.
3. **The naive-path finding.** 198 µs → 6.3 µs is a 31× win that exists only
   because the dispatch layer is owned rather than delegated.

The cost is stated as plainly as the benefit: reimplementing a runtime means
reimplementing its invariants, and §11.2 is what that looks like when one is
missed.

Also worth stating plainly: HIP has not been removed from the *repository*. The
SVM and Hybrid backends still use it (`hip_sampler.hip`), and the constraint on
this work was explicitly *"no CMake, ONLY on the MLIR part, no changes to SVM or
hybrid."* What V2 removed is HIP from **its own** dispatch and device-side
synchronization path. That removal is total and checkable:

```
$ grep -rn "#include.*hip"            src/clifft/gpu/mlir/v2/   # (none)
$ grep -rnE "\bhip[A-Z][A-Za-z]*\s*\(" src/clifft/gpu/mlir/v2/  # (none)
$ grep -rn "__syncthreads"             src/clifft/gpu/mlir/v2/
v2_ops.h:130:  // ... HIP's __syncthreads() expands to exactly this,
```

Zero HIP headers, zero HIP API calls. All 13 lines in the directory that match
"hip" case-insensitively are comments — and the sole `__syncthreads` reference
is the §11.2 post-mortem note explaining what V2's hand-rolled barrier had been
missing. Dispatch instead runs on 14 direct `hsa_*` call sites in
`v2_kernel.cc`.

### 13.6 What is still unmeasured

`batched16` is a benchmark mode, **not** something V2 does. V2's hot path is
`hsa_dispatch_and_wait` (the `persistent` shape, 6,326 ns), called exactly once
per run. A batched entry point exists — `hsa_dispatch_batch_and_wait`, which
reserves N packets plus a barrier in a single CAS — but grep finds **no caller**
outside the dispatch layer itself. It is capability, not usage.

Given §13.4a, that is the right call for now: with one dispatch per run and
millisecond-to-second kernels, batching has nothing to amortize. It would only
pay if V2 moved to a multi-dispatch structure — chunked shot batches for
progress reporting or memory-bounded rank-26 runs. Recorded in §16 as
conditional, not as a missed win.

The larger caveat on this whole section: an empty kernel isolates dispatch cost
by construction, which is what makes the comparison clean, and also what makes
it an upper bound on relevance. §13.4a is the correction, and it should be read
as part of the result rather than as a footnote to it.

---
