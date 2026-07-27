# Figure 13.1 — `diagrams/dispatch-latency.svg`

**Caption it must serve:** "Launch-to-completion latency per dispatch on gfx950
(job 50507). Log scale: the naive HSA path is 31× worse than V2's persistent
path, and batched HSA *with* a completion wait lands within 11 % of what HIP
charges for an *unsynchronized* enqueue."

**Title:** Per-Dispatch Latency — HSA vs HIP, Six Modes
**Subtitle:** gfx950 · one node · one job (50507) · 2,000 iterations × 5 reps · log scale, ns/dispatch

## What to draw

A horizontal log-scale bar chart, six bars, sorted fastest → slowest. HSA bars
in cyan `#53d8fb`, HIP bars in orange `#ff8800`. V2's actual hot path
(`persistent`) outlined in accent `#e94560`.

### DATA — steady-state ns/dispatch (exact; label every bar)

| runtime | mode | steady state |
|---|---|---|
| HSA | `batched16` | **2,320** |
| HIP | `launch_only` | **2,093** |
| HSA | `persistent` | **6,326** |
| HIP | `sync` | **10,737** |
| HIP | `stream_sync` | **10,997** |
| HSA | `naive` | **197,935** |

### DATA — what each mode actually does (needed as a legend/table panel)

| runtime | mode | what it does |
|---|---|---|
| HSA | `naive` | alloc kernarg + `agents_allow_access` + `signal_create` per dispatch, destroy + free after — what `hsa_dispatch_and_wait` did before `PersistentDispatcher` |
| HSA | `persistent` | kernarg + signal allocated once; packet written per dispatch; signal reset with `store_relaxed` — **V2's hot path** |
| HSA | `batched16` | 16 packets published back-to-back, doorbell rung once, wait once on the last |
| HIP | `sync` | `<<<>>>` + `hipDeviceSynchronize` |
| HIP | `stream_sync` | `<<<>>>` + `hipStreamSynchronize` |
| HIP | `launch_only` | `<<<>>>` only, no sync — HIP's **asynchronous enqueue floor** |

### DATA — the headline comparison (same kernel, same node, same completion semantics)

| | ns/dispatch | vs HIP `stream_sync` |
|---|---|---|
| HIP `stream_sync` | 10,997 | 1.00× |
| **HSA `persistent`** (V2's hot path) | **6,326** | **1.74× faster** (−4,671 ns) |
| **HSA `batched16`** (not yet used) | **2,320** | **4.74× faster** (−8,677 ns) |

### Required annotations

1. A bracket from `naive` to `persistent` labelled **31×** — "what
   `PersistentDispatcher` bought by hoisting the kernarg allocation, the signal
   creation and the `agents_allow_access` out of the loop."
2. A bracket from `stream_sync` to `persistent` labelled **1.74× / −4,671 ns**
   — "pure runtime abstraction: identical kernel, identical hardware queue,
   identical completion semantics, differing only in **who builds the AQL
   packet**."
3. A vertical dashed line at HIP's `launch_only` = 2,093, labelled "HIP's
   asynchronous enqueue **floor** — no completion wait at all". Then the
   striking comparison: HSA `batched16` at **2,320** *includes* a completion
   wait and lands **within 11 %** of it.
4. Reproducibility band, as a footnote strip: "peak-to-peak spread across reps
   1–4: **0.15 %** `persistent`, **0.02 %** `batched16`, **0.13 %** HIP `sync`,
   **0.09 %** `stream_sync`. Noisier: `launch_only` 0.44 %, `naive` **6 %**
   (191.6–203.6 µs) — the naive path makes two KFD ioctls and a page-table
   update per dispatch, so it inherits kernel-side scheduling jitter. It does
   not matter here, because the effect being measured is 31×."
5. Note that HIP `sync` rep 0 carries a first-touch cost (**13,668 → 10,739**)
   and is excluded from its steady-state column.

### Optional but valuable: a small AQL-packet inset

Show the two paths to the same hardware queue — HIP: `hipLaunchKernel` →
HIP runtime → ROCr → AQL packet → doorbell; HSA: V2 writes the AQL packet
directly → doorbell. Same queue, same doorbell, one fewer layer.
Include `kernarg_seg=40`, read back out of the loaded code object rather than
assumed, so the packet being timed is demonstrably carrying real arguments.

### Punchline band

"4,671 ns per dispatch is the price of asking someone else to fill in 64 bytes."
