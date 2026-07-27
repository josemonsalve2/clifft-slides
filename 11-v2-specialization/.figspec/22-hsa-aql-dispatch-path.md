# Figure 13.2 — `diagrams/hsa-aql-dispatch-path.svg`

**Caption it must serve:** "Why the same dispatch costs 197,935 ns or 6,326 ns
depending on what sits on the hot path. The naive HSA path re-allocates the
kernarg segment, re-authorizes GPU access to it, and creates a completion signal
*per dispatch* — three runtime round-trips before the packet is even written.
The persistent path allocates all three once at setup and reuses them, so the
per-dispatch cost is packet write, doorbell ring, signal wait. Same kernel, same
completion semantics, 31× apart."

**Title:** What Actually Happens on an AQL Dispatch
**Subtitle:** The 31× is not the doorbell — it is what the naive path re-creates before reaching it

## What to draw

Canvas ~1080 × 620. Three horizontal bands: **SETUP (once)**, **PER DISPATCH**,
**GPU**. The AQL queue sits in the middle as shared structure.

### THE SHARED STRUCTURE — draw this once, centre

A **queue ring buffer**: a row of ~6 slots, one highlighted as "slot = write
index mod size", with:

- `hsa_queue_load_write_index_relaxed()` → the slot
- The **AQL packet** drawn as a labelled 64-byte record with the fields that
  matter: `setup / grid_size / workgroup_size`, **`kernel_object`**,
  **`kernarg_address`** →, **`completion_signal`** →, and `header` written
  **last** *(annotate: "header written last — it is the release-store that makes
  the packet live")*
- `hsa_signal_store_relaxed(doorbell, index)` → **ring**
- `hsa_signal_wait_scacquire(completion)` → **wait**

Two arrows leave the packet: one to a **kernarg segment** block, one to a
**completion signal** block. These two blocks are the subject of the figure —
their *placement* is what differs between modes.

### THE TWO MODES — show placement, not two separate diagrams

**Naive (red `#cc2222`):** put `hsa_amd_memory_pool_allocate` (kernarg),
`hsa_amd_agents_allow_access`, and `hsa_signal_create` **inside the PER DISPATCH
band**, each with a red arrow marked *"runtime round-trip"*. Annotate the band:
**"3 allocations + 1 access grant, every dispatch"** and note the naive path
makes **two KFD ioctls and a page-table update per dispatch**.

**Persistent (green `#00cc66`):** the same three boxes drawn **inside the SETUP
band**, greyed as "done once", with green dashed reuse arrows down into the per
dispatch band. The per-dispatch band then contains only: **write packet → ring
doorbell → wait signal**.

Use one shared queue drawing with the three resource boxes shown in **both**
positions, connected by a bracket labelled *"the only difference"*. Do not draw
the queue twice.

### THE BATCHED VARIANT — a small third strip

`batched16`: **16 packets written, one doorbell ring, one wait**. Draw 16 small
slots filling, then a single ring/wait. Annotate: **"amortizes the doorbell and
the signal wait across 16 dispatches — 2,320 ns"**.

### RESULT STRIP — bottom, exact numbers

| mode | ns/dispatch | what sits on the hot path |
|---|---|---|
| HSA `naive` | **197,935** | alloc + access-grant + signal-create, per dispatch |
| HIP `stream_sync` | **10,997** | HIP's own bookkeeping above the same AQL queue |
| **HSA `persistent`** | **6,326** | packet write, doorbell, wait |
| HSA `batched16` | **2,320** | 1/16 of a doorbell and a wait |

Two callouts:
- **`persistent` is 1.74× faster than HIP `stream_sync` (−4,671 ns)** — V2's hot path.
- **`naive` → `persistent` is 31×.** Same queue, same packet, same kernel.

## DATA — verbatim from §13.1–§13.4, invent nothing

- Steady-state ns/dispatch: naive **197,935**; persistent **6,326**;
  batched16 **2,320**; HIP sync **10,737**; HIP stream_sync **10,997**;
  HIP launch_only **2,093**.
- persistent vs HIP stream_sync: **1.74× faster, −4,671 ns**.
- batched16 vs HIP stream_sync: **4.74× faster, −8,677 ns**.
- The benchmark's kernarg segment is **40 bytes**.
- Job 50507, node `smci350-rck-g03-d13-21`, gfx950, ROCm 7.2.3,
  2,000 iterations × 5 reps.

## Notes

- This EXTENDS `dispatch-latency.svg`, which is the timing bars. Do not redraw
  the bars — the subject here is the **resource path** that produces them.
- `batched16` must be labelled **"capability, not current usage"** — the report
  is explicit that V2 does not use it today.
- Do not claim the 1.74× is why V2 is fast end-to-end. §13.4a measures dispatch
  overhead at **0.03 % of runtime at the median**. If a footer is added, say:
  **"1.74× on a cost that is 0.03 % of median runtime — the argument for HSA is
  the barrier fence and the short tail, not this ratio."**
