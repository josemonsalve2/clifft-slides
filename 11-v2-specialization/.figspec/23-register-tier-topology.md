# Figure 9.2 — `diagrams/register-tier-topology.svg`

**Caption it must serve:** "The single largest optimization in V2's history was
not an arithmetic change — it was a topology change. The first V2 had one shape:
256 threads cooperate on one shot, amplitudes in LDS, a barrier between every
op. For `frame_h` — rank 0, four instructions — that means 255 lanes
synchronizing around one lane's scalar work, four times, per shot. SVM ran one
shot per thread and was 28.258× faster. The fix parameterized *how threads
cooperate* through five macros, so the same opcode arithmetic compiles two ways
from one source."

**Title:** 256 Threads Doing the Work of One
**Subtitle:** Rank 0 in the coop topology — and what changed when cooperation became a compile-time parameter

## What to draw

Canvas ~1060 × 590. Two topologies stacked, same four-instruction circuit.

### TOP: BEFORE — the coop topology on a rank-0 shot (red `#cc2222`)

Draw **one workgroup = 256 threads** as a grid or dense row of lanes. Then a
timeline of the four instructions of `frame_h`, and for each instruction:

- **One lane active** (`IS_OWNER (t == 0)`) — colour it `#e94560`.
- **255 lanes idle** — colour them dim grey, and label the block **"255 lanes
  idle"**.
- After each instruction, a full-width **`s_barrier`** bar spanning all 256
  lanes. Four instructions → **four barriers**.

Label the whole band: **one shot** — and note that SVM, on the same work, runs
**one shot per thread**. The asymmetry is the figure.

Annotate: **"amplitudes in LDS — for a circuit with no amplitudes worth speaking
of"**.

### BOTTOM: AFTER — shot-packed register tier (green `#00cc66`)

Same lane grid, but now **every lane owns a complete shot**. Colour all 256
lanes active. Show:

- **no barriers** — draw where they were, struck through, labelled
  **`v2_barrier() → nothing`**
- **`IS_OWNER 1`** — the tid0 guards evaporate, no idle lanes
- **`V2_STRIDE 1u`** — `for (i = t; i < iters; i += 256)` becomes
  `for (i = 0; i < iters; i += 1)`
- **`V2_REDUCE2 → identity`** — a stride-1 loop has already summed everything
- **`GpuComplex v[16]` in VGPRs**, not LDS

### CENTRE — the mechanism, as a code panel

Between the two bands, the actual macro block (monospace, syntax-accented):

```c
#ifdef V2_REGISTER
#  define V2_STRIDE 1u
   static inline u32 v2_tid(void)  { return 0u; }
#  define IS_OWNER 1
   /* v2_barrier() -> nothing; V2_REDUCE2 -> identity */
#else
#  define V2_STRIDE 256u
   static inline u32 v2_tid(void)  { return __builtin_amdgcn_workitem_id_x(); }
#  define IS_OWNER (t == 0)
   /* v2_barrier() -> fenced s_barrier; V2_REDUCE2 -> coop_reduce2 butterfly */
#endif
```

Callout (cyan `#53d8fb`): **"The tier knob parameterizes *cooperation*, not
arithmetic. One source, two topologies, byte-exact."**

### RESULT STRIP — exact

| circuit | before | after |
|---|---|---|
| `frame_h` (rank 0) | **28.258** | **1.215** |

And the commit's own measurement, quoted:

> `frame_h 30.6x→1.14x`, `circuit_d3 14.1x→1.13x`. Byte-exact vs SVM: **38/38**.

## DATA — verbatim from §9.1–§9.2, invent nothing

- `frame_h` V2/SVM: **28.258** at baseline → **1.215** after P0+P1.
- `frame_h` is rank **0**, **four instructions**.
- Original topology: **256 threads** per shot, a barrier after each instruction,
  amplitudes in LDS.
- Commit `715f8d0` measured: `frame_h 30.6x→1.14x`, `circuit_d3 14.1x→1.13x`,
  byte-exact **38/38**.
- Register tier: `V2_STRIDE 1u`, `v2_tid() == 0`, `IS_OWNER 1`, no barrier,
  `V2_REDUCE2` identity. Coop/global: `V2_STRIDE 256u`, `IS_OWNER (t == 0)`,
  fenced `s_barrier`, `coop_reduce2`.

## Notes

- **Do not claim P0 made V2 faster than SVM.** The report is explicit: it brought
  V2 *to the floor* — SVM's own topology — not past it. If a footer is added,
  use: **"P0 reaches the 1-shot-per-thread floor. It removes a catastrophe; it
  does not create an advantage."**
- The two numbers 28.258 (§9.1 table) and 30.6× (the commit message) are both
  real and are measured differently. Show **28.258 → 1.215** as the figure's
  headline and put the commit quote in a smaller provenance line. Do not average
  them or present them as one number.
- EXTENDS `memory-hierarchy-tiers.svg` (which states the final topology) and
  `optimization-timeline.svg` (which shows the performance step). Neither shows
  idle lanes or barrier issue, which is this figure's subject.
