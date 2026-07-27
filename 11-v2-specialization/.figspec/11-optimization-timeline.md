# Figure 9.1 — `diagrams/optimization-timeline.svg`

**Caption it must serve:** "V2/SVM kernel-time ratio across the twelve archived
benchmark runs. Each tier's curve is flat until the one change that addresses
it, then flat again. The y-axis is log-scaled to fit `frame_h`'s 28.258
baseline."

**Title:** Each Tier's Win Arrives in Exactly One Step
**Subtitle:** V2/SVM kernel-time ratio · log scale · lower is better · 1.0 = parity

## What to draw

A line chart: x = the seven reported optimization stages in order, y = V2/SVM
ratio on a **log scale** (must span 0.25 → 30). One line per circuit, eight
lines. Draw a horizontal **parity line at 1.0**, labelled.

### DATA — the exact table. Every point must be one of these numbers.

x-axis categories, in order:
`baseline` · `after P0+P1` · `after specializer` · `fenced+gated`* · `noise-specialized` · `global-specialized` · `final`

| circuit | baseline | after P0+P1 | after specializer | *fenced+gated* | noise-specialized | global-specialized | **final** |
|---|---|---|---|---|---|---|---|
| frame_h | **28.258** | 1.215 | 0.612 | *2.859* | 0.614 | 0.645 | **0.626** |
| circuit_d3_p0.001 | 15.193 | 1.081 | 1.116 | *2.117* | 0.524 | 0.504 | **0.525** |
| qv10 | 1.313 | 1.237 | **0.252** | *0.675* | 0.308 | 0.310 | **0.310** |
| surface_d7_t15 | 1.650 | 1.469 | 1.787 | *1.427* | **0.503** | 0.507 | **0.505** |
| surface_d9_t10 | 1.642 | 1.473 | 1.796 | *1.402* | **0.484** | 0.481 | **0.481** |
| surface_d7_t19 | 0.968 | 0.872 | 1.014 | 1.017 | 1.015 | **0.312** | **0.298** |
| surface_d9_t19 | 0.906 | 0.817 | 0.973 | 0.971 | 0.972 | **0.262** | **0.262** |
| surface_d11_t15 | 0.938 | 0.854 | 1.005 | 1.005 | 1.005 | **0.259** | **0.256** |

### Colour by tier (this is what makes the figure readable)

- `frame_h`, `circuit_d3_p0.001` — **register tier**, green `#00cc66`
- `qv10`, `surface_d7_t15`, `surface_d9_t10` — **coop tier**, orange `#ff8800`
- `surface_d7_t19`, `surface_d9_t19`, `surface_d11_t15` — **global tier**, red `#cc2222`

### Required annotations — the "one step" claim, four vertical markers

1. At **after P0+P1**: "register tier lands here — 28.258 → 1.215"
2. At **after specializer**: "coop tier lands here — qv10 1.237 → 0.252"
3. At **noise-specialized**: "noise-heavy coop lands here — surface_d7_t15 1.427 → 0.503"
4. At **global-specialized**: "global tier lands here — surface_d9_t19 0.972 → 0.262"

Each marker gets a vertical dashed rule in the corresponding tier colour.

### The `fenced+gated` column MUST be flagged as an artifact

Shade that x-position and label it clearly:

"**MEASUREMENT ARTIFACT, shown not dropped.** At `9d9cc68` the gate's verdict
was cached only in-process. `rocprofv3` spawns a fresh process per invocation,
so the gate's own validation dispatches re-ran *inside the profiled region* and
the digester summed them. The kernel trace shows **three dispatches where every
other run has one**. `bbb5e42` persisted the verdict to `<hsaco>.gate` and the
numbers returned to trend. Nothing had slowed down."

Include the trace evidence as a small monospace inset:
```
noise-fenced-gated  frame_h  clifft_v2_register x1 (19.1us) + clifft_v2_spec x2 (28.8us)
                    qv10     clifft_v2_coop     x1 (1270.2us) + clifft_v2_spec x2 (1667.2us)
noise-specialized   frame_h  clifft_v2_spec     x1 (10.5us)
                    qv10     clifft_v2_spec     x1 (1340.0us)
```

### Second required annotation: the ~1.0 plateaus are interpreter cells

Mark the flat ~1.0 segments (surface_d7_t19 / d9_t19 / d11_t15 across the first
five columns) with a light band labelled: "the kernel trace names
`clifft_v2_global` here — the specializer had not reached this tier. Every
movement in this chart is a **change of which kernel ran**, not a tuning effect."

### Node caveat — must appear as a footnote line

"Columns 1–8 of the archive ran on `smci350-rck-g03-d13-21`; columns 9–12,
including **final**, ran on `smci350-rck-g03-f13-21`. Every cell is a V2/SVM
ratio with both halves in the same job, so the comparison survives — but the
step between those two columns carries a node change as well as a code change."

### Punchline band

"Structural changes, not tuning: each tier is flat until the one commit that
addresses it, then flat again."
