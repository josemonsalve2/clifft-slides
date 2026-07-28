# Figure 14.2 — `diagrams/latency-not-bandwidth.svg`

**Caption it must serve:** "The global tier moves 2.43 TB in 3.173 s — 766 GB/s
against a ~8 TB/s device. The pipe is 90 % empty. That single fact decides three
separate optimization proposals at once: nothing that makes bytes *cheaper to
reach* can help a kernel that is not short of bytes. What it is short of is
**bytes in flight**. Throughput is concurrency divided by latency, and the only
intervention that moved it was the one that raised concurrency."

**Title:** The Pipe Is 90 % Empty
**Subtitle:** 766 GB/s of ~8 TB/s — the global tier is latency-bound, so three locality optimizations were decided by one counter

## The idea to draw

A **flow**, not a log-log roofline. Left to right: supply → constriction →
demand. The reader should see immediately that the wide part of the picture is
*unused*, and that the narrow part is narrow for a reason that has nothing to do
with the width of the pipe.

Canvas ~1100 × 620.

## UPPER — the flow (the heart of the figure, ~ y=70..330)

### Left: SUPPLY

A tall band, full available height (~200 px), in cyan `#53d8fb` at low fill
opacity, labelled:

- **HBM3E — ~8 TB/s peak**
- small: `MI350X · 288 GB · 8 stacks`

### Middle: THE CONSTRICTION

The band collapses — a smooth Sankey-like taper, not a step — into a **thin
ribbon roughly 1/10 the height of the supply band** (9.6 % is the true ratio;
draw it near that, do not draw it to scale so thin it vanishes — ~20 px against
~200 px). Colour the thin ribbon `#e94560`.

The huge tapered-off region above and below the ribbon is the point of the
figure. Fill it dim (`#0d0d1a` at 0.72, or white at 0.03) and label it across
its widest part:

- **7.2 TB/s of headroom, unused**
- small: `90 % of the device's memory bandwidth is idle while the kernel runs`

At the neck of the taper draw a **valve / aperture** glyph — *not* a wall.
Label it:

- **the limiter: bytes in flight**
- small: `not the width of the pipe`

This is the single most important label placement in the figure: the reader must
associate the constriction with concurrency, not with bandwidth.

### Right: DEMAND

The thin ribbon arrives at a block of CUs. Label:

- **256 CUs · 8 XCDs**
- **arithmetic intensity of the butterfly:** `U2 = 0.9` · `U4 = 1.9` flop/byte
- **machine balance ≈ 20 flop/byte**
- one line, in `#ffb35c`: **~10× to the memory-bound side of the ridge**

### The measurement strip (small, under the flow)

One line of monospace provenance, dim, ~10 px:

```
FETCH_SIZE 2,375,156,452 KB  /  3,173,389,086 ns  =  766 GB/s      TCC hit rate 68.9 %  (125,637,003,143 req · 39,048,149,329 miss)
```

## MIDDLE — Little's law, as two knobs (~ y=340..415)

A single centred equation panel, large enough to read:

**throughput  =  concurrency  ÷  latency**

Flanked by two knobs drawn as small dials or levers:

| knob | label | verdict colour |
|---|---|---|
| **latency** | "hide it — prefetch, staging, placement" | red `#cc2222` — *three attempts, all failed* |
| **concurrency** | "raise it — more resident workgroups" | green `#00cc66` — *the one that worked* |

## LOWER — the four interventions, decided by the flow above (~ y=425..555)

Four equal-width cards in a row. The first three red-bordered (`#cc2222`), the
fourth green (`#00cc66`). Each card: a heading, where it acts, and its
**measured** result.

### Card 1 — Register prefetch (VGPR pipeline)

- acts on: **latency**
- killed by: **the VGPR cap**
- `.vgpr_count 128` in *both* arms
- `.vgpr_spill_count` **152 → 375**
- bytes moved, unchanged: `2,375,292,491` → `2,375,404,660` KB (**0.005 %**)
- **rank 21 +10.3 % · rank 22 +8.4 % · rank 24 +7.3 % slower**
- one-line verdict: **prefetched 4 amplitudes by spilling 223 values**

### Card 2 — LDS tiling (V1 failure F8)

- acts on: **reuse**
- killed by: **3 barriers per tile, and an L2 that already prefetches a stride**
- **D5 −66.1 % · D7 −16.5 % · QEC −6.6 %**
- one-line verdict: **the butterfly is strided, not random**

### Card 3 — HBM stack placement

- acts on: **the pipe**
- killed by: **the flow above — 90 % headroom**
- also: NPS1 interleaves physical addresses across all 8 stacks; per-stack
  placement is not expressible
- one-line verdict: **no cross-workgroup sharing to co-locate**

### Card 4 — Resident pool (concurrency) ✓

- acts on: **concurrency**
- rank 24: **168 → 680 workgroups** (**4.05× more in flight**)
- **7.242 s → 3.468 s = 2.09×**
- one-line verdict: **the only knob that was actually turned down**

## BOTTOM BAND — the punchline (~ y=565..605)

Accent `#e94560` rounded rect, 0.10–0.18 fill, 1-px stroke, one line, 12–13 px:

**A kernel that uses 9.6 % of its bandwidth cannot be helped by moving bytes closer. It can only be helped by asking for more of them at once.**

## DATA — every number the figure may show

Nothing outside this block may appear as a number.

```
~8 TB/s          MI350X HBM3E peak bandwidth
288 GB           MI350X HBM capacity
8                HBM stacks / XCDs
256              CUs
766 GB/s         achieved (global tier, qv23_L5, rank 23)
9.6 %            766 GB/s as a fraction of ~8 TB/s
90 %             headroom, unused
7.2 TB/s         headroom in absolute terms
2.43 TB          bytes fetched
2,375,156,452    FETCH_SIZE, KB
3,173,389,086    kernel duration, ns
3.173 s          same duration in seconds
125,637,003,143  TCC_REQ_sum
39,048,149,329   TCC_MISS_sum
68.9 %           TCC hit rate
0.9              U2 arithmetic intensity, flop/byte
1.9              U4 arithmetic intensity, flop/byte
~20              machine balance, flop/byte
~10x             distance to the memory-bound side of the ridge
128              .vgpr_count, both prefetch arms, rank 22
152              .vgpr_spill_count, prefetch off
375              .vgpr_spill_count, prefetch on
223              additional values spilled (375 - 152)
4                amplitudes prefetched per iteration
2,375,292,491    FETCH_SIZE KB, prefetch off
2,375,404,660    FETCH_SIZE KB, prefetch on
0.005 %          difference between those two
+10.3 %          rank 21 slowdown from prefetch
+8.4 %           rank 22 slowdown from prefetch
+7.3 %           rank 24 slowdown from prefetch
-66.1 %          F8 LDS tiling, D5
-16.5 %          F8 LDS tiling, D7
-6.6 %           F8 LDS tiling, QEC
3                barriers per tile added by F8
168              resident workgroups, rank 24, before the pool fix
680              resident workgroups, rank 24, measured optimum
4.05x            168 -> 680
7.242 s          rank 24, 168 workgroups
3.468 s          rank 24, 680 workgroups
2.09x            7.242 / 3.468
```

## Style reminders specific to this figure

- The taper must read as **flow**, not as a bar chart. Use filled `<path>`
  regions with smooth cubic edges, in the idiom of the existing gradients.
- Green = the intervention that worked. Red = the three that did not. Cyan =
  supply and explanatory callouts. `#e94560` = the achieved ribbon and the
  punchline band. Do not use green anywhere in cards 1–3.
- The unused headroom region must be visually **larger than everything else in
  the upper half**. If a reader takes one thing from the figure, it is the size
  of that empty area.
