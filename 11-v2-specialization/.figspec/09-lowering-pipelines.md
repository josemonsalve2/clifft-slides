# Figure 8.1 — `diagrams/lowering-pipelines.svg`

**Caption it must serve:** "The two pipelines with per-stage line counts for
`circuit_d3`. V1's representation is largest where a human wrote it; V2's is
largest where the compiler expanded it at `-O0` and smallest again after `-O2`."

**Title:** Two Lowering Pipelines, Same Circuit
**Subtitle:** circuit_d3 · 344 bytecode instructions · lines per stage

## What to draw

Two vertical pipelines side by side, each stage a box whose **width or height is
proportional to its line count** (this is the whole point — the reader should
see V1 start huge and shrink, V2 start tiny, balloon, and collapse).

### DATA — V1 pipeline, circuit_d3 (exact)

| stage | lines |
|---|---|
| `1_emitted.mlir` | **23,002** |
| `2_opt.mlir` | 14,257 |
| `3_translate.ll` | 14,875 |
| `4_optO2.ll` | 13,747 |
| `5_isa.s` | **17,802** |

### DATA — V2 pipeline, circuit_d3 (exact)

| stage | lines |
|---|---|
| `1_emitted.c` | **383** |
| `2_clangO0.ll` | 19,551 |
| `3_clangO2.ll` | 8,132 |
| `4_isa.s` | **10,398** |

### DATA — V1 on the small circuit `frame_h` (4 instructions), for the inset

| stage | lines |
|---|---|
| `1_emitted.mlir` | 947 |
| `2_opt.mlir` | 606 |
| `3_translate.ll` | 562 |
| `4_optO2.ll` | 275 |
| `5_isa.s` | 524 |

(V2 has no rendered chain for `frame_h`; its emitted C is **43 lines**. Say so.)

### Required annotations

- V1's arc: "**starts huge and shrinks** — every stage is *recovering* from the
  emitter." Red.
- V2's arc: "**starts tiny, expands at -O0, collapses at -O2**. The expansion is
  a compiler artifact — `-O0` gives every local an `alloca` and every access an
  `addrspacecast` — not something the emitter wrote down." Cyan.
- The endpoint comparison, large and unmissable:
  **17,802 vs 10,398 ISA lines. V2 produces a 1.7× smaller kernel from a 60×
  smaller source.**
- Mark the source ratio 23,002 / 383 = **60×** on the top edge and the ISA ratio
  17,802 / 10,398 = **1.7×** on the bottom edge.

### Punchline band

"Two shapes, and they are opposites."
