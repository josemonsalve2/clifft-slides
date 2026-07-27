# Figure 5.2 — `diagrams/ir-density.svg`

**Caption it must serve:** "Lines of emitted source per bytecode instruction,
log scale. V1 (red) is superlinear on U2/U4-dense circuits. Hybrid (amber) is
linear with a constant of ~18. V2 (blue) converges to 1.00."

**Title:** Source Lines per Bytecode Instruction
**Subtitle:** log scale · lower is better · V2 converges to 1.00

## What to draw

A log-scale bar or dot chart, one group per circuit, three series (V1 red
`#cc2222`, Hybrid amber `#ff8800`, V2 accent `#e94560` — the caption says
"blue" for V2 but use the deck's highlight cyan `#53d8fb` for V2 and note that
the report caption will be updated to match; prefer cyan for V2).

### DATA — the anchor facts, exact and non-negotiable

- `qv10`: **140 bytecode instructions → 336,988 lines of MLIR** under V1.
  V2 emits **179 lines** for the same circuit with identical semantics —
  a **1,883× reduction**. This is the one genuinely anomalous row, anomalous
  by a factor of **35 against its neighbours**.
- The three longest circuits (**1,720 → 4,296 instructions**) sit at
  **55–69 lines/instr** under V1 — essentially **flat**. V1's bloat was
  per-instruction, not super-linear in circuit length.
- Two small-circuit rows read **236.8** and **125.8** lines/instr; those are
  *fixed preamble amortized over few instructions*, not a U2/U4 effect. Mark
  them with a footnote marker so they are not misread.
- V2 converges to **1.00** lines per instruction.
- Hybrid is linear with a constant of **~18**.

Because exact per-circuit values for every row are not all quoted, draw ONLY
the rows named above and label the x-axis by circuit-size class rather than
inventing a full table. Any bar you draw must carry a number from this list.

### Required annotations

1. A horizontal reference line at **1.00** labelled "V2 — one line of C per
   bytecode instruction".
2. A callout on `qv10`'s V1 bar: **336,988 lines from 140 instructions** and,
   pointing at V2's, **179 lines · 1,883×**.
3. A shaded band over the 1,720–4,296-instruction region labelled
   "55–69 lines/instr — flat: the bloat is per-instruction, not per-circuit".
4. A note that **236.8 / 125.8** are preamble amortization, not U2/U4.

### Punchline band

"A 387-instruction QV circuit could fail to compile while a 4,296-instruction
surface circuit succeeded. Length was never the variable — opcode mix was."
