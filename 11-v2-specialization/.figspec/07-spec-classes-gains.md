# Figure 7.1 — `diagrams/spec-classes-gains.svg`

**Caption it must serve:** "Per-class specialization gains, three bars per case:
instructions, VALU and SALU, ordered by VALU ratio. All three fall in all eight
cases; on S5, S6 and S8 the SALU bar is the tallest, which is the microbenchmark
form of the corpus-scale finding in §14.3. S7 (noise) is flat on all three, and
flat on registers too."

**Title:** Eight Specialization Classes — Measured Gains
**Subtitle:** interpreter → specialized, same op, same operands · ratios, higher is better

## What to draw

A grouped bar chart: eight groups (S1…S8), **ordered by VALU ratio descending**
(so: S1, S2, S5, S3, S4, S6, S8, S7), three bars per group.

Bar colours: instructions = `#aeb4c7`, VALU = `#53d8fb` (cyan), SALU =
`#e94560` (accent). Legend top-right.

### DATA — exact, all 24 numbers. Every bar must carry its ratio as a label.

| Case | Class | Tier | instrs | ratio | VALU | ratio | SALU | ratio |
|---|---|---|---|---|---|---|---|---|
| S1 | frame operand folding | register | 136→47 | 2.89× | 44→8 | **5.50×** | 74→35 | 2.11× |
| S2 | flag folding | register | 206→70 | 2.94× | 86→19 | **4.53×** | 93→45 | 2.07× |
| S5 | scatter-index folding | coop | 207→81 | 2.56× | 84→33 | 2.55× | 102→33 | **3.09×** |
| S3 | static rank tracking | coop | 164→83 | 1.98× | 67→27 | 2.48× | 84→48 | 1.75× |
| S4 | rank-folded reduction | coop | 331→240 | 1.38× | 139→96 | 1.45× | 133→93 | 1.43× |
| S6 | fused-matrix lookup | coop | 147→86 | 1.71× | 54→42 | 1.29× | 83→36 | **2.31×** |
| S8 | Pauli-mask index | register | 185→137 | 1.35× | 71→63 | 1.13× | 87→49 | **1.78×** |
| S7 | noise runtime loop | register | 447→407 | 1.10× | 214→210 | **1.02×** | 177→141 | 1.26× |

### Required annotations (these are the argument, not decoration)

1. A **1.0× baseline line** across the chart labelled "no gain".
2. Mark S5, S6, S8 — where the **SALU bar is the tallest** — with a small cyan
   bracket labelled "SALU falls fastest: work is **deleted**, not moved to the
   scalar pipe".
3. Mark **S7** in red as "the negative result — kept on purpose". Annotate:
   "1.10× instrs · 1.02× VALU · **VGPR 56→56, unchanged**. Data-dependent; there
   is nothing to fold. This case predicts the `circuit_d5` behaviour in §11.1."
4. Mark **S2** with an orange flag: "the one case where SGPR pressure goes **up**:
   13→26, while SALU instructions fall 93→45 and **branches fall 18→0**. The
   deleted branches became `s_cselect`s over folded literals, and literals need
   registers."
5. A note under the axis: "spread is **5.50× to 1.02×** — specialization is not
   a uniform multiplier; opcode mix predicts speedup."

### Punchline band

"All three counters fall in all eight cases. The mechanism is deletion, not
substitution — confirmed at corpus scale on 26 of 26 circuits (§14.3)."
