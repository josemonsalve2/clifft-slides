# Figure 7.3 — `diagrams/scalar-pipe-deletion.svg`

**Caption it must serve:** "What specialization actually deletes. The
interpreter must *fetch* the operands before it can *use* them: a `pc × 40`
address computation, a bytecode load, a `s_waitcnt`, a `v_readfirstlane` to move
the operand from vector to scalar registers, then a second dependent load. The
specialized form has the operands as literals in the instruction stream, so the
entire prologue is gone before any useful arithmetic begins. Both pipes issue
less — the work does not migrate from VALU to SALU."

**Title:** Specialization Deletes a Dependency Chain, Not Just Instructions
**Subtitle:** The interpreter's operand fetch is two dependent loads and a cross-pipe transfer — all of it before the first useful FLOP

## What to draw — two stacked lanes, INTERPRETER above, SPECIALIZED below

Canvas ~1040 × 560.

### TOP LANE: interpreter (red `#cc2222` accents)

A horizontal chain of pill-shaped stages, left to right, with **dependency
arrows between them** (this is the point — they are serial, not parallel):

1. `s_mul_i32  pc, 40` — address generation *(SALU)*
2. `s_load_dwordx4` — fetch bytecode word *(memory)*
3. `s_waitcnt lgkmcnt(0)` — **stall** *(draw this wider / hatched to read as a
   bubble)*
4. `v_readfirstlane_b32` — **VGPR → SGPR cross-pipe transfer**
5. `s_load_dword` — second, dependent load *(memory)*
6. `s_waitcnt lgkmcnt(0)` — **stall**
7. `s_cmp` / `s_cbranch` — switch dispatch *(SALU)*
8. → then, finally, `v_fma_f32` — **the first useful arithmetic** *(VALU)*

Label the span of stages 1–7 with a red brace: **"operand fetch — 0 FLOPs"**.
Only stage 8 is the actual work.

### BOTTOM LANE: specialized (green `#00cc66` accents)

Two pills only:

1. `s_mov_b32 s4, 0x2A` — the operand, as a **literal in the instruction
   stream** *(no load, no wait)*
2. `v_fma_f32` — the same useful arithmetic

Green brace under the gap where stages 1–7 used to be: **"deleted"**.

Draw the two lanes to the same horizontal scale so the specialized lane is
visibly a fraction of the interpreter lane.

### RIGHT PANEL or BOTTOM STRIP — the measured counts for S1

Exact, from §7.3's `S1_frame_cnot` line. Use a small two-column table:

| | interpreter | specialized |
|---|---|---|
| total instructions | 136 | 47 |
| VALU | 44 | 8 |
| SALU | 74 | 35 |
| VGPRs | 8 | 3 |
| branches | 6 | 4 |

Annotate: **2.89× fewer instructions, 5.50× fewer VALU**.

### THE COUNTER-INTUITIVE POINT — make this a callout box (cyan `#53d8fb`)

> Both pipes fall. SALU drops 74 → 35 **and** VALU drops 44 → 8.
> Nothing migrates from one pipe to the other — there is simply less of both.

And below it, the corpus-wide result:

> Across all **26 of 26** circuits, the SALU ratio sits **below** the VALU
> ratio. SALU range **0.110 – 0.361** — that is **2.8× to 9.1×** fewer scalar
> instructions.

## DATA — verbatim, invent nothing

- S1 `frame operand folding`, register tier: instrs **136→47 (2.89×)**,
  v_alu **44→8 (5.50×)**, vgpr **8→3**, branch **6→4**, sgpr **14→12**.
- Corpus: SALU ratio below VALU ratio on **26 of 26** circuits.
- SALU ratio range across the corpus: **0.110 – 0.361**.

The ISA mnemonics in the two lanes are **illustrative of the chain's shape**,
drawn from §7.3's disassembly excerpt. They are structural, not counted — do not
attach a count to any individual mnemonic.

## Notes

- This is an EXTENSION of `spec-classes-gains.svg` (which shows the aggregate
  per-class gains) and of `svm-interpreter.svg` (which is deck-10 MI300X art and
  is **not** used in this report). Neither shows the dependency chain.
- Do not draw a "before/after bar chart" — that already exists. The subject here
  is **serial dependency**, so the arrows between stages carry the meaning.
