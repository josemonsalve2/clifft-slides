# Figure 7.5 — `diagrams/branch-erasure-exec-mask.svg`

**Caption it must serve:** "S2's eighteen branches do not become better-predicted
branches — they cease to exist. A constant `flags` deletes one of the two paths
*outright*, along with the test that chose between them; a constant `slot`
deletes the bounds check. Even the genuinely random outcome is if-converted to
`s_cselect_b32`, because with constant `axis`/`slot` both arms write the same
fixed bit, so the compiler can pick the *value* rather than the *path*. This is
also the one place in the study where specialization measurably costs something:
SGPRs double, 13 → 26, because the folded literals need somewhere to live."

**Title:** Eighteen Branches, Then None
**Subtitle:** Constant operands change wavefront control flow — not just the instruction count

## What it must show

**Three columns**, left to right, on the same horizontal scale.

### Column 1 — the interpreter form (red `#cc2222`)

Title: **INTERPRETER · runtime flags**. Draw the control-flow graph of the op
body as it actually is, three diamonds and their arms:

    if (IS_OWNER)                        <- runtime guard
      if (flags & FLAG_IDENTITY)         <- two live paths
         mval = (flags & FLAG_SIGN) ? 1u : 0u
      else
         outcome = fget(st->px, axis)
         mval = outcome ^ ((flags & FLAG_SIGN) != 0)
      if (slot < V2_MAX_MEAS)            <- bounds check
         mset(st->meas, slot, mval)

Both arms of the middle diamond drawn as live (solid). Beside the diamonds, a
small `exec` mask glyph — a row of 64 lane cells, some lit, some dark — with the
note **divergent scalar branches manipulate `exec` and serialize the paths**.

Footer of the column: **18 branches**.

### Column 2 — what the constants do (cyan `#53d8fb`, the mechanism band)

Three labelled deletions, each an arrow pointing at the structure it removes in
column 1. Use three distinct verbs, because they are three distinct mechanisms
and the figure exists to separate them:

1. **`flags` is a literal → path DELETED.** Not predicted, not hoisted —
   deleted, along with the test that chose between them.
2. **`slot` is a literal → bounds check DELETED.** `slot < V2_MAX_MEAS` is
   decided at emit time.
3. **`axis`/`slot` are literals → the data-dependent compare is IF-CONVERTED.**
   Both arms write the same fixed bit, so the compiler picks the *value*, not
   the *path*.

### Column 3 — the specialized form (green `#00cc66`)

Title: **SPECIALIZED · literal flags**. A single straight-line block, no
diamonds. Show the two call sites at the top:

    v2_op_meas_dormant_static(st, 6u, 33u, 0u);
    v2_op_meas_dormant_random(st, 13u, 30u, 0u);

then the real assembly for the if-converted case, with the `s_cselect_b32`
highlighted and the `v_cmp` above it:

    v_cmp_gt_f64_e32 vcc, 0.5, v[0:1]    ; rng_uniform(st->rng) < 0.5
    s_and_b64        s[2:3], vcc, exec
    s_cselect_b32    s18, 0, 0x2000      ; resolved with a SELECT, not a branch

and beside it, the purest form of the same fold:

    s_bitset0_b32    s12, 13             ; mset of a constant slot: one instruction

Footer of the column: **0 branches** — with the stronger statement attached: a
`grep` over the whole specialized file returns **zero** `s_cbranch` or
`s_branch`, not merely none in the measurement logic.

### Bottom strip — the counters, including the cost

A single row of metric cards. Five are wins (green), **one is a cost (orange
`#ff8800`)** and must be visually distinct, not hidden:

    branches      18 → 0
    total instrs  206 → 70   (2.94x — the largest instruction ratio of the eight)
    VALU          86 → 19    (4.53x)
    SALU          93 → 45
    VGPR          25 → 14
    SGPR          13 → 26    <- ORANGE. the only case in the study where
                                specialization costs something
    s_load        9 → 4

Under the SGPR card, one line: *the eighteen deleted branches were replaced by
`s_cselect`s over folded literals, and those literals need scalar registers to
live in.*

## DATA — verbatim, invent nothing

- S2 `flag folding`, register tier: branches **18→0**; total instructions
  **206→70 (2.94×)**; VALU **86→19 (4.53×)**; SALU **93→45**; VGPR **25→14**;
  SGPR **13→26**; `s_load` **9→4**.
- 2.94× is **the largest instruction ratio of the eight**; S1's 5.50× remains
  the largest VALU ratio.
- The specialized assembly contains **no `s_cbranch` or `s_branch` at all** —
  a grep over the whole file returns zero, against **18** in the interpreter
  form.
- Assembly excerpt (`spec/S2_meas_dormant.s:49-65`, intervening stores and
  `v_mov` staging elided) exactly as quoted in the report.
- Call sites (`cases/S2_meas_dormant.c`): `v2_op_meas_dormant_static(st, 6u,
  33u, 0u)`, `v2_op_meas_dormant_static(st, 8u, 32u, 1u)`,
  `v2_op_meas_dormant_random(st, 13u, 30u, 0u)`.

## Notes

- This is an EXTENSION of `spec-classes-gains.svg`, whose S2 bar gives the
  ratios without the control-flow mechanism. Do not redraw the bars.
- The SGPR rise must be drawn as a **cost**, in orange, at the same visual weight
  as the wins. The report is explicit that this is the one place in the chapter
  where specialization measurably costs something, and a figure that shows only
  the wins would misrepresent it.
- Three mechanisms, three verbs: **deleted** (path), **deleted** (bounds check),
  **if-converted** (data-dependent compare). Do not collapse them into "branches
  removed" — the separation is the point.
