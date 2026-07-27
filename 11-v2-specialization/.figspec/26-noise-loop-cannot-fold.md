# Figure 7.4 — `diagrams/noise-loop-cannot-fold.svg`

**Caption it must serve:** "The boundary of specialization, stated as a
mechanism. `OP_NOISE_BLOCK` covers the half-open site range `[start,
start+count)`, and the specializer does fold both bounds to literals. But the
loop does not iterate over that range — it consumes only the sites the PRNG
selects, and `draw_next_noise` advances the cursor by a random amount drawn
through an `ocml_log_f64` hazard call. The trip count is a function of the random
stream, so it stays a loop in both forms. The constants exist; they are simply
not the values that control the expensive part."

**Title:** Constants the Specializer Cannot Use
**Subtitle:** `start` and `count` fold to literals — but the loop is driven by the PRNG, not by the range

## What to draw

Canvas ~1040 × 580.

### LEFT — the op, with a data-dependence overlay

Show the real body (monospace, from `v2_ops_body.inc:438-451`):

```c
u32 end = start + count;
while (st->next_noise >= start && st->next_noise < end) {
    u32 site_idx = st->next_noise;
    apply_noise_site(st, sites, channels, site_idx);
    st->next_noise = site_idx + 1u;
    draw_next_noise(st, hazards, num_sites);
}
```

Overlay two colours:

- **Green `#00cc66` — folded.** `start` → `131u`, `count` → `57u`. Highlight the
  two literals in the specialized call form:
  `v2_op_noise_block(st, ..., 131u, 57u)`.
- **Red `#cc2222` — cannot fold.** `st->next_noise` and the entire
  `draw_next_noise` chain. Draw a **cycle arrow** from `draw_next_noise` back to
  the `while` condition, labelled **"trip count ← PRNG"**.

The visual claim: the folded values are on the *guard*, the unfoldable value is
on the *induction*.

### CENTRE — the one thing specialization does buy

An inset showing the **only** fold that reaches the generated code:

> `st->next_noise >= start && st->next_noise < end`
> **→ one unsigned compare** (the two-compare range test collapses when both
> bounds are literal)

Label it honestly: **"real, and small"**.

### RIGHT — what stays, drawn as a cost stack

A vertical stack of the per-iteration work that specialization does **not**
touch:

1. `apply_noise_site` — table lookup
2. PRNG advance
3. **`ocml_log_f64`** — the hazard draw *(draw this as the tallest block)*

Label the stack: **"unchanged in both forms"**.

### BOTTOM — the measured result, and why the family is weak

A compact table (exact):

| | interpreter | specialized | ratio |
|---|---|---|---|
| total instructions | 447 | 407 | 1.10× |
| VALU | 214 | 210 | **1.02×** |
| SALU | 177 | 141 | |
| branches | 23 | 20 | |
| VGPR | 56 | 56 | **no relief** |
| SGPR | 86 | 85 | |
| `s_load` | 9 | 5 | |

Then the connection to whole-circuit results:

> `circuit_d5` issues **1,720** calls, of which **329 (19.1 %)** are
> noise/readout. A noise op costs **447** interpreter-form instructions against
> **136** for a frame op.
> The d5 family runs at **0.786 – 0.856** — the corpus's weakest wins.

### CALLOUT (cyan `#53d8fb`)

Quote §14.4 verbatim — this is a prediction that preceded the measurement, and
the report's own wording is the safest way to say so:

> "The prediction §7.9 made from a single microbenchmark, **before any of these
> circuits were measured**, was that `circuit_d5` would be the corpus's weakest
> coop-tier result. **It is.**"

## DATA — verbatim from §7.9 and §14.4, invent nothing

- S7: instrs **447→407 (1.10×)**, VALU **214→210 (1.02×)**, SALU **177→141**,
  branches **23→20**, VGPR **56→56**, SGPR **86→85**, `s_load` **9→5**.
- `circuit_d5`: **1,720** calls; noise/readout **329**, i.e. **19.1 %**.
- Noise op: **447** interpreter-form instructions; frame op: **136**.
- d5 family end-to-end ratios: **0.786 – 0.856**.
- The specialized call form is
  `v2_op_noise_block(st, noise_sites, noise_channels, noise_hazards, num_noise_sites, 131u, 57u)`.

## Notes

- EXTENDS `spec-classes-gains.svg`, where S7 is a nearly-flat bar. A flat bar
  states the negative result; this figure explains **why**, which is the point.
- Do not present this as a failure of the specializer. The report frames it as a
  **boundary**: "specialization removes the scaffolding, not the math." A one-
  line footer with that quote is appropriate.
- Do not attach the `V2_NOISE_ATTR` attribute story to this figure — §11.1
  retracts the theory that it was the noise regression's cause. This figure is
  about the loop's data dependence only.
