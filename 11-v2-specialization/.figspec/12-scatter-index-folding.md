# Figure 9.2 — `diagrams/scatter-index-folding.svg`

**Caption it must serve:** "The global-tier bottleneck. Left: the interpreter
recomputes `scatter_bits_2` per amplitude, 8–12 VALU each, 256 threads deep.
Centre: SVM's runtime scatter LUT, which it disables on the global tier. Right:
V2's compile-time fold — three constant masks and a `v_or3`."

**Title:** Three Answers to One Index Computation
**Subtitle:** scatter_bits_2 → insert_zero_bit · the global tier's bottleneck (commit 5d10409)

## What to draw — three panels, left to right, plus a results strip

### LEFT — V2's interpreter (red `#cc2222`)

Recomputes `scatter_bits_2` **per amplitude**, **8–12 VALU each**, paid
**256-way per shot**. Draw 256 threads (or a representative strip of them, with
"× 256" notation) each running the same shift/mask/or chain. Label the cost.

### CENTRE — SVM's answer (orange `#ff8800`)

A **runtime lookup table in LDS**, `scatter_lut[kSharedMaxAmplitudes/2]`, keyed
by `(axis1, axis2, k, mode)`. Draw the LDS table and the per-amplitude load
from it. Then the critical annotation, in red:

"**SVM leaves this LUT OFF for the global tier.** That is precisely why V2's
global specialization wins so decisively."

### RIGHT — V2's answer (accent `#e94560`)

Fold the index arithmetic **at compile time**: three constant masks and a single
`v_or3`. Draw the folded form as a tiny code panel next to the sprawling left
panel — the size contrast is the argument. Annotate:

"strictly better — **no table, no LDS for the table, no lookup**"

### DATA — the bottleneck analysis, verbatim from commit 5d10409

> "Bottleneck analysis (rocprofv3, V2 global vs SVM): V2 did 2× the VALU (2.34B
> vs 1.18B) and 25× more L2 misses at the same wall time — the extra VALU is
> per-amplitude scatter-index recompute (`scatter_bits_2` → `insert_zero_bit`,
> ~8–12 VALU each) that the runtime interpreter pays 256-way per shot and **SVM
> avoids via a scatter LUT it leaves OFF for global.**"

Put `2.34B vs 1.18B VALU` and `25× more L2 misses` as two large stat callouts.

### DATA — the result strip (exact)

Speedup over the global interpreter:

| circuit | speedup |
|---|---|
| surface_d7_t19 | **3.37×** |
| surface_d9_t19 | **3.78×** |
| surface_d11_t15 | **3.87×** |

And in the progressive table those three go from **~1.0 to ~0.26–0.31 in a
single step**.

### Punchline band

"SVM's answer was a runtime table. V2's answer was to not need one."
