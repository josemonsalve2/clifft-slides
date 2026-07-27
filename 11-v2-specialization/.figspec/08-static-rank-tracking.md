# Figure 7.2 — `diagrams/static-rank-tracking.svg`

**Caption it must serve:** "The specializer walks the bytecode maintaining `k`,
so every rank-dependent bound is a literal at its use site. The interpreter must
reload `st->active_k` at every instruction because any preceding op could have
changed it."

**Title:** Static Rank Tracking — the specializer knows `k` at every call site
**Subtitle:** S3 · 164→83 instructions (1.98×) · VALU 67→27 (2.48×) · VGPR 18→8 · SGPR 16→10

## What to draw — three panels, left to right

### Panel 1: the walk

A short vertical program listing, with a `k` column beside it that the
specializer maintains as it walks. Show k *growing statically across the
program* — the report's exact phrasing is "the compiler sees the rank grow,
statically, across the program", and it names the concrete case where an
`EXPAND` site "gets `2u`". Draw ~6 rows: a frame op (k unchanged), an EXPAND
(k: 1→2, annotate `active_k = 2u` folded here), an array op, a second EXPAND
(k: 2→3), an array op, a measure (k: 3→2). Colour the k column cyan.

### Panel 2: what folds at the use site

Three side-by-side code pairs, interpreter (red-tinted) vs specialized
(green-tinted), monospace:

```c
// interpreter                    // specialized
n = 1u << (st->active_k - 1);     n = 4u;            // literal trip count
im = dagger ? -w.im : w.im;       im = -w.im;        // sign folded, ?: gone
v[i + half]                       v[i + 4]           // constant displacement
```

### Panel 3: the `s_load` accounting — 6 → 4

This is the subtle part and the figure must get it right: the count does **not**
fall by simple deletion. Draw it as a two-column ledger, exactly:

| load | interpreter | specialized |
|---|---|---|
| `0x20` — `pc` | ✓ | — |
| `0x18` — `instrs` | ✓ | — |
| `0x0` — kernarg block | ✓ | ✓ |
| `0x278` — `st->active_k` | ✓ | — |
| dynamically-indexed frame word (`s10`, `s0`) | ✓ ✓ | — |
| `st->px` / phase operands at fixed offsets | — | ✓ ✓ ✓ |

Annotate: "**three** interpreter-only loads (`pc`, `instrs`, `active_k`) and
**both** dynamically-indexed frame reads disappear; **three** statically-addressed
loads appear. Net 6→4 — but every surviving load now has a **compile-time
address**."

Highlight `0x278` specially: the interpreter form contains exactly one
`s_load_dword ..., 0x278` (that is `active_k`'s offset in `V2State`); the
specialized form contains **none**.

### Punchline band

"What it does **not** buy: the loop is still there. At rank 8 with
V2_STRIDE = 256 that is a handful of iterations; at rank 22, thousands. V1 tried
to emit those iterations as code and produced 20 MB of IR that took 221 s to
compile. V2 emits one loop with a known bound — which is what LLVM's loop
optimizer actually wants."
