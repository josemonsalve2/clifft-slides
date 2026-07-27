# Figure 8.2 — `diagrams/f64-attribution.svg`

**Caption it must serve:** "V1's f64 instruction volume for `circuit_d3`,
decomposed by the A/B experiment. 68 % is 54 inlined copies of a hand-written log
polynomial; the 1,380 that remain are the same PRNG, `cnorm` and `cscale` sites
V2 has. V2 keeps `log()` as a call to `__ocml_log_f64` because it links ROCm
device bitcode; V1 had no such link and had to write the polynomial in MLIR."

**Title:** Where V1's 4,347 f64 Instructions Come From
**Subtitle:** A/B on V1's own stage-3 IR through V1's own pipeline (opt -O2 | llc -O2) · one variable changed

## What to draw

Left: a stacked bar decomposing variant A's f64 count. Right: the full A/B
table. Bottom: the mechanism.

### DATA — the experiment, exact

Both variants compile **V1's own stage-3 IR** through **V1's own pipeline**,
changing exactly one thing:

- **A** — as-is. Asserted to reproduce `v1/circuit_d3.5_isa.s` **byte-for-byte**.
- **B** — `clifft_log` and `clifft_draw_next_noise` marked `noinline`. Same
  amplitude arithmetic, same PRNG; the transcendental stays a call.

```
variant, isa_lines, isa_total_instrs, v_f64, v_f32, v_pk_f32, scratch_ops, scratch_bytes, log_expansions
A,           17802,            16675,  4347,  1586,        0,          56,           156,             54
B,           12038,            11148,  1380,  1586,        0,         477,           224,              1
```

Every one of those cells must appear in the figure's table panel.

### The decomposition (compute from the above; these are the exact results)

- f64 total, A: **4,347**
- f64 attributable to inlined `log()`: 4,347 − 1,380 = **2,967 = 68 %**
- f64 that remains (same PRNG / `cnorm` / `cscale` sites V2 also has): **1,380**
- total instructions moved: 16,675 − 11,148 = **5,527**
- `log_expansions`: **54 → 1**

Draw the stacked bar as 2,967 red (inlined log polynomial, 54 copies) over
1,380 grey (irreducible), with the 68 % called out.

### The control that makes it an experiment

**`v_*_f32` is identical across A and B — 1,586 either way.** Draw this as a
flat, unchanged bar beside the f64 bars, labelled "the amplitude arithmetic
never moved". This is what rules out the naive reading ("V1 was just doing
everything in double").

### The tradeoff B exposes (do not omit — it is why B is not simply "better")

`scratch_ops` **56 → 477** and `scratch_bytes` **156 → 224**. Keeping `log` as
a call moves the cost from instruction volume into spill traffic.

### The mechanism panel

Both backends follow the same rule for the same reason. Quote both, side by side:

V2 (`v2_ops.h`):
```c
static inline double cnorm(CV2Complex v) {
    double re = (double)v.re, im = (double)v.im; return re * re + im * im;
}
```

V1 (`mlir_emit.cc:826`, inside `emit_cnorm`):
```
// Match gold cnorm (hip_sampler.hip): extend each f32 component to f64,
// then square and sum in f64. Squaring in f32 first (then extending) loses
// precision and flips borderline stabilizer measurement branches on
// RNG-path-dependent shots.
```

Then the asymmetry: **V2 links ROCm device bitcode and calls `__ocml_log_f64`;
V1 had no such link and had to write the polynomial in MLIR.** The count is
exact, not estimated — `0x3FD5555555555555` is a coefficient unique to that
polynomial and it occurs 54 times.

### Punchline band

"V1's 23× f64 volume was not a precision policy. It was a missing link line."
