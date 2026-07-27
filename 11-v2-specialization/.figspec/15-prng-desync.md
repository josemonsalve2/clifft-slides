# Figure 12.1 — `diagrams/prng-desync.svg`

**Caption it must serve:** "The desynchronization mechanism. A single clamp taken
on one side and not the other shifts the entire remaining draw sequence for that
shot. The outcome of the clamped branch is *identical* on both sides; it is the
bookkeeping that diverges."

**Title:** One Skipped Draw Desynchronizes Two Backends Forever
**Subtitle:** xoshiro256++ · a shifted stream is not approximately right, it is a different sequence

## What to draw

Two parallel PRNG streams (CPU top, GPU bottom) as a row of numbered draw
tokens: `r0 r1 r2 r3 r4 r5 …`. At measurement #3, the CPU **clamps** (consumes
nothing) and the GPU **draws**. From that point the two rows are offset by one
token — draw the misalignment explicitly and shade everything downstream in red
with the label "decorrelated for the rest of the shot".

Crucially: put a green ✓ on measurement #3's *outcome* on both sides. **The
outcome is the same.** The label must be "same outcome, different bookkeeping" —
that is the counter-intuitive part the caption insists on.

### DATA — the hinge, quote verbatim in a code panel

```c
// The clamp decision must match the SVM's on every call: a branch clamped on
// one side and rolled on the other consumes a PRNG draw the other never did,
// and the two streams never resynchronize. See V2_DUST_EPS.
static inline u8 sample_branch(u64* rng, double p0, double p1, double total) {
    double eps = V2_DUST_EPS * total;
    if (p1 <= eps) return 0;          // <-- returns WITHOUT drawing
    if (p0 <= eps) return 1;          // <-- returns WITHOUT drawing
    return (rng_uniform(rng) * total < p0) ? 0u : 1u;
}
```

Highlight the two early-return lines in accent `#e94560` — both skip
`rng_uniform`.

### DATA — the reference case, exact (`H; T^4; H; M`, pinned at tests/test_svm.cc:1995)

| precision | p0 | vs `1e-18` | behaviour |
|---|---|---|---|
| fp64 | **2.465e-32** | ≤ eps | **clamp**, no draw |
| fp32 | **8.882e-16** | > eps | **no clamp**, draw consumed |

Draw this as the concrete instance driving the two streams. Annotate: "both
sides still pick the same outcome — `p0/total` is `1 − 1e-15`, so any uniform
draw lands the same way — but the GPU has consumed a draw the CPU has not."

And the frequency claim: "**in a surface-code circuit nearly every stabilizer
measurement is deterministic, so this fires constantly.**"

### The two failure modes panel (both must appear; the report separates them deliberately)

1. **Branch-probability rounding.** Probabilities are reduced in f64 but from
   f32 *inputs*, so `p0/total` is good to only **~1e-6 relative** after summing
   `2^k` terms. Prediction: divergence scales with amplitudes summed — invisible
   at register rank, visible at coop rank 10.
2. **A mis-calibrated clamp threshold.** Independent of rounding, and — as it
   turned out — **the dominant effect.**

Mark #2 as the dominant one.

### Punchline band

"A threshold that never fires is a correctness bug, not dead code."
