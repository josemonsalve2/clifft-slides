## 12. The gap: f32 vs f64, and how it was narrowed

The GPU carries amplitudes in f32; the CPU reference carries them in f64. That
much is by design, and it is stated in the device ABI:

```c
// ---- Complex amplitude (f32 storage; f64 used only in reductions) ----------
typedef struct __attribute__((aligned(8))) { float re; float im; } CV2Complex;
```

against `std::complex<double>` throughout `src/clifft/svm/svm.h`. What the
in-tree documentation asserted — `docs/v2/pre_V2.md:66`, *"f32 vs f64
accumulation differs across backends"* — is prose. This section replaces it with
an experiment, and the result is more specific and more actionable than the
prose suggests.

### 12.1 Why a precision difference becomes a *correctness* difference

The mechanism is not that the answers are slightly different. It is that the two
backends can consume **different numbers of random numbers**.

`sample_branch` is the hinge:

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

Both early returns skip `rng_uniform`. So if one backend clamps and the other
does not, the clamping side is one draw behind forever. The PRNG is
xoshiro256++: a shifted stream is not *approximately* right, it is a different
sequence. Every subsequent measurement in that shot decorrelates.

This yields two distinct failure modes, and separating them is the whole story
of this section:

1. **Branch-probability rounding.** Probabilities are reduced in f64 but from
   f32 *inputs*, so `p0/total` is good to only ~1e-6 relative after summing
   `2^k` terms. When `rand*total` lands within that window of `p0`, the two
   sides take different branches. Prediction: divergence should scale with
   amplitudes summed — invisible at register rank, visible at coop rank 10.
2. **A mis-calibrated clamp threshold.** Independent of rounding, and — as it
   turned out — the dominant effect.

<figure>
<img src="diagrams/prng-desync.svg" alt="How one skipped PRNG draw desynchronizes two backends permanently" width="100%">
<figcaption><b>Figure 12.1</b> — The desynchronization mechanism. A single
clamp taken on one side and not the other shifts the entire remaining draw
sequence for that shot. The outcome of the clamped branch is <em>identical</em>
on both sides; it is the bookkeeping that diverges.</figcaption>
</figure>

### 12.2 The dominant cause: a constant that outlived its precision

`V2_DUST_EPS` was copied verbatim from the SVM's `kDustEpsilon = 1e-18`
(`svm_internal.h:46`). That value is calibrated for `std::complex<double>`, whose
analytically-zero interference lands at 1e-30…1e-24. V2 stores f32, where the
same interference concentrates at `fp32_eps² = 1.4e-14` — **four decades above
the threshold**. On the GPU the dust branch therefore *never clamped*.

The reference case is a four-line circuit (`H; T^4; H; M`, the one
`tests/test_svm.cc:1995` pins):

| precision | p0 | vs `1e-18` | behaviour |
|---|---|---|---|
| fp64 | 2.465e-32 | ≤ eps | **clamp**, no draw |
| fp32 | 8.882e-16 | > eps | **no clamp**, draw consumed |

Both sides still pick the same *outcome* — `p0/total` is `1 - 1e-15`, so any
uniform draw lands the same way — but the GPU has consumed a draw the CPU has
not. In a surface-code circuit nearly every stabilizer measurement is
deterministic, so **this fires constantly**.

As the commit puts it: *"A threshold that never fires is a correctness bug, not
dead code."*

**Sizing the replacement.** The new value is not a guess. A branch probability is
a sum of `half = 1 << (active_k - 1)` squared fp32 magnitudes, and the rounding
error is *relative to each amplitude* — so the dust floor does **not** grow with
term count; summing more terms averages the residuals rather than accumulating
them. Measured (residual model, `p1/total` over 2,000 trials per rank):

| rank | median | max |
|---|---|---|
| 1 | 9.7e-15 | **1.2e-13** |
| 4 | 1.3e-14 | 5.8e-14 |
| 12 | 1.42e-14 | 1.6e-14 |
| 26 | 1.42e-14 | 1.5e-14 |

The distribution concentrates on `fp32_eps² = 1.42e-14` and **tightens** with
rank; the widest tail is at *low* rank (~1.2e-13). One constant therefore covers
ranks 1–26. `1e-11` sits ~2 decades above that tail and ~5 decades below the
smallest probability f32 can meaningfully carry, leaving margin on both sides.
Nothing real is clamped away: genuine small probabilities (the SVM comment cites
`R_ZZ` angles producing ~1e-16) are not representable in f32 storage in the
first place.

This is worth pausing on, because it inverts the intuition the "f32 vs f64"
framing invites. The dust floor is **rank-independent**, and the *low*-rank
circuits have the widest spread. A threshold picked for the largest circuit would
have been wrong for the smallest.

### 12.3 The A/B that confirmed it

Two arms, identical in every respect but the constant (job 50444, node
`smci350-rck-g03-f13-21`):

| arm | `V2_DUST_EPS` | d5 (coop, rank 10) shot 0 | d3 (register, rank 4) shot 0 |
|---|---|---|---|
| A | `1e-18` (fp64-calibrated) | gpu=[0] cpu=[1] **MISMATCH** | gpu=[0] cpu=[0] ok |
| B | `1e-11` (fp32-calibrated) | gpu=[1] cpu=[1] **match** | gpu=[0] cpu=[0] ok |

The rank-dependence prediction from §12.1 holds exactly: the register-tier
circuit agreed under *both* thresholds, and the coop-tier circuit disagreed only
under the fp64-calibrated one. Fixed in `2a015fd` — **a constant-only change**;
all four `sample_branch` call sites are untouched.

### 12.4 Post-fix verification: separating logic from statistics

The mistake available here is to check aggregate counts, see them differ by a
percent or two, and conclude the backend is still wrong. `d5_verify.sh`
(job 50453) therefore asks two different questions.

**Q1 — same-stream, shot-for-shot. Must be EXACT.** Twelve seeds × three
circuits, comparing the GPU seed against the CPU seed that reproduces the
identical 256-bit xoshiro state:

> **Q1: 36/36 exact.**

Every shot matches. This is the *logical* claim, and it is binary: either the
two backends compute the same thing or they do not.

**Q2 — statistical convergence.** Two *independent* streams sampling the same
distribution should converge as `1/√shots`:

| circuit | shots | cpu | gpu | rel gap | z |
|---|---|---|---|---|---|
| circuit_d5_p0.001 | 2,000 | 709 | 661 | 6.77 % | −1.60 |
| | 10,000 | 3,395 | 3,364 | 0.913 % | −0.46 |
| | 50,000 | 17,005 | 16,862 | 0.841 % | −0.96 |
| | 200,000 | 67,376 | 66,968 | 0.606 % | −1.37 |
| circuit_d3_p0.001 | 2,000 | 52 | 63 | 21.15 % | 1.04 |
| | 10,000 | 250 | 270 | 8.00 % | 0.89 |
| | 50,000 | 1,225 | 1,224 | 0.082 % | −0.02 |
| | 200,000 | 4,811 | 4,922 | 2.31 % | 1.14 |

**Every |z| < 1.7.** The residual gap is sampling noise between two independent
streams, not a numerical defect — and Q1 is what proves that: since
shot-for-shot agreement is exact, the aggregate difference *can only be* the
difference between two valid samples of the same distribution.

Note the `circuit_d3` row at 200,000 shots: the relative gap goes *up* (0.082 %
→ 2.31 %) while `|z|` stays near 1. Reading the percentage alone would suggest a
regression at scale. The z-score says otherwise, and the z-score is the right
statistic — the raw counts are small (≈4,800 of 200,000), so the relative gap is
dominated by Poisson noise.

<figure>
<img src="diagrams/q2-convergence.svg" alt="Statistical convergence of GPU vs CPU observable counts" width="100%">
<figcaption><b>Figure 12.2</b> — Q2 convergence. Relative gap shrinks roughly as
1/√shots while every z-score stays inside ±1.7. The two backends are sampling
the same distribution with different streams.</figcaption>
</figure>

### 12.5 What "narrowing the gap" actually meant

The framing this section inherited — f32 vs f64 — is not what was fixed, and
saying so plainly matters more than preserving the tidier story:

- The gap was **not** about precision in the *arithmetic*. No accumulation order
  was changed, no reduction was promoted to f64, no amplitude was widened.
- It was about a **threshold constant calibrated for one precision and left in
  place when the storage format changed** — whose effect was to make one side
  consume a random number the other side did not.
- The fix is one token. The diagnosis was the entire cost.

And the residual, genuine f32-vs-f64 effect predicted in §12.1 — branch
probabilities good to only ~1e-6 relative — is real but is *not* what the d5
divergence was. It remains a bounded source of rare divergence at high rank, and
nothing in this corpus has yet isolated an instance of it. Stated as an open
item rather than a solved one (§16).

---
