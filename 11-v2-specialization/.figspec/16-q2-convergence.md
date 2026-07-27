# Figure 12.2 — `diagrams/q2-convergence.svg`

**Caption it must serve:** "Q2 convergence. Relative gap shrinks roughly as
1/√shots while every z-score stays inside ±1.7. The two backends are sampling
the same distribution with different streams."

**Title:** Two Independent Streams, One Distribution
**Subtitle:** Q1: 36/36 shots byte-exact · Q2: every |z| < 1.7

## What to draw — two stacked panels sharing a log x-axis (shots: 2,000 → 200,000)

### TOP PANEL — relative gap, %, log-y

Two series, one per circuit, plus a dashed `1/√shots` reference curve.

### BOTTOM PANEL — z-score, linear y, centred on 0

Two series. Draw shaded bands at ±1 and ±2. Every point must land inside ±1.7.

### DATA — exact. Every point.

| circuit | shots | cpu | gpu | rel gap | z |
|---|---|---|---|---|---|
| circuit_d5_p0.001 | 2,000 | 709 | 661 | 6.77 % | −1.60 |
| circuit_d5_p0.001 | 10,000 | 3,395 | 3,364 | 0.913 % | −0.46 |
| circuit_d5_p0.001 | 50,000 | 17,005 | 16,862 | 0.841 % | −0.96 |
| circuit_d5_p0.001 | 200,000 | 67,376 | 66,968 | 0.606 % | −1.37 |
| circuit_d3_p0.001 | 2,000 | 52 | 63 | 21.15 % | 1.04 |
| circuit_d3_p0.001 | 10,000 | 250 | 270 | 8.00 % | 0.89 |
| circuit_d3_p0.001 | 50,000 | 1,225 | 1,224 | 0.082 % | −0.02 |
| circuit_d3_p0.001 | 200,000 | 4,811 | 4,922 | 2.31 % | 1.14 |

### The Q1 banner — must be prominent, it carries the logical claim

A band across the top: "**Q1: 36 / 36 shots exact.** Shot-for-shot byte
agreement. This is the *logical* claim and it is binary."

Then the inference, spelled out: "**because** Q1 is exact, the aggregate
difference in Q2 *can only be* the difference between two valid samples of the
same distribution."

### The required trap annotation

Circle the `circuit_d3` 200,000-shot point in **both** panels and annotate:

"The relative gap goes **up** (0.082 % → 2.31 %) while |z| stays near 1. Reading
the percentage alone would suggest a regression at scale. The z-score is the
right statistic — the raw counts are small (≈4,800 of 200,000), so the relative
gap is dominated by Poisson noise."

### Punchline band

"Every |z| < 1.7. The residual gap is sampling noise between two independent
streams, not a numerical defect."
