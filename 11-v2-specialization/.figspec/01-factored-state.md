# Figure 2.1 — `diagrams/factored-state.svg`

**Caption it must serve (already in the report, do not restate it in the SVG):**
"The factored state. Dormant qubits cost zero storage; the Pauli frame costs 2
bits per qubit; only the active subspace costs exponential memory. Rank growth
is driven by non-Clifford gates and undone by measurement."

**Title:** The Factored State — why rank, not qubit count, sets the cost
**Subtitle:** `|psi> = gamma · U_C · P · ( |phi>_A ⊗ |0>_D )` — src/clifft/svm/svm.h:92-103

## What to draw

Left two-thirds: the five factors of the equation as five labelled blocks,
sized *in proportion to what they actually store*. Right third: a small
"rank timeline" showing k moving up and down.

### DATA — the five factors (exact, from the report's table)

| symbol | what it is | storage |
|---|---|---|
| `gamma` | global scalar: phase + deferred normalization | one complex scalar |
| `U_C` | the Clifford part, tracked symbolically | implicit in the frame |
| `P` | a Pauli frame — one X bit and one Z bit per qubit | 2 × ceil(N/64) words |
| `\|phi>_A` | the **active** subspace: dense statevector over k qubits | 2^k complex amplitudes |
| `\|0>_D` | the **dormant** qubits, provably in the computational basis | nothing |

Draw `gamma` tiny, `U_C` as a dashed outline (it has no storage of its own),
`P` as a thin bit-strip, `|phi>_A` as the one large block (make it visibly
dominate — this is the whole point), and `|0>_D` as an empty/ghosted block
labelled "0 bytes". Colour `|phi>_A` with the accent `#e94560`; the others in
neutral greys with `P` in cyan `#53d8fb`.

### DATA — the rank rules (exact)

- A pure-Clifford circuit never grows k: every gate is a frame update, `2^k = 1`,
  simulation is polynomial.
- Each non-Clifford op (`T`, `ROT`, `EXPAND`) may promote a dormant qubit:
  `k → k+1`, **doubling** the dense array.
- Each measurement of an active qubit collapses it: `k → k-1`.

Draw this as a small staircase on the right: a horizontal axis labelled
"program order", a step up annotated `T / ROT / EXPAND → k+1 (array doubles)`
in orange, a step down annotated `measure → k−1` in green, and a dashed
horizontal line marking **peak rank** with the label
"peak_rank fixes both the footprint (2^peak_rank amplitudes) and the tier".

### Punchline band (bottom, accent)

"A 100-qubit Clifford circuit has k = 0. Cost is set by peak rank, not by N."
