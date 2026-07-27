# Figure 6.1 — `diagrams/three-loops.svg`

**Caption it must serve:** "The three nested loops. V1 flattened the middle one
into code. V2 replaces it with a call sequence whose *operands* are constants
but whose *bodies* are shared and still contain their own loops."

**Title:** The Three Loops — and the One Rule for Each
**Subtitle:** docs/v2/V2.md:104-132 §3.1 · unrolling the wrong loop is what killed V1

## What to draw

Concentric / nested loop boxes, outermost to innermost, each with its rule
badge and its justification. Then a right-hand panel contrasting V1's and V2's
treatment of the middle loop.

### DATA — the four rules, verbatim from the report table

| loop | rule | why |
|---|---|---|
| **over shots** | **ALWAYS a runtime loop** | shots are data; a persistent kernel with work-stealing pulls them |
| **over the bytecode** | **ALWAYS a runtime loop\*** | the operand sequence is data; unrolling it is the disease |
| **over amplitude groups** | **stays a loop**, but bounds/constants may specialize | at rank 10 that is 256–512 groups; flattening reinstates the bloat |
| **the 2×2 / 4×4 butterfly** | **fully unrolled** | a constant 4–16 multiply-adds; no size risk |

Badge colours: "runtime loop" = green, "loop with folded bounds" = orange,
"fully unrolled" = accent `#e94560`.

### The asterisk — this is the figure's real content

V2 does not *unroll* the bytecode loop. It **removes** it, by emitting one
**call** per instruction instead of one **block**. Draw this as a three-way
comparison strip along the bottom, in monospace code panels:

**Interpreter** (loop + dispatch):
```c
for (pc = 0; pc < n; ++pc)
    switch (instr[pc].opcode) { ... 41 arms ... }
```
label: "O(1) code, runtime dispatch"

**V1** (unrolled blocks) — draw as many small stacked blocks, visibly
overflowing its panel:
```c
/* instruction 0 */  ...block...
/* instruction 1 */  ...block...
/* × 336,988 lines */
```
label in red: "O(n × body) code — the disease"

**V2** (call sequence):
```c
v2_op_frame_h(st, 3u);
v2_op_array_cnot(st, v, 1u, 2u, 4u);
v2_op_meas_active(st, v, 0u, 5u);
```
label in accent: "O(n) code · one call site per instruction · bodies shared"

Annotate the key sentence: **"a call site is O(1) IR regardless of what the
callee does."**

### Punchline band

"The bodies still contain their own loops. Only the operands became constants."
