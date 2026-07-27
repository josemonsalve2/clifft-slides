# Figure 6.2 — `diagrams/one-library-two-consumers.svg`

**Caption it must serve:** "The whole trick, in one signature. Every opcode body
is a `static inline` function taking its operands *and the pre-op `active_k`* by
value. The interpreter calls it with runtime values; the specializer emits the
same call with literals. The bodies are the same bodies — so byte-exactness is
by construction, not by testing. This is what retires V1's duplication tax: V1
had to reimplement each opcode in its emitter, and the two implementations could
drift."

**Title:** One Operand Library, Compiled Two Ways
**Subtitle:** Same semantics, different knowledge — the specializer changes the *arguments*, never the body

## What to draw

Canvas ~1080 × 600. A **Y shape inverted**: two consumers at the top converging
on one shared library in the middle, then splitting by tier at the bottom.

### CENTRE — the shared library (this is the hub, draw it prominently)

A box labelled **`v2_ops.h` + `v2_ops_body.inc`**, containing the two real
signatures in monospace:

```c
static inline void v2_op_array_cnot(V2State* st, CV2Complex* v,
                                    u32 active_k, u32 a1, u32 a2);
static inline void v2_op_meas_dormant_static(V2State* st, u32 axis,
                                             u32 slot, u32 flags);
```

Highlight **`u32 active_k`** in accent `#e94560` with a small label:
**"the pre-op rank, passed by value — this is what makes the loop bounds
foldable"**.

### TOP LEFT — the interpreter consumer (cyan `#53d8fb`)

Box: **`coop_interpreter.c`**, feeding a call with **runtime values**:

```c
v2_op_array_cnot(st, v, st->active_k, in.axis_1, in.axis_2)
```

Annotate each argument's origin: `st->active_k` ← a **load**;
`in.axis_1` / `in.axis_2` ← **fields of a 32-byte bytecode instruction**, loaded
every shot, every instruction.

### TOP RIGHT — the specializer consumer (accent `#e94560`)

Box: **`v2_specializer.cc`**, feeding the *same call* with **literals**:

```c
v2_op_array_cnot(st, v, 8u, 2u, 5u)
```

Annotate: **no load, no bytecode fetch — the values are in the instruction
stream.** Draw the input to this side as **bytecode + statically tracked rank**,
i.e. the specializer knows `active_k` at emit time.

Draw both call arrows converging into the **same** library box. Make it visually
obvious there is one body, not two. Label the convergence:
**"same body — byte-exact by construction"**.

### BOTTOM — the tier split (cooperation, not arithmetic)

From the library box, two arrows down into two compilations of the *same
source*:

- **`-DV2_REGISTER`** → `V2_STRIDE 1u`, `v2_tid() == 0`, `IS_OWNER 1`,
  barrier → nothing, `V2_REDUCE2` → identity
- **default (coop / global)** → `V2_STRIDE 256u`,
  `v2_tid() == workitem_id_x()`, `IS_OWNER (t == 0)`, fenced `s_barrier`,
  `coop_reduce2` butterfly

Label this split: **"parameterizes *cooperation*, not arithmetic
(`v2_ops.h:128-152`)"**.

### CALLOUT (green `#00cc66`)

> **Byte-exactness is by construction, not by testing.**
> V1 reimplemented each opcode inside its emitter, so the emitter's version and
> the interpreter's version could drift. V2 has one implementation and two
> call sites.

### DATA STRIP — exact

- The specializer handles **35 of the 41 opcodes**; unsupported opcodes fall
  back to the interpreter.
- `reg_circuit_d3`: **344** bytecode instructions produce exactly **344**
  `v2_op_*` calls in **383** C lines.
- V2 source density on the five nontrivial examples: **1.11, 1.27, 1.02, 1.00,
  1.01** lines/instr.

## DATA — verbatim from §6.2–§6.3, invent nothing

Both signatures and both call forms above are quoted verbatim from §6.2. Do not
alter a token, including `u32`, `CV2Complex*`, and the `8u, 2u, 5u` literals.

## Notes

- EXTENDS `three-loops.svg`, which explains *where* the loops live. This figure
  explains **source sharing and byte-exactness**, which that figure does not
  cover.
- Do not draw the specializer as "unrolling". The report is emphatic that V2
  emits **one call per instruction** (density 1.00), and that unrolling is what
  V1 did wrong. The literals are in the **arguments**, not in an expanded body.
