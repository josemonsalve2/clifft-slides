# Figure 2.2 — `diagrams/bytecode-layout.svg`

**Caption it must serve:** "The 32-byte instruction. Every field in the fixed
header, and the payload's discriminant, is known to the ahead-of-time compiler.
This is the entire raw material available to a specializer."

**Title:** The 32-Byte Instruction — everything a specializer can see
**Subtitle:** src/clifft/backend/backend.h · 41 opcodes · static_assert(sizeof(Instruction) == 32)

## What to draw

A byte-ruler across the top (0…31), then the fixed 8-byte header drawn as
labelled byte cells, then the seven payload variants stacked as alternative
interpretations of bytes 8..31 (a classic union diagram: one row per variant,
all aligned to the same byte columns).

### DATA — the fixed header (exact offsets)

| offset | field |
|---|---|
| 0 | `opcode` (uint8_t) |
| 1 | `flags` |
| 2 | (field at offset 2) |
| 3 | `_pad` |
| 4 | `axis_1` (uint16_t) — virtual axis (target/control) |
| 6 | `axis_2` (uint16_t) — virtual axis 2 |

### DATA — the seven payload variants, bytes 8..31 (exact, verbatim)

| tag | member | fields |
|---|---|---|
| A | `math` | `double weight_re, weight_im` |
| B | `classical` | `uint32_t classical_idx, expected_val` |
| C | `pauli` | `uint32_t cp_mask_idx, condition_idx` |
| D | `multi_gate` | `uint64_t mask` |
| E | `u2` | `uint32_t cp_idx` → `fused_u2_nodes` |
| F | `u4` | `uint32_t cp_idx` → `fused_u4_nodes` |
| G | `exp_val` | `uint32_t cp_exp_val_idx, exp_val_idx` |
| — | `raw[24]` | full payload access |

Each variant is padded out to the full 24 bytes; show the padding as a hatched
or dimmed remainder so the union's shape is obvious.

### The argument the figure must make

Variants **E**, **F** and **G** are *indices*, not data. Draw an arrow from
those three rows out to a separate box on the right labelled `ConstantPool`
(device buffer) holding "fused 2×2 / 4×4 unitaries, N-bit Pauli masks,
noise-channel tables". Annotate that arrow, in cyan, with the exact
specialization limit:

"the **index** folds to an immediate; the **contents** stay in a device buffer"

Colour the fixed header green (fully foldable) and the indirection arrow red.

### Punchline band

"Everything left of byte 8 is a compile-time constant. §7's S6 and S8 measure
exactly what the right-hand side costs."
