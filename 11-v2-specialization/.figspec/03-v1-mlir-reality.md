# Figure 5.1 — `diagrams/v1-mlir-reality.svg`

**Caption it must serve:** "Intended (left): a domain dialect lowered
progressively through custom passes. Actual (right): direct `llvm`-dialect
emission, three stock passes, one of them a no-op. The middle levels were never
built."

**Title:** V1's MLIR — Intended vs Actual
**Subtitle:** the middle of the lowering pipeline was never built

## What to draw — a two-column comparison

### LEFT: "Intended" (draw greyed/ghosted, dashed borders — none of it exists)

A tall progressive-lowering ladder, top to bottom, each rung a dialect box with
a custom-pass arrow between:
`clifft dialect` → `linalg / affine` → `scf` → `gpu` → `llvm` → AMDGCN ISA
Label the arrows "custom lowering passes". Put a small "NEVER BUILT" stamp
across the middle three rungs in red `#cc2222`.

### RIGHT: "Actual" (solid, real)

`mlir_emit.cc` → **`llvm` dialect, emitted directly** → three stock passes →
`mlir-translate` → LLVM IR → AMDGCN ISA.

The three passes, in order, with their exact measured effect on `frame_h`
(from `lowering/v1_passes/frame_h/`):

### DATA — exact, do not alter

| pass | frame_h lines in → out | what it did |
|---|---|---|
| `canonicalize` | 947 → 736 | constant dedup: `llvm.mlir.constant` 240 → 59 |
| `cse` | 736 → 608 | redundant `shl`/`and`/`xor` elimination |
| `convert-func-to-llvm` | 608 → 608 | **PROVABLE NO-OP** |

The third pass must be visually flagged. The measured diff between
`1_cse.mlir` and `2_convert-func-to-llvm.mlir` is **two lines, and both are the
"IR Dump After" banner comment**. 608 lines in, 608 out, byte-identical apart
from the header. Reason: *there is no `func` dialect to convert, because the
emitter never produced any.*

Render that as a small inset code panel (monospace, dark `#0d0d1a` panel) with
a two-line diff:

```
- // -----// IR Dump After CSE (cse) //----- //
+ // -----// IR Dump After ConvertFuncToLLVMPass (convert-func-to-llvm) //----- //
```

with the `-` line in red and the `+` line in green, and a caption line
"the entire diff · 608 → 608 lines".

### Punchline band

"Two passes of cleanup after a sloppy emitter, and one that does nothing.
This is not compilation — it is recovery."
