## 8. Progressive lowering: what the compiler does with it

Chapter 7 measured what specialization does to a *single* op in isolation. This
chapter follows a whole circuit down both pipelines, stage by stage, and shows
the actual IR at each boundary.

A note on framing, because the project's own documentation is misleading here
and the ground rule of this report is to trust data over text. **V2 does not
emit MLIR.** The directory is called `src/clifft/gpu/mlir/v2/` for historical
reasons, but `v2_specializer.cc:127` emits a C translation unit whose first line
is `#include "clifft/gpu/mlir/v2/v2_ops.h"`, and the path to `.hsaco` is
C → clang → llvm-link → opt → llc → ld.lld. The name is the only MLIR left in
V2. §5 established the corresponding fact about V1: its MLIR was 100 % `llvm`
dialect, it authored **zero** custom passes, and one of the three stock passes
it ran is a complete no-op.

So this chapter cannot show "the diff each MLIR pass we introduced produces,"
because no such pass exists. What it can show — and does — is the real
progressive lowering of both pipelines, rendered as diffs, which turns out to be
a more useful comparison anyway: it shows *where each pipeline's representation
actually collapses*, and V1's and V2's collapse in completely different places.

### 8.1 The corpus

Thirteen stage pairs are rendered, each as a hunk-scoped `.diff` (greppable, for
this report) and a colorized side-by-side `.html` (for the deck). All under
`V2_performance/lowering/diffs/`:

| Pipeline | Circuit | Chain |
|---|---|---|
| `v1` | `frame_h`, `circuit_d3` | `emitted.mlir` → `opt.mlir` → `translate.ll` → `optO2.ll` → `isa.s` |
| `v2` | `circuit_d3` | `emitted.c` → `clangO0.ll` → `clangO2.ll` → `isa.s` |
| `v1pass` | `frame_h` | `canonicalize` → `cse` → `convert-func-to-llvm` |

The renderer (`lowering/render_diffs.py`) documents its own methodology, and the
reasoning is worth reproducing because it governs how these diffs should be
read:

> Why not just `diff -u`: the stage files are up to 20k lines and the
> interesting change is almost never at the top. `--window` extracts the N
> largest contiguous change hunks so the report can quote a hunk that actually
> shows the transform, rather than the first one alphabetically.
>
> Stage files are **NOT line-aligned** across a lowering boundary (MLIR SSA
> names get renumbered by every pass), so a raw line diff between `.mlir` and
> `.ll` is noise. That is intentional: the diff is evidence of *how much* the
> representation changed, and the hunk excerpts show *what kind* of change.

Diff sizes, which are themselves a signal:

```
   16  v1pass.frame_h.1_cse__2_convert-func-to-llvm.diff     <- the no-op
  189  v1.circuit_d3.4_optO2__5_isa.diff
  189  v1.frame_h.4_optO2__5_isa.diff
  189  v2.circuit_d3.3_clangO2__4_isa.diff
  248  v1.circuit_d3.2_opt__3_translate.diff
  248  v1.frame_h.2_opt__3_translate.diff
  255  v1.circuit_d3.1_emitted__2_opt.diff
  255  v1.frame_h.1_emitted__2_opt.diff
  328  v2.circuit_d3.1_emitted__2_clangO0.diff
  404  v1.frame_h.3_translate__4_optO2.diff
  456  v2.circuit_d3.2_clangO0__3_clangO2.diff
  488  v1.circuit_d3.3_translate__4_optO2.diff
  576  v1pass.frame_h.0_canonicalize__1_cse.diff
```

### 8.2 Line counts through both pipelines

The most compact way to see the difference is the size of the representation at
each stage, for the same circuit (`circuit_d3`, 344 instructions, register tier):

| Stage | V1 | | Stage | V2 |
|---|---|---|---|---|
| `1_emitted.mlir` | **23,002** | | `1_emitted.c` | **383** |
| `2_opt.mlir` | 14,257 | | `2_clangO0.ll` | 19,551 |
| `3_translate.ll` | 14,875 | | `3_clangO2.ll` | 8,132 |
| `4_optO2.ll` | 13,747 | | | |
| `5_isa.s` | 17,802 | | `4_isa.s` | 10,398 |

And the small circuit, `frame_h` (4 instructions), on V1 only — V2 does not have
a rendered chain for it, but its emitted C is 43 lines:

| Stage | V1 `frame_h` |
|---|---|
| `1_emitted.mlir` | 947 |
| `2_opt.mlir` | 606 |
| `3_translate.ll` | 562 |
| `4_optO2.ll` | 275 |
| `5_isa.s` | 524 |

Two shapes, and they are opposites:

- **V1 starts huge and shrinks.** 23,002 lines of `llvm`-dialect MLIR for 344
  instructions. Every optimization stage is *recovering* from the emitter.
- **V2 starts tiny and expands, then collapses.** 383 lines of C → 19,551 lines
  of unoptimized IR (because `-O0` gives every local variable an `alloca` and
  every access an `addrspacecast`) → 8,132 lines at `-O2`. The expansion is a
  *compiler artifact of `-O0`*, not something the emitter wrote down, and `-O2`
  removes it entirely.

The endpoint is what matters: **17,802 vs 10,398 ISA lines for the same
circuit.** V2 produces a 1.7× smaller kernel from a 60× smaller source.

<figure>
<img src="diagrams/lowering-pipelines.svg" alt="V1 and V2 lowering pipelines with per-stage sizes" width="100%">
<figcaption><b>Figure 8.1</b> — The two pipelines with per-stage line counts for
<code>circuit_d3</code>. V1's representation is largest where a human wrote it;
V2's is largest where the compiler expanded it at <code>-O0</code> and smallest
again after <code>-O2</code>.</figcaption>
</figure>

---

### 8.3 V1's MLIR stage: what `canonicalize` and `cse` actually did

`mlir_codegen.cc:65-67` runs exactly three passes. Snapshots were taken with
`mlir-opt --mlir-print-ir-tree-dir` so each pass's output is a separate file
(`lowering/v1_passes/frame_h/`):

```
0_canonicalize.mlir   736 lines
1_cse.mlir            608 lines
2_convert-func-to-llvm.mlir  608 lines
```

(from an emitted 947.)

**`canonicalize` folded duplicate constants.** The op census for `frame_h`
before and after the MLIR stage:

| Op | emitted | after opt |
|---|---|---|
| `llvm.mlir.constant` | 240 | **59** |
| `llvm.ptr` | 192 | 160 |
| `llvm.and` | 54 | 9 |
| `llvm.shl` | 73 | 34 |
| `llvm.xor` | 68 | 29 |

The emitter materialized the same constant over and over — 240 constant ops for
a four-instruction circuit — and canonicalization deduplicated them down to 59.
This is not a sophisticated transform; it is cleanup after a naive emitter.

**`cse` deleted redundant loads and recomputation.** The
`0_canonicalize → 1_cse` diff is the largest in the corpus (576 lines), and the
hunk it selects is a PRNG update inside the shot loop that had been emitted
twice:

```diff
     %138 = llvm.load %45 : !llvm.ptr -> i64
-    %139 = llvm.getelementptr inbounds %45[1] : (!llvm.ptr) -> !llvm.ptr, i64
-    %140 = llvm.load %139 : !llvm.ptr -> i64
-    %141 = llvm.getelementptr inbounds %45[2] : (!llvm.ptr) -> !llvm.ptr, i64
-    %142 = llvm.load %141 : !llvm.ptr -> i64
-    %145 = llvm.add %138, %144 : i64
-    %146 = llvm.shl %145, %16 : i64
-    %147 = llvm.lshr %145, %15 : i64
-    %148 = llvm.or %146, %147 : i64        ; xoshiro256++ rotate
-    %151 = llvm.xor %142, %138 : i64
-    %152 = llvm.xor %144, %140 : i64
-    ...
-    %165 = llvm.lshr %33, %35 : i64        ; frame word index, computed
-    %166 = llvm.getelementptr inbounds %49[%165] : ...
-    %175 = llvm.lshr %33, %35 : i64        ; ...and computed AGAIN
-    %176 = llvm.getelementptr inbounds %51[%175] : ...
```

Note the last four lines: `llvm.lshr %33, %35` appears twice with identical
operands, once for `px` and once for `pz`. The emitter had no notion of a
common subexpression, so it wrote out the word-index computation separately for
each frame array. CSE removes it. This is exactly the class of redundancy that
**does not exist in V2's C**, because in V2 that computation is written once in
`fget` and inlined by clang after constant-folding the index away entirely
(§7.3).

**`convert-func-to-llvm` did nothing at all.** The full rendered diff is 16
lines, and the only change is the pass-name banner:

```diff
-// -----// IR Dump After CSE (cse) //----- //
+// -----// IR Dump After ConvertFuncToLLVMPass (convert-func-to-llvm) //----- //
 module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
   llvm.func @llvm.amdgcn.workitem.id.x() -> i32
   llvm.func @llvm.amdgcn.workgroup.id.x() -> i32
   llvm.func @llvm.amdgcn.s.barrier()
   llvm.func @llvm.amdgcn.ds.bpermute(i32, i32) -> i32
   llvm.func @clifft_log(%arg0: f64) -> f64 {
```

608 lines in, 608 lines out, byte-identical below the banner. There is no `func`
dialect to convert *because the emitter never produced any* — everything was
already `llvm.func`. This is the clearest single piece of evidence that V1 was
not using MLIR as a multi-level IR: the pass whose entire job is to lower a
higher abstraction level found nothing to lower.

> **What this means for the "MLIR passes" material.** The report's goal asked
> for progressive-lowering diffs of the passes introduced. The honest answer is
> that the value MLIR added to V1 was *two stock cleanup passes recovering from
> the emitter's own verbosity*. That is a real thing to show, and it is shown
> above — but it is not a compiler-engineering contribution, and V2 got the same
> cleanup for free by writing C and letting clang's front end never emit the
> redundancy in the first place.

---

### 8.4 V1: `translate` → `opt -O2`, where the real work happened

MLIR's contribution ended at 608 lines. The transform that actually mattered
happened in LLVM. `frame_h.3_translate.ll` (562) → `frame_h.4_optO2.ll` (275)
is a 2× collapse, and the hunk shows what:

```diff
-define amdgpu_kernel void @compiled_mlir_kernel(i64 inreg %0, ..., i32 inreg %9) {
-  %11 = alloca i32, align 4, addrspace(5)
-  %12 = addrspacecast ptr addrspace(5) %11 to ptr
-  %13 = alloca i64, i32 4, align 8, addrspace(5)
-  %14 = addrspacecast ptr addrspace(5) %13 to ptr
-  %15 = alloca i64, align 8, addrspace(5)
-  ...
-  %27 = alloca i8, i32 4096, align 1, addrspace(5)     ; <- 4 KB of scratch
-  %28 = addrspacecast ptr addrspace(5) %27 to ptr
-  %31 = call i32 @llvm.amdgcn.workitem.id.x()
+; Function Attrs: mustprogress nofree norecurse nounwind willreturn
+define amdgpu_kernel void @compiled_mlir_kernel(..., ptr noalias captures(none) %3,
+       ptr noalias readnone captures(none) %4, ...) local_unnamed_addr #4 {
+  %.global = addrspacecast ptr %3 to ptr addrspace(1)
+  %11 = tail call i32 @llvm.amdgcn.workitem.id.x()
```

Ten `alloca`/`addrspacecast` pairs — the emitter's model of local state as stack
slots — promoted to SSA registers by SROA/mem2reg, and the kernel gains
`nofree norecurse nounwind willreturn` plus `noalias readnone captures(none)`
on the pointer arguments. **LLVM did this, not MLIR.** The MLIR stage had the
same `alloca`s in front of it for two passes and left every one of them
standing.

For the larger circuit the collapse is far weaker: `circuit_d3` goes 14,875 →
13,747, only 7.6 %. That is the size-adaptive detuning from §5.3 starting to
bite — at 20 MB of IR, LLVM's own pass pipeline gets throttled
(`mlir_kernel_cache.cc:104-141`), and the very optimizations that rescued
`frame_h` stop being affordable.

The final stage, `optO2 → isa`, actually *grows* the file: 275 → 524 lines for
`frame_h`, 13,747 → 17,802 for `circuit_d3`. That is normal — one IR instruction
becomes several machine instructions plus directives — but for `circuit_d3` it
means V1 shipped 17,802 lines of ISA for a 344-instruction circuit.

---

### 8.5 V2: `emitted.c` → `clangO0` → `clangO2`

V2's chain has one interesting boundary and it is a very different one.

**`emitted.c` → `clangO0.ll`: 383 → 19,551 lines.** A 51× expansion, and the
diff shows exactly why:

```diff
-  %pauli_masks.addr.ascast = addrspacecast ptr addrspace(5) %pauli_masks.addr to ptr
-  %readout_noise.addr.ascast = addrspacecast ptr addrspace(5) %readout_noise.addr to ptr
-  %shot_id.ascast = addrspacecast ptr addrspace(5) %shot_id to ptr
-  store ptr %instrs, ptr %instrs.addr.ascast, align 8
-  store i32 %num_instrs, ptr %num_instrs.addr.ascast, align 4
-  ...
-  %0 = load i32, ptr %peak_rank.addr.ascast, align 4
-  %1 = load i32, ptr %num_instrs.addr.ascast, align 4
```

At `-O0` clang gives every parameter and local a stack slot in `addrspace(5)`,
an `addrspacecast` to generic, a store on entry, and a load at each use. For a
function with 20-odd parameters called 344 times' worth of inlined bodies, that
is thousands of ops. The op census at `2_clangO0` for `circuit_d3`:

| Op | count |
|---|---|
| `load` | 4,898 |
| `store` | 2,488 |
| **`alloca`** | **2,268** |
| **`addrspacecast`** | **2,259** |
| `call` | 1,770 |

**`clangO0.ll` → `clangO2.ll`: 19,551 → 8,132.** And the census after:

| Op | `-O0` | `-O2` |
|---|---|---|
| `alloca` | 2,268 | **0** |
| `addrspacecast` | 2,259 | **8** |
| `call` | 1,770 | 72 |
| `load` | 4,898 | 1,014 |
| `store` | 2,488 | 1,103 |
| `fmul` | 12 | **448** |
| `shufflevector` | 0 | **459** |

Read those last two rows carefully, because they are the whole point. `alloca`
goes to **zero** — every piece of shot-local state now lives in registers.
`call` collapses 1,770 → 72 as the `always_inline` op bodies are inlined (the
72 survivors are the `noinline` noise ops from `V2_NOISE_ATTR` plus ocml calls).
And `fmul` *rises* from 12 to 448 while `shufflevector` appears from nothing:
that is the SLP vectorizer packing the complex arithmetic into `<2 x float>`
operations. In the final ISA those show up as:

```
v_pk_add_f32   431
v_pk_mul_f32   416
```

**Packed** f32 math — two complex components per instruction. This is a
throughput win that only becomes available *because* the amplitudes ended up in
registers rather than in a stack array, which in turn is only possible because
the specializer wrote a straight-line body with statically-known indices.

The `-O0` → `-O2` boundary is therefore doing the same job MLIR's
canonicalize/cse did for V1 — plus SROA, plus inlining, plus vectorization —
and doing it far more thoroughly, because clang's IR was never asked to survive
a hand-written emitter's idea of structure.

**`clangO2.ll` → `isa.s`: 8,132 → 10,398.** Instruction selection and register
allocation. The top of this file is the one place V2's own cost shows:

```asm
v2_op_noise_block:                      ; @v2_op_noise_block
	s_mov_b32 s33, s32
	scratch_store_dword off, v63, s33 offset:68 ; 4-byte Folded Spill
	scratch_store_dword off, v40, s33 offset:64 ; 4-byte Folded Spill
	scratch_store_dword off, v41, s33 offset:60 ; 4-byte Folded Spill
	... 15 more spills ...
```

`v2_op_noise_block` is emitted `noinline` (§7.9, §11.1), so it is a real
function with a real prologue, and a callee-saved register spill sequence. That
is the price of the FP-scheduling fence. It shows up in the `circuit_d3` ISA
census as 406 `scratch_store_dwordx2` + 303 `scratch_store_dwordx4`, and it is
one of the two reasons V2's register-tier ScratchSize is 928 bytes where V1's
was 156 (the other being the shot-packed register layout of §9.1). §11 returns
to whether that trade is worth it — the short answer, from the gate results, is
that it is, because the alternative is a correctness failure rather than a
slower kernel.

---

### 8.6 What the two pipelines' diffs collectively show

| | V1 | V2 |
|---|---|---|
| Where the representation is largest | at emission (23,002 lines, human-authored) | at `-O0` (19,551 lines, compiler-authored, thrown away) |
| What the "MLIR stage" contributed | constant dedup + CSE; one pass a no-op | n/a — no MLIR |
| Where `alloca`s die | LLVM `-O2`, and only on small circuits | clang `-O2`, completely (2,268 → 0) |
| Whether the optimizer stays enabled | no — detuned by IR size above threshold | yes — IR never gets large enough to matter |
| Vectorization | none observed | 459 `shufflevector` → 431 `v_pk_add_f32`, 416 `v_pk_mul_f32` |
| Final ISA, `circuit_d3` | 17,802 lines | **10,398 lines** |
| Compile time, `circuit_d3` | 3.58 s | **2.04 s** |
| Compile time, `surface_d7_t19` | **221.48 s** | (V2 emits the same circuit in ~4,346 C lines; see §9) |

The single sentence version: **V1 asked the compiler to clean up after the
emitter, and past a certain size the compiler declined. V2 gave the compiler
something it was already good at.**

---
