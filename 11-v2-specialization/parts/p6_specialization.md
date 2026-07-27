## 7. What V2 specializes — eight classes, eight experiments

Chapter 6 described the mechanism: the specializer emits one `v2_op_*()` call
per bytecode instruction, with the operands and the statically-tracked
`active_k` written out as C literals. This chapter answers the obvious follow-up
question — *what does that actually buy, per opcode family?* — and it answers it
with measurements rather than with reasoning about what the compiler ought to
do.

### 7.1 The experiment

Eight cases, one per specialization class. Each case is a single `.c` file that
compiles **twice from the same source**, calling the **identical** `v2_op_*`
body out of `v2_ops.h`:

| Build | Meaning |
|---|---|
| `-DSPEC_FORM=0` | **interpreter form** — operands arrive in a `CV2Instr` loaded from device memory at a runtime `pc`, exactly as `coop_interpreter.c`'s `for(pc) switch` supplies them |
| `-DSPEC_FORM=1` | **specialized form** — operands are the literal constants `v2_specializer.cc` emits for that instruction |

The two forms differ in **exactly one variable**: whether the operands were
compile-time constants. Everything else — the op body, the tier macro, the
compiler, the flags — is held fixed. The flags are the production flags copied
out of `v2_compile_cache.cc`:

```
clang --target=amdgcn-amd-amdhsa -mcpu=gfx950 -ffreestanding -nostdlib -nogpulib \
      -std=c23 -O2 -ffp-contract=off -I<root>/src -S
```

One methodology point deserves quoting, because getting it wrong would have
inflated every number in this chapter. From `spec_examples/harness.h`:

> The interpreter form reads its operands out of `instrs`, a plain
> `const CV2Instr*` kernel argument indexed by a runtime `pc`, which is exactly
> how `coop_interpreter.c`'s `for(pc)switch` supplies them. **NOT volatile:**
> marking it volatile would force system-coherent (`sc0 sc1`) reads the real
> interpreter never issues and would overstate the interpreter's cost. The
> operands are unknowable at compile time simply because `instrs` points at
> device memory.

So the interpreter form is not a strawman. It is given every optimization the
real interpreter gets; the only thing withheld is knowledge of the operand
values. The harness kernel signature is the same in both builds:

```c
#define CASE_KERNEL_HEAD                                                    \
    __attribute__((visibility("default"))) __attribute__((amdgpu_kernel))   \
    void case_kernel(V2State* st, CV2Complex* v, CV2Complex* scratch,       \
                     const CV2Instr* instrs, u32 pc,                        \
                     const CV2FusedU2Entry* fused_u2, ...)

#define IN(n) (instrs[pc + (n)])   // fetch instruction pc+n, the interpreter's access pattern
```

Metrics are scraped from the emitted assembly by `build_examples.sh`:
instruction count, `num_vgpr`/`numbered_sgpr`/`private_seg_size` from the
AMDGPU `.set` directives, and counts of `s_load`, `ds_read`/`ds_write`, `v_*`,
and `s_cbranch`/`s_branch`. Raw output is in
`V2_performance/lowering/spec_examples/stats.csv`; the `.s` files for both forms
of all eight cases are checked in alongside it, and `build_examples.sh`
reproduces all sixteen byte-for-byte (`ARCH` defaults to `gfx950`; the
register-tier cases additionally get `-DV2_REGISTER=1`, which is what selects
`V2_STRIDE 1` over `V2_STRIDE 256`).

### 7.2 A measurement note, because an earlier draft of this chapter got it wrong

The first version of the harness reported an *ISA line count* — `wc -l` of the
`.s` file. That number is dominated by assembler directives, comments, labels
and the AMDGPU metadata block: the non-instruction remainder is 199–316 lines
per case, which for these small kernels is 41–82 % of the file. Because that
overhead is nearly constant across the two forms — it differs by 4–23 lines
where the instruction counts differ by up to 126 — it diluted every ratio
toward 1.0 and systematically *understated* the effect this chapter is about.
S1 read 1.37× where the instruction ratio is 2.89×; every case moved the same
direction:

| Case | line ratio (wrong) | instruction ratio (right) |
|---|---|---|
| S1 | 1.37× | 2.89× |
| S2 | 1.59× | 2.94× |
| S3 | 1.30× | 1.98× |
| S4 | 1.23× | 1.38× |
| S5 | 1.44× | 2.56× |
| S6 | 1.26× | 1.71× |
| S7 | 1.07× | 1.10× |
| S8 | 1.14× | 1.35× |

The table below counts instructions — tab-indented lines that are neither
`.directive` nor `; comment`. Two further bugs are fixed in the same pass: the
resource columns were scraped from the `; NumVgprs:` comment, which LLVM emits
as a *symbolic expression* (`max(56, amdgpu.max_num_vgpr)`) for any kernel that
calls an external function — so S7, which calls `__ocml_log_f64`, scraped to
**0** and appeared to use no vector registers at all, when it in fact uses 56 in
both forms — and `; NumSgprs:` does not appear in this LLVM's output at all
(only `; TotalNumSgprs:`, itself symbolic), so that column read 0 for all
sixteen files. All three are read from the `.set` directives now.

A third, subtler bug is documented in `build_examples.sh:38-42` and is worth
repeating because it is the kind that produces a *plausible* wrong answer:

> A literal tab, not the escape: `/usr/bin/grep` here is ugrep, which does not
> expand `\t` inside an ERE when the script runs non-interactively (it does when
> typed at a prompt — so this silently counted 0 in batch and the right number
> by hand).

An instruction count that reads 0 in the CSV and the correct value when checked
by hand at a prompt is worse than one that is simply wrong.

```
S1_frame_cnot    instrs  136->47   (2.89x)  vgpr   8->3    v_alu   44->8    (5.50x)  branch   6->4
S2_meas_dormant  instrs  206->70   (2.94x)  vgpr  25->14   v_alu   86->19   (4.53x)  branch  18->0
S3_expand_rank   instrs  164->83   (1.98x)  vgpr  18->8    v_alu   67->27   (2.48x)  branch   9->6
S4_meas_active   instrs  331->240  (1.38x)  vgpr  24->14   v_alu  139->96   (1.45x)  branch  15->7
S5_array_cnot    instrs  207->81   (2.56x)  vgpr  16->8    v_alu   84->33   (2.55x)  branch  10->8
S6_array_u2      instrs  147->86   (1.71x)  vgpr  24->18   v_alu   54->42   (1.29x)  branch  10->2
S7_noise_block   instrs  447->407  (1.10x)  vgpr  56->56   v_alu  214->210  (1.02x)  branch  23->20
S8_apply_pauli   instrs  185->137  (1.35x)  vgpr  22->21   v_alu   71->63   (1.13x)  branch   3->3
```

With the scalar-memory and register columns, which is where the story actually
is:

| Case | Class | Tier | instrs | VALU | VGPR | SGPR | `s_load` | `ds_*` | branch |
|---|---|---|---|---|---|---|---|---|---|
| S1 | frame operand folding | register | 136→47 (2.89×) | 44→8 (5.50×) | 8→3 | 14→12 | 6→3 | 0→0 | 6→4 |
| S2 | flag folding | register | 206→70 (2.94×) | 86→19 (4.53×) | 25→14 | 13→26 | 9→4 | 0→0 | **18→0** |
| S3 | static rank tracking | coop | 164→83 (1.98×) | 67→27 (2.48×) | 18→8 | 16→10 | 6→4 | 0→0 | 9→6 |
| S4 | rank-folded reduction | coop | 331→240 (1.38×) | 139→96 (1.45×) | 24→14 | 30→26 | 7→5 | **4→4** | 15→7 |
| S5 | scatter-index folding | coop | 207→81 (2.56×) | 84→33 (2.55×) | 16→8 | 24→10 | **4→1** | 0→0 | 10→8 |
| S6 | fused-matrix lookup | coop | 147→86 (1.71×) | 54→42 (1.29×) | 24→18 | 23→16 | 12→7 | 0→0 | 10→2 |
| S7 | noise runtime loop | register | 447→407 (1.10×) | 214→210 (1.02×) | 56→56 | 86→85 | 9→5 | 0→0 | 23→20 |
| S8 | Pauli-mask index | register | 185→137 (1.35×) | 71→63 (1.13×) | 22→21 | 44→42 | 17→11 | 0→0 | 3→3 |

Four things fall out of this table immediately, and they set up the rest of the
report:

1. **The VALU ratios still exceed the instruction ratios, but the gap is now
   honest.** S1 deletes 82 % of its vector ALU work and 65 % of its
   instructions. What replaces the deleted VALU is *scalar* work — the SGPR
   column is the tell, and §7.3 and §14.3 develop it.
2. **The spread is enormous — 5.50× down to 1.02× on VALU.** Specialization is
   not a uniform multiplier. It is worth a great deal on frame/flag/index ops
   and almost nothing on data-dependent ops. Which opcodes a circuit is made of
   therefore predicts its speedup, and §14 confirms that it does.
3. **S7 is the negative result, and it is in the table on purpose.** 1.10×
   instructions, 1.02× VALU, and — now that the column is read correctly —
   **VGPR 56→56, unchanged**. The noise block is genuinely data-dependent; there
   is nothing to fold, and specialization does not even relieve register
   pressure. This is the case that predicts the `circuit_d5` regression in §11.1,
   and the corrected register number strengthens that prediction rather than
   weakening it.
4. **S2 is the one case where SGPR pressure goes *up*, 13→26.** That is the
   scalar-substitution effect made visible: the work removed from the vector pipe
   did not vanish, it moved, and it now lives in scalar registers. On the
   register tier this is a good trade — SGPRs are not the occupancy limiter there
   — but it is the mechanism, not a free lunch.

<figure>
<img src="diagrams/spec-classes-gains.svg" alt="Per-class specialization gains" width="100%">
<figcaption><b>Figure 7.1</b> — Per-class specialization gains, instructions vs
VALU, ordered by VALU ratio. The gap between the two bars is the
scalar-substitution effect: work is not only deleted, it is moved off the vector
pipe — most visibly in S2, whose SGPR count doubles as its branch count goes to
zero. S7 (noise) is flat on both, and flat on registers too.
</figcaption>
</figure>

The remaining sections take each class in turn: the op body, the two call
forms, and what the assembly diff shows.

---

### 7.3 S1 — frame operand folding

**The op.** `OP_FRAME_CNOT` touches only the Pauli frame. Two bit reads, two bit
XORs, at word/bit offsets derived from the axes (`v2_ops_body.inc:12-19`):

```c
static inline __attribute__((always_inline)) void v2_op_frame_cnot(V2State* st, u32 ctrl, u32 tgt) {
    u32 t = v2_tid();
    if (IS_OWNER) {
        int px_c = fget(st->px, ctrl), pz_tt = fget(st->pz, tgt);
        fxor(st->px, tgt, px_c); fxor(st->pz, ctrl, pz_tt);
    }
    v2_barrier();
}
```

`fget`/`fxor` are `(w[a >> 6] >> (a & 63)) & 1` style bit accessors. With a
**constant** axis, `a >> 6` and `a & 63` are compile-time; the word index
becomes a fixed offset and the bit becomes a fixed mask. With a **runtime** axis
each access costs a shift/mask chain *plus a dynamic index* into `st->px` /
`st->pz`.

**The two forms** (`cases/S1_frame_cnot.c`):

```c
#if SPEC_FORM
    v2_op_frame_cnot(st, 3u, 10u);
    v2_op_frame_cnot(st, 3u, 12u);
    v2_op_frame_cz(st, 3u, 6u);
#else
    CV2Instr i0 = IN(0), i1 = IN(1), i2 = IN(2);
    v2_op_frame_cnot(st, i0.axis_1, i0.axis_2);
    v2_op_frame_cnot(st, i1.axis_1, i1.axis_2);
    v2_op_frame_cz(st, i2.axis_1, i2.axis_2);
#endif
```

**What the assembly shows.** This is the single clearest diff in the whole
study, so it is worth reading both sides. The specialized form, in full:

```asm
case_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	s_mov_b32   s7, 0
	s_waitcnt   lgkmcnt(0)
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[2:3], s[0:1], 0x28
	s_waitcnt   lgkmcnt(0)
	s_and_b32   s6, s4, 8               ; fget(px, 3)  -> a constant mask
	s_cmp_eq_u64 s[6:7], 0
	s_cbranch_scc1 .LBB0_2
	s_xor_b32   s4, s4, 0x400           ; fxor(px, 10) -> a constant mask
	v_mov_b32_e32 v2, 0
	v_mov_b64_e32 v[0:1], s[4:5]
	global_store_dwordx2 v2, v[0:1], s[0:1]
	s_and_b32   s6, s2, 0x400
	...
```

Every frame access is `s_and_b32` / `s_xor_b32` against a **literal mask** on
the **scalar** unit. The full set of masks in the specialized kernel is `8`,
`0x400`, `0x1000` and `64` — respectively bit 3 (`ctrl=3`), bit 10 and bit 12
(the two CNOT targets) and bit 6 (the CZ target), each appearing exactly where
the source says it should.

Three CNOT/CZ ops compile to **8 VALU instructions total, and all 8 are `v_mov`**
— four `v_mov_b32_e32 v2, 0` supplying the zero address offset and four
`v_mov_b64_e32 v[0:1], s[..]` staging a scalar result into a vector register for
`global_store`. Not one arithmetic vector instruction survives. The frame logic
itself has left the vector pipe entirely; what remains is purely the
scalar→vector staging the store instruction's encoding requires.

The interpreter form for the identical three ops. The instruction fetch
(`interp/S1_frame_cnot.s:9-18`):

```asm
	s_load_dword s6, s[0:1], 0x20       ; pc
	s_load_dwordx2 s[2:3], s[0:1], 0x18 ; instrs
	v_mov_b32_e32 v3, 0
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s7, s6, 40                ; pc * sizeof(CV2Instr) -- address math
	s_mul_hi_u32 s5, s6, 40
	s_add_u32 s4, s2, s7
	s_addc_u32 s5, s3, s5
	global_load_dword v0, v3, s[4:5] offset:2   ; load the instruction
```

The `v_readfirstlane_b32 s9, v0` that moves the loaded operand to the scalar
unit follows five lines later, at `:23`, after an `s_waitcnt vmcnt(0)` that
stalls on the load. Then, for each frame access (`:34-46`, with three
interleaved scalar instructions for the *second* frame word elided):

```asm
	v_lshrrev_b32_e32 v2, 3, v0         ; a >> 6 -> word index, DYNAMIC
	v_and_b32_e32 v2, 0x1ff8, v2
	v_readfirstlane_b32 s2, v2
	s_load_dwordx2 s[6:7], s[0:1], s2 offset:0x0  ; dynamically-indexed frame word
	v_lshlrev_b64 v[0:1], v0, 1         ; 1 << (a & 63) -> bit mask, DYNAMIC
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v4, s6, v0
	v_and_b32_e32 v5, s7, v1
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]
```

Note the two `s_waitcnt`s. They are not incidental: the interpreter's operand
arrives from memory, so the dependent address math cannot issue until the load
retires, and the frame word it then addresses is itself a second dependent load.
The specialized form has no such chain — its masks are immediates in the
instruction encoding. It contains **2** `s_waitcnt`s against the interpreter
form's **15**, and both of the two guard the unavoidable two-step
kernel-argument load (kernarg → `st`, then `st` → the frame words).

Three separate costs are visible and all three disappear under specialization:

1. **Instruction fetch** — `s_mul_i32 s7, s6, 40` plus a `global_load_dword`,
   per instruction. The bytecode itself is a memory operand. (`40` is
   `sizeof(CV2Instr)`, which `device_abi_checks.cc:58` pins equal to
   `sizeof(GpuInstr)`; the multiply is `pc * sizeof(instr)` with a runtime `pc`,
   so it needs a 64-bit product — hence the `s_mul_i32`/`s_mul_hi_u32` pair and
   the `s_add_u32`/`s_addc_u32` that follow.)
2. **`v_readfirstlane_b32`** — the operand arrives in a *vector* register
   (it came from a `global_load` whose address is uniform but whose result the
   compiler must materialize in a VGPR), so it must be moved to the scalar unit
   before it can be used as an address. That is a cross-pipe transfer per
   operand.
3. **Dynamic bit math on the vector pipe** — `v_lshrrev_b32` / `v_and_b32` /
   `v_lshlrev_b64` to compute the word index and the bit mask. Under
   specialization these are *literals in the instruction encoding*.

The 44 → 8 VALU drop (5.50×) is the sum of (2) and (3), and the whole kernel
falls 136 → 47 instructions (2.89×) — the largest instruction-level gain in the
study. Note also VGPR 8 → 3: with the operands and masks resolved, the op needs
almost no vector registers, which directly raises tier occupancy (§9). The
scalar side barely moves, 14 → 12 SGPRs, because the *same* frame words are
still being read and XORed — they are simply addressed by literal masks now
instead of by computed ones.

> **This is the mechanism behind the SALU finding in §14.3.** V2's speedup is
> not primarily "fewer instructions." It is *the same computation moved from the
> vector pipe to the scalar pipe*, where it runs once per wavefront instead of
> once per lane, in parallel with vector work.

---

### 7.4 S2 — flag folding on dormant measurements

**The op.** `OP_MEAS_DORMANT_STATIC` writes a measurement slot without touching
the amplitude array. `flags` selects between a constant-outcome path
(`FLAG_IDENTITY`) and a frame-read path, and supplies the sign XOR
(`v2_ops_body.inc:53-65`):

```c
static inline __attribute__((always_inline)) void v2_op_meas_dormant_static(V2State* st, u32 axis, u32 slot, u8 flags) {
    u32 t = v2_tid();
    if (IS_OWNER) {
        u8 mval;
        if (flags & FLAG_IDENTITY) mval = (flags & FLAG_SIGN) ? 1u : 0u;
        else {
            u8 outcome = (u8)fget(st->px, axis);
            mval = outcome ^ (u8)((flags & FLAG_SIGN) != 0);
        }
        if (slot < V2_MAX_MEAS) mset(st->meas, slot, mval);
    }
    v2_barrier();
}
```

**The two forms** (`cases/S2_meas_dormant.c`):

```c
#if SPEC_FORM
    v2_op_meas_dormant_static(st, 6u, 33u, 0u);
    v2_op_meas_dormant_static(st, 8u, 32u, 1u);
    v2_op_meas_dormant_random(st, 13u, 30u, 0u);
#else
    CV2Instr i0 = IN(0), i1 = IN(1), i2 = IN(2);
    v2_op_meas_dormant_static(st, i0.axis_1, i0.a, i0.flags);
    ...
#endif
```

**Result: 18 branches → 0.** This is the cleanest positive in the study. A
constant `flags` deletes one of the two paths *outright* — not predicts it,
deletes it — along with the test that chose between them. A constant `slot`
deletes the `slot < V2_MAX_MEAS` bounds check. And `mset(st->meas, slot, ...)`
on a bit-packed array becomes a fixed word and a fixed mask, exactly as in S1.

The specialized assembly contains **no `s_cbranch` or `s_branch` at all** —
not merely none in the measurement logic; a `grep` over the whole file returns
zero, against 18 in the interpreter form. What remains is a straight scalar
sequence. Even the `rng_uniform` compare inside the `_random` variant, which
is genuinely data-dependent, is if-converted (`spec/S2_meas_dormant.s:49-65`,
with the intervening stores and `v_mov` staging elided):

```asm
	v_cvt_f64_u32_e32 v[8:9], s11       ; xoshiro output hi -> double
	v_cvt_f64_u32_e32 v[10:11], s10     ; ... and lo
	v_ldexp_f64  v[0:1], v[8:9], 32     ; hi * 2^32
	v_add_f64    v[0:1], v[0:1], v[10:11]
	v_ldexp_f64  v[0:1], v[0:1], s18    ; * 2^-53
	v_cmp_gt_f64_e32 vcc, 0.5, v[0:1]   ; rng_uniform(st->rng) < 0.5
	s_and_b64    s[2:3], vcc, exec
	s_cselect_b32 s18, 0, 0x2000        ; ... resolved with a SELECT, not a branch
```

Note the `s_cselect_b32`: even the genuinely random outcome is if-converted,
because with constant `axis`/`slot` both arms write the same fixed bit and the
compiler can pick the *value* rather than the *path*. The neighbouring
`s_bitset0_b32 s12, 13` is the same fold in its purest form — `mset` of a
constant slot on a bit-packed array is one instruction with the bit number
in the encoding.

VALU falls 86 → 19 (4.53×) and the kernel as a whole 206 → 70 instructions.
The **2.94× is the largest instruction ratio of the eight**, though S1's 5.50×
remains the largest VALU ratio.

This is also the case where **SGPR usage doubles, 13 → 26** — the only such case
in the study. Nothing was lost: the eighteen deleted branches and the vector work
behind them were replaced by scalar selects and the literals they select between,
all of which need scalar registers. It is worth stating plainly because it is the
clearest single instance of the pattern §14.3 generalizes — specialization does
not simply *delete* work, it **moves work from the vector pipe to the scalar
pipe**, and the scalar pipe is not free either. It is just very much cheaper: one
lane instead of sixty-four, and it issues in parallel with vector instructions.

Why this matters more on a GPU than the raw count suggests: on AMDGCN a taken
`s_cbranch` inside a divergent region forces `s_and_saveexec_b64` /
`s_or_b64 exec` mask manipulation and serializes the two sides. Eighteen of
those in a three-instruction sequence is pure interpretive overhead — and
dormant measurements are extremely common in surface-code circuits, which is
why the surface family shows some of the largest gains in §14.

---

### 7.5 S3 — static rank tracking (the load-bearing one)

The case file's own header comment states the claim:

> S3 — **STATIC RANK TRACKING.** This is the load-bearing specialization: the
> specializer knows `active_k` at every program point, so `half = 1u << active_k`
> becomes a literal and the strided sweep gets a compile-time trip count. The
> interpreter must re-read `st->active_k` after every rank-changing op.
> **Note what is NOT happening: the sweep is still a LOOP. V1 unrolled it.**

That last sentence is the entire difference between V1 and V2 in one line, and
§5 already showed what unrolling cost.

**How the specializer knows.** It maintains `k` as it walks the program
(`v2_specializer.cc:26-56`) — `++*k` on the expansion family, `--*k` on
rank-reducing measurements:

```cpp
case Opcode::OP_EXPAND:
    o << "v2_op_expand(st, v, " << *k << "u);"; ++*k; break;
case Opcode::OP_EXPAND_T:
    o << "v2_op_expand_t(st, v, " << *k << "u, " << a1 << "u, 0);"; ++*k; break;
case Opcode::OP_MEAS_ACTIVE_INTERFERE:
    o << "v2_op_meas_active_interfere(st, v, " << *k << "u, " << a1 << "u, "
      << a << "u, " << flags << "u);";
    if (*k) --*k; break;
```

Compare the interpreter, which has to re-read it every iteration
(`coop_interpreter.c:50-53`):

```c
for (u32 pc = 0; pc < num_instrs; ++pc) {
    CV2Instr ins = instrs[pc];
    u32 k = st->active_k;          // <- a LOAD, every instruction
    switch (ins.opcode) {
```

**The op** (`v2_ops_body.inc:85-96`):

```c
static inline __attribute__((always_inline)) void v2_op_expand_t(V2State* st, CV2Complex* v, u32 active_k, u32 axis, int dagger) {
    u32 t = v2_tid();
    u32 half = 1u << active_k;
    int px = fget(st->px, axis);
    double imag = dagger ? -V2_INV_SQRT2 : V2_INV_SQRT2;
    if (px) imag = -imag;
    CV2Complex phase; phase.re = (float)V2_INV_SQRT2; phase.im = (float)imag;
    for (u32 i = t; i < half; i += V2_STRIDE) v[i + half] = cmul(v[i], phase);
    v2_barrier();
    if (IS_OWNER) st->active_k = active_k + 1;
    v2_barrier();
}
```

**The two forms** (`cases/S3_expand_rank.c`) — note that only the *third*
argument changes:

```c
#if SPEC_FORM
    v2_op_expand_t(st, v, 0u, 0u, 1);
    v2_op_expand_t(st, v, 1u, 3u, 0);
    v2_op_expand(st, v, 2u);
#else
    v2_op_expand_t(st, v, st->active_k, i0.axis_1, 1);
    v2_op_expand_t(st, v, st->active_k, i1.axis_1, 0);
    v2_op_expand(st, v, st->active_k);
#endif
```

Because the specializer tracks the rank, the second call gets `1u` and the third
gets `2u` — the compiler sees the rank *grow*, statically, across the program.

**What it buys:** 164→83 instructions (1.98×), VALU 67→27 (2.48×), VGPR 18→8,
SGPR 16→10. The `1u << active_k` becomes a literal bound, so the loop's trip
count is known; `dagger` being a literal folds the `imag` sign at compile time
(the `?:` and the negate both disappear); and the store address `v[i + half]`
gets a constant displacement instead of a computed one. The `s_load` count drops
6→4, and the specific load that disappears is visible in the artifacts: the
interpreter form contains exactly one `s_load_dword ..., 0x278` — the
`st->active_k` read, `active_k`'s offset in `V2State` — and the specialized form
contains **none**. The full accounting is worth reading, because the count does
not fall by simple deletion:

| | interpreter | specialized |
|---|---|---|
| `0x20` — `pc` | ✓ | — |
| `0x18` — `instrs` | ✓ | — |
| `0x0` — kernarg block | ✓ | ✓ |
| `0x278` — `st->active_k` | ✓ | — |
| dynamically-indexed frame word (`s10`, `s0`) | ✓ ✓ | — |
| `st->px` / phase operands at fixed offsets | — | ✓ ✓ ✓ |

Three interpreter-only loads (`pc`, `instrs`, `active_k`) and both
*dynamically-indexed* frame reads disappear; three *statically*-addressed loads
appear in their place. The net is 6→4, but the character of what remains is the
point: every surviving load in the specialized form has a compile-time address.

**What it does not buy, and this is the point:** the loop is still there. At
rank 8 with `V2_STRIDE = 256` it is a handful of iterations; at rank 22 it is
thousands. V1 tried to emit those iterations as code (§5.3) and produced 20 MB
of IR that took 221 s to compile. V2 emits *one loop with a known bound*, which
is what LLVM's loop optimizer actually wants.

<figure>
<img src="diagrams/static-rank-tracking.svg" alt="Static rank tracking through a program" width="100%">
<figcaption><b>Figure 7.2</b> — The specializer walks the bytecode maintaining
<code>k</code>, so every rank-dependent bound is a literal at its use site. The
interpreter must reload <code>st-&gt;active_k</code> at every instruction because
any preceding op could have changed it.</figcaption>
</figure>

---

### 7.6 S4 — rank-folded cooperative reduction

**The op** (`v2_ops_body.inc:135-165`) — the most expensive single opcode in the
ISA, and the one with the ABI constraint from §3:

```c
static inline V2_NOISE_ATTR void v2_op_meas_active_interfere(V2State* st, CV2Complex* v, u32 active_k,
                                               u32 axis, u32 slot, u8 flags) {
    u32 t = v2_tid();
    u32 half = 1u << (active_k - 1u);
    int pz = fget(st->pz, axis);
    double lp = 0.0, lm = 0.0;
    for (u32 i = t; i < half; i += V2_STRIDE) {
        CV2Complex vi = v[i], vh = v[i + half];
        lp += cnorm(cadd(vi, vh)); lm += cnorm(csub(vi, vh));
    }
    double p_plus, p_minus; V2_REDUCE2(t, lp, lm, &p_plus, &p_minus);
    if (IS_OWNER) {
        u8 b = sample_branch(st->rng, p_plus, p_minus, p_plus + p_minus);
        st->branch = b;
        u8 m_abs = b ^ (u8)pz;
        if (slot < V2_MAX_MEAS) mset(st->meas, slot, m_abs ^ (u8)((flags & FLAG_SIGN) != 0));
    }
    v2_barrier();
    for (u32 i = t; i < half; i += V2_STRIDE) {          // fold
        CV2Complex vi = v[i], vh = v[i + half];
        CV2Complex folded = (st->branch == 0) ? cadd(vi, vh) : csub(vi, vh);
        v[i] = cscale(folded, V2_INV_SQRT2);
    }
    v2_barrier();
    if (IS_OWNER) {
        st->active_k = active_k - 1;
        u8 m_abs = mget(st->meas, slot) ^ (u8)((flags & FLAG_SIGN) != 0);
        fset(st->px, axis, m_abs != 0); fset(st->pz, axis, 0);
    }
    v2_barrier();
}
```

Two full sweeps over `2^(k-1)` amplitudes with an f64 two-way reduction between
them, and a PRNG draw at the branch point.

**The two forms** (`cases/S4_meas_active.c`) are a single call:

```c
#if SPEC_FORM
    v2_op_meas_active_interfere(st, v, 8u, 5u, 12u, 0u);
#else
    v2_op_meas_active_interfere(st, v, st->active_k, i0.axis_1, i0.a, i0.flags);
#endif
```

**Result:** 331→240 instructions (1.38×), VALU 139→96 (1.45×), branches 15→7,
VGPR 24→14, SGPR 30→26. Constant `k` fixes both trip counts and the LDS offsets
the reduction indexes with; constant `slot`/`flags` folds the bounds check and
the sign XOR as in S2.

**The `ds_op` column stays at 4 in both forms**, and that is the load-bearing
observation for §12. Those four are two `ds_write2_b64` / `ds_read2_b64` pairs —
the cross-wave combine step, where each wave's partial sums go through LDS —
and specialization does **not** remove or reorder them.

The column understates the evidence, though, because `build_examples.sh` counts
`ds_(read|write)` only. Counting the shuffle instructions as well:

| | interpreter | specialized |
|---|---|---|
| `ds_bpermute_b32` | 32 | 32 |
| `ds_write2_b64` / `ds_read2_b64` | 2 / 2 | 2 / 2 |

**32 `ds_bpermute_b32` in both forms, exactly.** That is the butterfly: six XOR
steps at offsets 32/16/8/4/2/1, two f64 values per step, two 32-bit halves per
f64 — and the second, four-lane reduction on top. Every one survives
specialization unchanged, in the same order. That is deliberate:
`v2_ops.h:235-236` records the constraint —

> MUST reproduce SVM `coop_reduce2`'s exact summation order or f64 rounding
> diverges at measurement branch points.

The reduction is part of the ABI. If specialization had reassociated it, V2
would compute different `p_plus`/`p_minus` in the last bits, `sample_branch`
would take a different branch on some shot, the PRNG streams would desync, and
the outputs would stop being byte-exact. The `4→4` in that column is the
evidence that it did not. This is also why the compile line carries
`-ffp-contract=off` (§6.4): FMA contraction would change the summation the same
way.

VALU 1.45× is the *lowest* gain among the array ops, and that is expected: most
of the 139 vector instructions are the actual complex arithmetic — `cadd`,
`csub`, `cnorm`, `cmul` — which is real work, not interpretive overhead. There
is nothing to fold in an `f32` multiply. Specialization removes the *addressing*
around the arithmetic, not the arithmetic.

---

### 7.7 S5 — scatter-index folding

**The op** (`v2_ops_body.inc:168-180`):

```c
static inline __attribute__((always_inline)) void v2_op_array_cnot(V2State* st, CV2Complex* v, u32 active_k, u32 c, u32 tg) {
    u32 t = v2_tid();
    u64 c_bit = 1ull << c, t_bit = 1ull << tg;
    u64 iters = 1ull << (active_k - 2u);
    for (u64 i = t; i < iters; i += V2_STRIDE) {
        u64 base = scatter_bits_2(i, c, tg) | c_bit;
        CV2Complex a = v[base], b = v[base | t_bit];
        v[base] = b; v[base | t_bit] = a;
    }
    v2_barrier();
    if (IS_OWNER) { int px_c = fget(st->px, c), pz_t = fget(st->pz, tg); fxor(st->px, tg, px_c); fxor(st->pz, c, pz_t); }
    v2_barrier();
}
```

`scatter_bits_2(i, a, b)` inserts two zero bits at positions `a` and `b` — the
standard "enumerate all indices with these two bits cleared" trick. With runtime
axes it is a `min`/`max` plus two variable-shift mask constructions plus three
shift/or steps, all on 64-bit values, *per iteration*.

**The two forms** (`cases/S5_array_cnot.c`):

```c
#if SPEC_FORM
    v2_op_array_cnot(st, v, 8u, 2u, 5u);
    v2_op_array_cz(st, v, 8u, 1u, 6u);
#else
    v2_op_array_cnot(st, v, st->active_k, i0.axis_1, i0.axis_2);
    v2_op_array_cz(st, v, st->active_k, i1.axis_1, i1.axis_2);
#endif
```

**The assembly diff is dramatic.** Specialized, the entire index computation for
`k=8, c=2, t=5` collapses to five instructions with literal masks:

```asm
	v_cmp_gt_u32_e64 s[0:1], 64, v0     ; i < iters, iters = 1<<(8-2) = 64, a LITERAL
	v_lshlrev_b32_e32 v3, 1, v0
	v_and_b32_e32 v2, 3, v0             ; scatter_bits_2 collapsed to
	v_and_b32_e32 v3, 24, v3            ;   three constant masks
	v_and_b32_e32 v4, 0xc0, v1          ;   and one v_or3
	v_or3_b32     v2, v3, v2, v4
	v_lshlrev_b32_e32 v6, 3, v2
	global_load_dwordx2 v[2:3], v6, s[6:7] offset:288   ; base|t_bit -> a fixed DISPLACEMENT
	global_load_dwordx2 v[4:5], v6, s[6:7] offset:32
```

`t_bit = 1 << 5` at 8 bytes per amplitude is `offset:288` versus `offset:32` —
the paired access becomes a **constant address displacement**, folded into the
memory instruction's encoding. No second address register, no add.

The interpreter form has to build all of it at runtime. First the trip count,
which cannot be known until `active_k` has been loaded
(`interp/S5_array_cnot.s:31-35`):

```asm
	s_load_dword s23, s[4:5], 0x278     ; st->active_k -- a memory round trip
	s_add_i32 s23, s23, -2              ; k - 2
	v_lshrrev_b64 v[4:5], s23, v[0:1]   ; iters = 1<<(k-2), by VARIABLE shift
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]
```

then the index math itself, once per loop entry (`:41-48`):

```asm
	s_min_u32 s10, s22, s21             ; scatter_bits_2: min/max of the two axes
	s_max_u32 s14, s22, s21
	v_and_b32_e32 v2, 0xffff, v2
	s_and_b32 s8, 0xffff, s21
	s_lshl_b64 s[10:11], -1, s10        ; ...then variable-shift masks
	s_lshl_b64 s[14:15], -1, s14
	v_lshlrev_b64 v[2:3], v2, 1         ; 1 << c, on the VECTOR pipe
	s_lshl_b64 s[8:9], 1, s8
```

followed at `:49-50` by two `s_not_b64` to complete the masks. Those two
`s_not_b64` appear **twice** in the interpreter file — at `:49-50` and again at
`:151-152`, once per call site, since `array_cnot` and `array_cz` each rebuild
the masks — and **zero** times in the specialized file.

Every one of those operands is a literal in the specialized form, so the entire
sequence — both blocks, both call sites — collapses into the five constant-mask
instructions quoted above.

**Result:** 207→81 instructions (2.56×), VALU 84→33 (2.55×), VGPR 16→8, SGPR
24→10, and **`s_load` 4→1** — the largest relative scalar-load reduction in the
study. The four interpreter loads are, in file order:

```
:9   s_load_dword   s6,      s[0:1], 0x20    ; pc
:10  s_load_dwordx2 s[2:3],  s[0:1], 0x18    ; the instrs pointer
:27  s_load_dwordx4 s[4:7],  s[0:1], 0x0     ; the kernel argument block
:31  s_load_dword   s23,     s[4:5], 0x278   ; st->active_k
```

and the specialized form contains exactly one:

```
:9   s_load_dwordx4 s[4:7],  s[0:1], 0x0     ; the kernel argument block
```

Only the third survives — it is the kernel argument pointer, which nothing can
remove. The other three are, precisely, the three things specialization knows:
where in the program we are, what the instruction says, and what the rank is.

The 16→8 VGPR halving is significant beyond the instruction count: on the coop
tier, VGPR count sets waves-per-SIMD occupancy, and array two-qubit ops
dominate the register pressure of any circuit with a lot of entangling gates.
§9 traces how this fed into the LDS/occupancy work.

---

### 7.8 S6 — fused-matrix table lookup (a bounded gain)

This case exists to establish an *upper bound on what specialization can do for
a data-dependent op*, and the case file says so:

> S6 — fused-matrix table lookup. `v2_op_array_u2` indexes `fused_u2[cp]` then
> picks `matrices[in_state]` where `in_state` comes from the **LIVE Pauli
> frame**, so the row is **NOT foldable** — only the table entry `cp` and the
> axis are.

**The op** (`v2_ops_body.inc:284-305`):

```c
static inline __attribute__((always_inline)) void v2_op_array_u2(V2State* st, CV2Complex* v, u32 active_k, u32 axis,
                                  const CV2FusedU2Entry* fused_u2, u32 cp) {
    u32 t = v2_tid();
    int in_state = (fget(st->pz, axis) ? 2 : 0) | (fget(st->px, axis) ? 1 : 0);
    const CV2Complex* mat = fused_u2[cp].matrices[in_state];    // <- in_state is RUNTIME
    if (axis < active_k) {
        u64 axis_bit = 1ull << axis;
        u64 iters = 1ull << (active_k - 1u);
        for (u64 i = t; i < iters; i += V2_STRIDE) {
            u64 i0 = scatter_bits_1(i, axis), i1 = i0 | axis_bit;
            CV2Complex a = v[i0], b = v[i1];
            v[i0] = cadd(cmul(a, mat[0]), cmul(b, mat[1]));
            v[i1] = cadd(cmul(a, mat[2]), cmul(b, mat[3]));
        }
    }
    v2_barrier();
    if (IS_OWNER) {
        u8 out = fused_u2[cp].out_states[in_state];
        fset(st->px, axis, (out & 1) != 0); fset(st->pz, axis, (out & 2) != 0);
    }
    v2_barrier();
}
```

`in_state` is read from the Pauli frame *at execution time*. The frame depends
on measurement outcomes, which depend on the PRNG, which depends on the shot.
No amount of static analysis recovers it. The matrix row therefore stays a
runtime load in both forms.

**Result:** 147→86 instructions (1.71×), VALU 54→42 (1.29×), `s_load` 12→7,
SGPR 23→16, but **branches 10→2**. What *did* fold: `cp` (the table entry — a
fixed displacement into `fused_u2`), `axis` (so `scatter_bits_1` and `axis_bit`
become constant masks, as in S5), and crucially `axis < active_k` — the case
calls `v2_op_array_u2(st, v, 8u, 4u, fused_u2, 17u)`, i.e. `axis=4`,
`active_k=8`, so with both constant the guard is decided at compile time and the
`if` is entered unconditionally. That single fold is most of the 10→2 branch
drop; the two survivors in the specialized form are at `:22` and `:73` and are
both `s_cbranch_execz`, the loop's own exec-mask guards, which no amount of
constant folding removes.

The VALU ratio of 1.29× is the honest ceiling for this class. Four complex
multiplies and two adds per amplitude pair are irreducible arithmetic. This is
the same shape as S4: specialization removes the scaffolding, not the math.

---

### 7.9 S7 — the noise runtime loop (the negative result)

**The op** (`v2_ops_body.inc:438-451`):

```c
static inline V2_NOISE_ATTR void v2_op_noise_block(V2State* st, const CV2NoiseSite* sites, const CV2Channel* channels,
                                     const double* hazards, u32 num_sites, u32 start, u32 count) {
    u32 t = v2_tid();
    if (IS_OWNER) {
        u32 end = start + count;
        while (st->next_noise >= start && st->next_noise < end) {
            u32 site_idx = st->next_noise;
            apply_noise_site(st, sites, channels, site_idx);
            st->next_noise = site_idx + 1u;
            draw_next_noise(st, hazards, num_sites);
        }
    }
    v2_barrier();
}
```

`OP_NOISE_BLOCK` covers the half-open site range `[start, start+count)`, but it
**consumes only those sites the PRNG selects**. `draw_next_noise` performs an
`ocml_log_f64` hazard draw and advances `st->next_noise` by a random amount. The
loop's trip count is a function of the random stream. It stays a loop in **both**
forms; specialization folds only `start` and `count`.

**The two forms** (`cases/S7_noise_block.c`):

```c
#if SPEC_FORM
    v2_op_noise_block(st, noise_sites, noise_channels, noise_hazards, num_noise_sites, 131u, 57u);
    v2_op_noise(st, noise_sites, noise_channels, noise_hazards, num_noise_sites, 188u);
#else
    v2_op_noise_block(st, ..., i0.a, i0.b);
    v2_op_noise(st, ..., i1.a);
#endif
```

**Result: 447→407 instructions (1.10×), VALU 214→210 (1.02×), branches 23→20,
VGPR 56→56, SGPR 86→85.** This is essentially nothing — and the register
columns, which an earlier draft misreported as `0→0` because LLVM prints them
symbolically for kernels that call external functions (see the note in §7.2),
say something stronger than "no gain": **specialization does not reduce this
op's register pressure at all.** 56 VGPRs in both forms. That matters for §11.1,
because register pressure is precisely the cost that straight-lining hundreds of
these into one function *adds*.

The assembly diff confirms it — what changed is only the range test
(`interp:11-38` vs `spec:15-25`):

```diff
-	s_load_dword s0, s[4:5], 0x20              ; pc
-	s_mul_i32 s6, s0, 40                       ; instruction address
-	s_load_dwordx2 s[70:71], s[4:5], 0x8       ; ins.a = start, ins.b = count
-	s_add_i32 s69, s71, s70                    ; end = start + count, at RUNTIME
-	s_cmp_lt_u32 s0, s70                       ; next_noise < start
-	s_cselect_b64 s[2:3], -1, 0
-	s_cmp_ge_u32 s0, s69                       ; next_noise >= end
-	s_cselect_b64 s[4:5], -1, 0
-	s_or_b64 s[2:3], s[2:3], s[4:5]
 	s_load_dword s0, s[64:65], 0x27c           ; st->next_noise -- in BOTH forms
+	s_add_i32 s1, s0, 0xffffff7d               ; next_noise - 131
+	s_cmp_gt_u32 s1, 56                        ; ...single unsigned compare vs 56
```

The two-compare range test becomes the classic single-compare `(x - lo) > (hi -
lo)` idiom. That is the whole win: four scalar instructions. The 210 surviving
VALU instructions are `apply_noise_site` and the `ocml_log_f64` hazard draw, and
none of them care what `start` was.

**Why this negative result is important.** It is the first-principles prediction
of the `circuit_d5` regression in §11.1. A circuit whose instruction mix is
dominated by noise ops has almost nothing for the specializer to fold — so
specialization buys ~1.05× on the op bodies, while *paying* the cost of
straight-lining hundreds of them into one function (register pressure, I-cache
pressure, and the FP-scheduling hazard that `V2_NOISE_ATTR` exists to fence).

`coop_circuit_d5.c`, the emitted specialization for `circuit_d5`, contains
**1,720** `v2_op_*` calls. Counting them by opcode gives the mix directly:

| Opcode family | calls | share | S7 verdict |
|---|---:|---:|---|
| `frame_cnot` / `frame_cz` / `frame_h` / `frame_swap` | 757 | 44.0 % | folds well (S1) |
| `meas_dormant_static` / `_random` | 215 | 12.5 % | folds well (S2) |
| **`noise` / `noise_block` / `readout_noise`** | **329** | **19.1 %** | **does not fold (S7)** |
| `apply_pauli` | 135 | 7.8 % | bounded (S8) |
| `detector` / `observable` | 109 | 6.3 % | — |
| array ops (`array_t`, `multi_cnot`, `cnot`, `s`, `multi_cz`) | 136 | 7.9 % | bounded (S5, S6) |
| `expand_t` / `swap_meas_interfere` | 39 | 2.3 % | folds (S3, S4) |

Nearly a fifth of the program is in the one class S7 measured at 1.02× — and
that class is by far the most *expensive* per call (447 instructions in the
interpreter form, against 136 for a frame op), so its share of runtime is much
larger than its share of call sites. The frame and measurement ops that do fold
well are individually cheap. That is the regression in one table: the ops worth
specializing are the ones that cost little, and the op that dominates the time
is the one specialization cannot touch. §11.1 shows what happened when it was
specialized anyway.

---

### 7.10 S8 — Pauli-mask index folding (a bounded gain, for a different reason)

**The op** (`v2_ops_body.inc:406-413`):

```c
static inline __attribute__((always_inline)) void v2_op_apply_pauli(V2State* st, const CV2Mask* pauli_masks, u32 cp, u32 cond_slot) {
    u32 t = v2_tid();
    if (IS_OWNER && mget(st->meas, cond_slot) != 0) {
        const CV2Mask* m = &pauli_masks[cp];
        for (u32 w = 0; w < CLIFFT_V2_PAULI_WORDS; ++w) { st->px[w] ^= m->x[w]; st->pz[w] ^= m->z[w]; }
    }
    v2_barrier();
}
```

The case comment states the boundary precisely:

> the mask **INDEX** and the measurement slot fold, the mask **CONTENTS** do not
> (they live in a device buffer).

**Result:** 185→137 instructions (1.35×), VALU 71→63 (1.13×), VGPR 22→21, SGPR
44→42, `s_load` 17→11, **branches 3→3**.

This is a different shape of bounded gain from S6. In S6 the *arithmetic* was
irreducible; here the **memory traffic** is. `pauli_masks[cp]` with a constant
`cp` becomes a fixed displacement — visible directly in the load offsets, which
go from a *computed* index in the interpreter form

```
:32  s_load_dwordx2 s[4:5], s[34:35], s7 offset:0x50    ; index in a REGISTER
```

to three literal displacements in the specialized form, one per call site:

```
:19  s_load_dwordx16 s[0:15],  s[36:37], 0x0
:74  s_load_dwordx16 s[0:15],  s[36:37], 0x58   ; pauli_masks[1]
:118 s_load_dwordx16 s[0:15],  s[36:37], 0xb0   ; pauli_masks[2]
```

`0x58` is 88 bytes, which is `sizeof(CV2Mask)` for
`CLIFFT_V2_PAULI_WORDS = 5` — two arrays of five `uint64_t` plus a sign byte,
padded to 8. The stride between the three constants is exactly one mask. That is
the 17→11 `s_load` drop. But the five words still have to be *loaded* and XORed
into the frame, and the number of loads is set by the mask width, not by the
operand.

**The branch count does not move — but two of the three branches change kind,
and that is the more interesting fact.** The interpreter form has one
`s_cbranch_scc1` and two `s_cbranch_vccnz`; the specialized form has three
`s_cbranch_scc1`. A `vccnz` branch tests the *vector* condition code — the
lane-mask result of a `v_cmp` — where `scc1` tests the scalar condition code.
Specialization did not delete the branch, but it moved the *predicate* that
drives it off the vector pipe, exactly as §7.3 described for the arithmetic.
What it could not do is remove the branch entirely, because the condition
`mget(st->meas, cond_slot) != 0` tests a **runtime measurement outcome**. It is
the same wall as S6's `in_state`: data that only exists once the shot is running.
Constant `cond_slot` fixes *which bit* is tested; nothing fixes its value.

Together S6, S7, and S8 delimit the technique. Specialization folds:

- **operand-derived address and mask arithmetic** (S1, S5) — large win;
- **control flow selected by constant flags and bounds** (S2, S3, S6's guard) —
  large win, and the branch removal matters more on a GPU than the count suggests;
- **loop trip counts derived from a statically-tracked rank** (S3, S4, S5) — the
  structural win, and the one V1 tried to get by unrolling instead.

It does **not** fold:

- floating-point arithmetic on amplitudes (S4, S6);
- anything indexed by the live Pauli frame or a measurement outcome (S6, S8);
- anything whose trip count depends on the PRNG (S7).

That taxonomy predicts the per-circuit results in §14 well enough that it is
worth stating as the chapter's conclusion: **a circuit's speedup under V2 is a
function of how much of its instruction mix falls in the first list.**

---
