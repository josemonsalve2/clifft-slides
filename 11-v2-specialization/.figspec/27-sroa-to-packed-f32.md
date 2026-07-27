# Figure 8.3 — `diagrams/sroa-to-packed-f32.svg`

**Caption it must serve:** "The one real arithmetic difference, and the chain
that produces it. Because the specializer writes a straight-line body with
statically-known indices, the amplitudes end up in registers rather than in a
stack array — so SROA can promote them, so the SLP vectorizer can pack the
complex arithmetic into `<2 x float>`, so the machine gets `v_pk_*_f32`. V1 has
zero packed f32 instructions; V2 has 847. Note the counter-intuitive middle
step: `fmul` *rises* 12 → 448 and `shufflevector` appears from nothing, and the
program gets faster."

**Title:** Registers, Then Packing
**Subtitle:** Why `fmul` rising from 12 to 448 is the good news

## What it must show

A left-to-right transformation chain in **four stages**, with the IR census
underneath the first two and the ISA consequence under the last. This is an
**extension of `lowering-pipelines.svg`**, which gives per-stage *sizes*; this
figure gives the *mechanism* inside the `-O0 → -O2` collapse. Do not redraw a
stage-size chart.

### Stage 1 — `clangO0.ll` (red / `#cc2222`)

Label: **`-O0` · every local on the stack**. Show a stack-slot pile: an
`addrspace(5)` box with the amplitude array in it, and an arrow marked "escaped
address" from it. Census, as a two-column table (op, count):

    load           4,898
    store          2,488
    alloca         2,271
    addrspacecast  2,262
    call           1,770

Caption under the census: **19,551 IR lines**.

### Stage 2 — the passes (cyan `#53d8fb`, drawn as a labelled band, not a box)

Three named passes in sequence, each a pill: **SROA / mem2reg**, **inlining**,
**SLP vectorizer**. Between the pills, short arrows. Under the band, one line:
*the amplitudes have statically-known indices, so their addresses do not
escape.*

### Stage 3 — `clangO2.ll` (green `#00cc66`)

Label: **`-O2` · the state is in registers**. Same census, `-O0 → -O2`, with the
direction of each change coloured — falls in green, **rises in cyan** (they are
not regressions):

    alloca          2,271 →     3     (green)
    addrspacecast   2,262 →     8     (green)
    call            1,770 →    72     (green)
    load            4,898 → 1,014     (green)
    store           2,488 → 1,103     (green)
    fmul               12 →   448     (cyan)
    shufflevector       0 →   459     (cyan)

Caption under the census: **8,132 IR lines**.

Two callouts attached to this stage:

- On `alloca 2,271 → 3`, an orange `#ff8800` note: **three survive** — they are
  aggregates with escaping addresses, and they are the `private=336` scratch
  residue. **The claim is not that `alloca` reached zero. It did not.** What
  matters is that 2,271 grew with circuit length and 3 does not.
- On `call 1,770 → 72`, a small note: the 72 survivors are the `noinline` noise
  ops from `V2_NOISE_ATTR`, plus ocml calls.

### Stage 4 — the ISA (green, the payoff)

Two instruction pills with counts:

    v_pk_add_f32   431
    v_pk_mul_f32   416

and beside them the comparison bar: **V1 `v_pk_*_f32` = 0**, **V2 = 847**.

Under it, the lane arithmetic stated plainly: *V2 covers 1,705 f32 lanes in 858
instructions; V1 needs 1,586 instructions for 1,586 lanes.* Two lanes per
instruction, drawn as a small two-cell glyph beside `v_pk_add_f32`.

### Bottom strip

One sentence, full width, accent-bordered:

**"The real instruction-level win is packing, and it is 2× on the f32 path — not
23× on the f64 one."**

## DATA — verbatim, invent nothing

- `clangO0.ll` → `clangO2.ll`: **19,551 → 8,132** lines.
- `-O0` census: `load` **4,898**, `store` **2,488**, `alloca` **2,271**,
  `addrspacecast` **2,262**, `call` **1,770**.
- `-O0 → -O2`: `alloca` **2,271 → 3**; `addrspacecast` **2,262 → 8**; `call`
  **1,770 → 72**; `load` **4,898 → 1,014**; `store` **2,488 → 1,103**; `fmul`
  **12 → 448**; `shufflevector` **0 → 459**.
- Final ISA: `v_pk_add_f32` **431**, `v_pk_mul_f32` **416**.
- `v_pk_*_f32`: V1 **0**, V2 **847**.
- Lanes: V2 **1,705** f32 lanes in **858** instructions; V1 **1,586**
  instructions for **1,586** lanes.
- Scratch residue: `private=336`.

## Notes

- The three surviving `alloca`s must be drawn. The report explicitly corrects an
  earlier draft that read "2,268 → 0", and the figure must not reintroduce the
  error it corrects.
- Colour discipline: the rising counts (`fmul`, `shufflevector`) are **cyan
  explanatory**, never red. They are the mechanism, not a regression.
- Do not attribute the win to precision. §8.6's conclusion is that both backends
  compute the same quantities at the same widths.
