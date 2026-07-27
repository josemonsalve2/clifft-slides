## 2. The SVM: what is actually being simulated

Before any GPU discussion, it is worth being precise about what the machine
executes, because every subsequent design decision — the tiers, the
specialization classes, the f32/f64 gap, even the noise regression — follows
from the structure of this one object.

### 2.1 The factored state

`clifft` does not carry a dense statevector over all *N* qubits. It carries the
factored representation documented at `src/clifft/svm/svm.h:92-103`:

```
|psi> = gamma · U_C · P · ( |phi>_A ⊗ |0>_D )
```

| symbol | what it is | storage |
|---|---|---|
| `gamma` | global scalar: phase + deferred normalization | one complex scalar |
| `U_C` | the Clifford part, tracked symbolically | implicit in the frame |
| `P` | a Pauli frame — one X bit and one Z bit per qubit | 2 × `ceil(N/64)` words |
| `|phi>_A` | the **active** subspace: a dense statevector over `k` qubits | 2^k complex amplitudes |
| `|0>_D` | the **dormant** qubits, provably still in the computational basis | nothing |

The whole performance story lives in `k`, the **active rank**. A circuit that is
pure Clifford never grows `k` at all: every gate is a frame update, `2^k = 1`,
and simulation is polynomial. Each non-Clifford operation (`T`, `ROT`, an
`EXPAND`) may promote a dormant qubit into the active set, `k → k+1`, doubling
the dense array. Each measurement of an active qubit collapses it back, `k → k-1`.

The **peak rank** of a compiled circuit — the maximum `k` over the whole program
— therefore determines both the memory footprint (2^peak_rank complex
amplitudes) and, as §3 shows, which GPU execution strategy is even viable.

<figure>
<img src="diagrams/factored-state.svg" alt="Factored state representation" width="100%">
<figcaption><b>Figure 2.1</b> — The factored state. Dormant qubits cost zero
storage; the Pauli frame costs 2 bits per qubit; only the active subspace costs
exponential memory. Rank growth is driven by non-Clifford gates and undone by
measurement.</figcaption>
</figure>

### 2.2 The bytecode

The **Schrödinger Virtual Machine** is a bytecode interpreter over that state.
The instruction set is *localized*: the ahead-of-time compiler has already
resolved all global topology into 1- and 2-qubit virtual-axis operations, so the
VM never evaluates basis spans or commutation relations
(`src/clifft/backend/backend.h:21-23`).

The opcode set (`backend.h:25-86` — **41 opcodes** plus the `NUM_OPCODES`
sentinel) partitions cleanly by *what it costs*. The count matters later: it is
41 arms that a CPU interpreter must dispatch over and that a specialized kernel
never dispatches over at all.

| family | count | opcodes | cost |
|---|---|---|---|
| **Frame** | 6 | `FRAME_{CNOT,CZ,H,S,S_DAG,SWAP}` | O(1) — two bit reads, two bit XORs. No amplitude touched. |
| **Array** | 13 | `ARRAY_{CNOT,CZ,SWAP,MULTI_CNOT,MULTI_CZ,H,S,S_DAG,T,T_DAG,ROT,U2,U4}` | O(2^k) — a sweep over amplitude groups. |
| **Expansion** | 4 | `EXPAND`, `EXPAND_T`, `EXPAND_T_DAG`, `EXPAND_ROT` | O(2^k) and **grows k** |
| **Measurement** | 5 | `MEAS_DORMANT_{STATIC,RANDOM}`, `MEAS_ACTIVE_{DIAGONAL,INTERFERE}`, `SWAP_MEAS_INTERFERE` | dormant: O(1). active: O(2^k) reduction, **shrinks k** |
| **Forced variants** | 5 | the five `*_FORCED` mirrors of the measurement family | synthesized at runtime by a bytecode rewrite; read the outcome from a side buffer instead of the PRNG |
| **Classical / noise** | 8 | `APPLY_PAULI`, `NOISE`, `NOISE_BLOCK`, `READOUT_NOISE`, `DETECTOR`, `POSTSELECT`, `OBSERVABLE`, `EXP_VAL` | O(1)–O(words), but **PRNG-consuming** |

6 + 13 + 4 + 5 + 5 + 8 = 41. ✓

The `FRAME_*` family is why `clifft` is fast at all: a Clifford gate is two bit
flips. The `ARRAY_*` and `EXPAND` families are where the GPU work is. The noise
family is where, as §11 shows, specialization stops paying.

Each instruction is exactly **32 bytes** — asserted, not assumed
(`backend.h:163`) — so that two land in one 64-byte cache line
(`backend.h:92`):

```c
struct alignas(32) Instruction {
    Opcode   opcode;      // offset 0
    uint8_t  _reserved;   // offset 1
    uint8_t  flags;       // offset 2   FLAG_SIGN | FLAG_HIDDEN | FLAG_IDENTITY | FLAG_EXPECTED_ONE
    uint8_t  _pad;        // offset 3
    uint16_t axis_1;      // offset 4   virtual axis (target/control)
    uint16_t axis_2;      // offset 6   virtual axis 2
    union {               // offsets 8..31 — seven payload variants + raw access
        struct { double   weight_re, weight_im;         } math;       // A
        struct { uint32_t classical_idx, expected_val;  } classical;  // B
        struct { uint32_t cp_mask_idx, condition_idx;   } pauli;      // C
        struct { uint64_t mask;                         } multi_gate; // D
        struct { uint32_t cp_idx;                       } u2;         // E  -> fused_u2_nodes
        struct { uint32_t cp_idx;                       } u4;         // F  -> fused_u4_nodes
        struct { uint32_t cp_exp_val_idx, exp_val_idx;  } exp_val;    // G  -> exp_val_masks
        uint8_t raw[24];                                              //     full payload access
    };
};

static_assert(sizeof(Instruction) == 32, "Instruction must be exactly 32 bytes");
```

(Padding members elided above; each variant is padded out to the full 24 bytes
in the header.)

Note variants **E**, **F**, and **G**: their entire payload is a `uint32_t`
*index*. Anything that does not fit in 24 bytes — fused 2×2 and 4×4 unitary
matrices, full N-bit Pauli masks, noise-channel tables — lives in a side
`ConstantPool` and is referenced by index. **That indirection is exactly what
§7's S6 and S8 experiments measure the specialization limit of**: the *index*
folds to an immediate, the *contents* stay in a device buffer. A specializer
can eliminate the load of the index; it cannot eliminate the load of the matrix.

<figure>
<img src="diagrams/bytecode-layout.svg" alt="32-byte instruction encoding" width="100%">
<figcaption><b>Figure 2.2</b> — The 32-byte instruction. Every field in the
fixed header, and the payload's discriminant, is known to the ahead-of-time
compiler. This is the entire raw material available to a specializer.</figcaption>
</figure>

### 2.3 The interpreter loop, and why its shape matters

The CPU reference uses a **computed-goto threaded dispatch table**
(`svm_kernels.inl:2191-2196`), sized to 256 and initialized with designated
initializers so each of the 41 opcodes gets its own indirect-branch history
entry:

```cpp
#if defined(__GNUC__) || defined(__clang__)
    // Threaded dispatch table (computed gotos) gives each opcode its own
    // indirect-branch history entry, dramatically improving prediction.
    // Sized to 256; designated initializers map enums directly to labels.
    static const void* dispatch_table[256] = {
        [static_cast<uint8_t>(Opcode::OP_FRAME_CNOT)] = &&L_OP_FRAME_CNOT,
        [static_cast<uint8_t>(Opcode::OP_FRAME_CZ)]   = &&L_OP_FRAME_CZ,
        ...
    };
```

That comment is the thesis of this entire report, stated by the CPU backend
years earlier. **Dispatch is a first-class cost**, and it is worth writing
non-portable code to attack. On a CPU you fight it with branch-target
prediction — the `#if defined(__GNUC__)` guard exists because the technique is a
compiler extension, and someone decided the win justified the portability
fallback path.

On a GPU there is no branch predictor to help you. A wavefront that hits an
indirect branch does not "predict"; it serializes across whatever divergent
targets its 64 lanes want. The CPU's mitigation is unavailable, which is why the
GPU port cannot use this trick, and why removing dispatch *entirely* — rather
than making it cheaper — is the only move available. That move is worth
**2.8–9.1× on the scalar unit** (§14.2).

### 2.4 The PRNG, and why it must be bit-exact

Stochastic simulation means every backend consumes the same random stream in the
same order, or results are not comparable shot-for-shot. `clifft` uses
**xoshiro256++** seeded by **SplitMix64** (`svm.h:21-34`, the Blackman–Vigna
reference implementation, CC0), and converts to a double with an
explicitly-specified expression (`svm.h:148-150`):

```cpp
// CRITICAL: Do NOT use std::uniform_real_distribution -- its output is
// implementation-defined and varies across compilers (GCC vs Clang vs MSVC).
[[nodiscard]] double random_double() { return static_cast<double>(rng_() >> 11) * 0x1.0p-53; }
```

The V2 device code reproduces this byte-for-byte (`v2_ops.h:143-164`), down to
the rotate constants and the same `>> 11` / `0x1.0p-53` mapping:

```c
static inline u64 v2_rotl64(u64 x, int k) { return (x << k) | (x >> (64 - k)); }
static inline u64 rng_next(u64* s) {
    u64 result = v2_rotl64(s[0] + s[3], 23) + s[0];
    u64 t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;    s[3] = v2_rotl64(s[3], 45);
    return result;
}
static inline double rng_uniform(u64* s) {
    return (double)(rng_next(s) >> 11) * 0x1.0p-53;
}
```

Note what is *not* here: no `ocml` call, no device-library dependency, no
fast-math-sensitive arithmetic. The conversion is an integer shift and an exact
power-of-two multiply, so it is bit-identical on any target with IEEE doubles.
This is the one place in the whole pipeline where "compile the same source for
both" was not good enough — the CPU side is C++ and the device side is freestanding
C23, so the sequence had to be transliterated and then verified by output
comparison rather than shared.

The consequence, and it is the key to understanding §12: **any divergence in how
many random numbers a backend draws is unrecoverable.** Not "slightly different
results" — the two streams decorrelate permanently from the first extra draw.
A branch taken differently, an early-exit that skips a draw, a dust threshold
that clamps on one side and rolls on the other: all of these are the same bug.

### 2.5 What this means for a GPU

Three properties of the workload determine everything downstream:

1. **The hot loop is a butterfly over amplitude pairs**, not a matrix multiply.
   `ARRAY_U2` applies a 2×2 complex matrix to 2^(k-1) amplitude pairs;
   `ARRAY_U4` a 4×4 to 2^(k-2) quadruples. There is no GEMM here — which is why
   **`SQ_INSTS_MFMA` is 0.0 in every counter block collected** (51 of 52
   backend×circuit cells in the `20260726T182433Z_report-final-postdust` run;
   the 52nd, `qv24_L4_seed42`'s SVM side, has an empty counter block, so it is
   *unmeasured* rather than nonzero) — and why the matrix cores, the headline
   feature of this chip, are simply not part of this story. §14.6 returns to
   what that costs.
2. **The working set spans four orders of magnitude.** At peak rank 0 the
   "statevector" is one complex number; at rank 24 it is 16.7 million. No single
   parallelization strategy is right for both, which is the origin of the tier
   system (§3.2).
3. **Shots are embarrassingly parallel, but each shot is sequential.** A shot is
   a walk through the bytecode with a private PRNG stream. The parallelism is
   *across* shots, and the per-shot work is a dependent chain — which sets the
   ceiling on what any amount of ILP inside one shot can buy.

---
