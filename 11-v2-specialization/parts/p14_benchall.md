## 15. The full benchmark report

§14 argued a thesis. This chapter is the underlying data, in full, with no
selection: 26 circuits, both backends, every counter collected, every launch
parameter recorded. **Where any earlier chapter of this report disagrees with a
number here, this chapter is correct.** §1–§10 were written against earlier runs
on a different node; several of their figures have been superseded and are
retained only where the text says so explicitly.

### 15.1 Provenance

```json
{ "run_id":  "20260727T125310Z_report-final-allfixtures",
  "label":   "report-final-allfixtures",
  "commit":  "79d4463",
  "branch":  "mlir-v2",
  "dirty":   false,
  "v2_specialize": true,
  "n_circuits":    26 }

{ "slurm_job_id": 50793,
  "node":         "smci350-rck-g03-d13-21",
  "partition":    "mi350x-es",
  "arch":         "gfx950",
  "note": "mi350x-es is heterogeneous. Do NOT compare ratios across nodes." }
```

> **The node caveat, stated once and meant throughout.** This job ran on
> `d13-21`. The absolute timings in §1–§10 come from `f13-21`. V2/SVM *ratios*
> within this job are sound — both arms, same job, same node, interleaved. The
> absolute microsecond figures must not be compared against those chapters.
> §1.3(d) is the record of what happens when this rule is broken: a projected
> ratio of ~0.40 for the d5 family against a measured 0.84, wrong by 2× in the
> flattering direction.

`v2_specialize: true` is not decoration. `V2_SPECIALIZE` is opt-in; with it
unset the harness measures the *interpreter*, and the resulting numbers look
like a 2–3× regression that is really a configuration error. The manifest
records the flag so a reader can tell which kernel was measured.

### 15.2 What each run does

Per circuit, per backend, five `rocprofv3` invocations — plus one untraced warm-up
whose purpose is to move `.hsaco` compilation and gating *off* the traced path:

```bash
prof() {  # $1=outdir  rest=cmd
  local outdir="$1"; shift; local cmd="$*"
  mkdir -p "$outdir"
  eval "$cmd" >/dev/null 2>&1   # warm: compile+gate the .hsaco OFF the traced path
  timeout 300 rocprofv3 --kernel-trace   --stats --output-format csv -d "$outdir/kt"   -- $cmd
  timeout 300 rocprofv3 --hsa-core-trace --stats --output-format csv -d "$outdir/hsa"  -- $cmd
  timeout 300 rocprofv3 --pmc $PMC_A --output-format csv -d "$outdir/pmcA" -- $cmd
  timeout 300 rocprofv3 --pmc $PMC_B --output-format csv -d "$outdir/pmcB" -- $cmd
  timeout 300 rocprofv3 --pmc $PMC_C --output-format csv -d "$outdir/pmcC" -- $cmd
}
```

The counters are split into three groups because the hardware cannot collect
them simultaneously (`bench_all.sh:81-83`):

```bash
PMC_A="SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_MFMA,SQ_INSTS_SALU,SQ_INSTS_LDS"
PMC_B="TCC_HIT_sum,TCC_MISS_sum"
PMC_C="SQ_BUSY_CYCLES,GRBM_GUI_ACTIVE,SQ_WAIT_INST_LDS,SQ_WAVE_CYCLES"
```

**Each group is a separate execution of the kernel.** Ratios *within* a group,
and ratios of one counter *between the two backends*, are sound. Quotients of
counters drawn from *different* groups are not, absent an argument that the runs
did identical work — §14.5 retracts a claim that violated exactly this.

Shot counts vary by circuit, from 20,000 on the small register-tier fixtures
down to 500 on `qv24_L4`, chosen so that each run completes inside the 300 s
`timeout` (`bench_all.sh:90-113`). Shot count therefore differs *between*
circuits but is identical between the two backends of any one circuit, which is
what the ratio requires.

### 15.3 The complete result table

Every circuit. Kernel time is `per_dispatch_ns_median`; each backend issues
exactly one dispatch of the kernel under test.

| # | circuit | tier | shots | V2 (µs) | SVM (µs) | V2/SVM |
|---:|---|---|---:|---:|---:|---:|
| 1 | `circuit_d3_p0.001` | register | 20,000 | 220.7 | 433.6 | 0.509 |
| 2 | `circuit_d5_p0.0005` | coop | 10,000 | 8,643.0 | 10,101.6 | 0.856 |
| 3 | `circuit_d5_p0.001` | coop | 10,000 | 8,599.8 | 10,158.4 | 0.847 |
| 4 | `circuit_d5_p0.002` | coop | 10,000 | 8,729.9 | 10,318.6 | 0.846 |
| 5 | `circuit_d5_p0.003` | coop | 10,000 | 8,808.8 | 10,516.1 | 0.838 |
| 6 | `circuit_d5_p0.005` | coop | 10,000 | 8,727.0 | 10,713.5 | 0.815 |
| 7 | `cultivation_d5` | coop | 20,000 | 16,421.3 | 20,900.8 | 0.786 |
| 8 | `four_t` | register | 20,000 | 13.1 | 22.3 | 0.587 |
| 9 | `frame_h` | register | 20,000 | 12.1 | 16.3 | 0.742 |
| 10 | `qv10` | coop | 20,000 | 1,385.6 | 4,349.7 | 0.319 |
| 11 | `qv20_L8_seed42` | global | 2,000 | 1,583,246.4 | 2,162,954.0 | 0.732 |
| 12 | `qv20_seed42` | global | 2,000 | 1,804,648.5 | 2,123,250.3 | 0.850 |
| 13 | `qv21_L8_seed42` | global | 2,000 | 2,977,385.5 | 4,045,316.0 | 0.736 |
| 14 | `qv22_L6_seed42` | global | 1,000 | 3,179,444.7 | 3,243,841.8 | 0.980 |
| 15 | `qv23_L5_seed42` | global | 1,000 | 6,086,167.4 | 6,146,952.4 | 0.990 |
| 16 | `qv24_L4_seed42` | global | 500 | 7,241,815.1 | 8,211,443.4 | 0.882 |
| 17 | `surface_d11_t10` | coop | 10,000 | 37,986.3 | 81,369.2 | 0.467 |
| 18 | `surface_d11_t15` | global | 5,000 | 19,589.4 | 75,946.8 | 0.258 |
| 19 | `surface_d11_t19` | global | 5,000 | 20,423.1 | 79,842.1 | 0.256 |
| 20 | `surface_d7_t10` | coop | 10,000 | 11,044.2 | 20,673.5 | 0.534 |
| 21 | `surface_d7_t15` | coop | 10,000 | 11,225.3 | 21,997.6 | 0.510 |
| 22 | `surface_d7_t19` | global | 5,000 | 6,246.9 | 20,152.3 | 0.310 |
| 23 | `surface_d7_t5` | register | 20,000 | 2,051.1 | 3,372.2 | 0.608 |
| 24 | `surface_d9_t10` | coop | 10,000 | 21,631.9 | 44,093.9 | 0.491 |
| 25 | `surface_d9_t15` | global | 5,000 | 11,368.7 | 41,325.4 | 0.275 |
| 26 | `surface_d9_t19` | global | 5,000 | 11,925.6 | 45,207.5 | 0.264 |

**Aggregates.** Arithmetic mean 0.626, median 0.670, **wins 26 / 26**.

Geometric mean — the correct average for a set of ratios, and the one that does
not let a single large speedup dominate:

| tier | n | geomean | best | worst |
|---|---:|---:|---:|---:|
| register | 4 | 0.606 | 0.509 | 0.742 |
| coop | 11 | 0.633 | 0.319 | 0.856 |
| global | 11 | 0.508 | 0.256 | 0.990 |
| **all** | **26** | **0.573** | 0.256 | 0.990 |

The global tier has both the best geomean *and* the widest spread — it contains
the surface circuits that fold hardest and the QV circuits that resist most.
Tier alone does not predict speedup; instruction mix (§14.4) and pool occupancy
(§14.5) do.

> **Total kernel time across the corpus is a misleading statistic, and is
> reported here only to be dismissed.** Summing all 26: V2 23.088 s against SVM
> 26.445 s, a ratio of 0.873. That number is dominated by the six QV circuits,
> which alone account for 22.9 s of V2's 23.1 s because they are run at high
> rank for thousands of shots. It describes the composition of this fixture list,
> not the backend. The per-circuit ratios above are the result.

### 15.4 Tier assignment

Tier is a function of `peak_rank` alone (`v2_kernel.cc:312-321`):

```cpp
constexpr uint32_t kRegMaxRank  = 4;
constexpr uint32_t kCoopMaxRank = 10;
const Tier tier = flat.peak_rank <= kRegMaxRank  ? REG
                : flat.peak_rank <= kCoopMaxRank ? COOP : GLOBAL;
```

This is worth stating because the fixture names actively mislead. Within the
surface family, `surface_d11_t10` runs **coop** while `surface_d11_t15` and
`surface_d11_t19` run **global** — same code distance, different tier — and
`surface_d7_t5` runs on the **register** tier despite its name suggesting kinship
with the other `surface_d7_*` circuits. The name encodes the circuit's
construction; the tier follows from the peak rank the StatevectorSqueeze pass
leaves behind. **Read the measured LDS size, not the fixture name.**

### 15.5 Launch geometry and resource footprint

| circuit | V2 LDS | SVM LDS | V2 scr | SVM scr | V2 VGPR | SVM VGPR | SGPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `circuit_d3_p0.001` | 0 | 0 | 1,024 | 4,480 | 64 | 60 | 112 |
| `four_t` | 0 | 0 | 656 | 4,480 | **32** | 60 | 112 |
| `frame_h` | 0 | 0 | 656 | 4,480 | **32** | 60 | 112 |
| `surface_d7_t5` | 0 | 0 | 1,040 | 4,480 | 64 | 60 | 112 |
| `qv10` | 13,312 | 23,040 | 0 | 8 | **36** | 64 | 112 |
| `circuit_d5_*` (5) | 13,312 | 23,040 | 336 | 8 | 64 | 64 | 112 |
| `cultivation_d5` | 13,312 | 23,040 | 336 | 8 | 64 | 64 | 112 |
| `surface_*` coop (4) | 13,312 | 23,040 | 288 | 8 | 64 | 64 | 112 |
| `surface_*` global (5) | 1,024 | 8,704 | 320–352 | 56 | 64 | 64 | 112 |
| `qv20_seed42` | 1,024 | 8,704 | 96 | 56 | **52** | 64 | 112 |
| `qv20_L8` | 1,024 | 8,704 | 112 | 56 | **56** | 64 | 112 |
| `qv21_L8` | 1,024 | 8,704 | 112 | 56 | **52** | 64 | 112 |
| `qv22_L6` | 1,024 | 8,704 | 448 | 56 | 64 | 64 | 112 |
| `qv23_L5` | 1,024 | 8,704 | 576 | 56 | 64 | 64 | 112 |
| `qv24_L4` | 1,024 | 8,704 | 480 | 56 | 64 | 64 | 112 |

Three observations, all structural:

- **LDS.** V2's coop kernels declare `lds_v[1024]` + `lds_red_scratch[512]` at
  8 B each = 12,288 B, plus `lds_state` → 13,312 B. The global kernels declare
  neither, because at rank > 10 the amplitudes live in HBM; their 1,024 B is
  `lds_state` and `lds_shot` (`v2_specializer.cc:185-186`, `:204-205`). SVM
  provisions 23,040 / 8,704 B for the same tiers. The 8.5× global-tier gap is a
  consequence of where the amplitudes live, not of tuning.
- **Scratch, register tier.** 656–1,040 B against SVM's uniform 4,480 B — 4.3×.
  SVM must provision for the worst opcode it *might* interpret; V2 provisions for
  the opcodes the circuit actually contains.
- **VGPRs.** V2 uses fewer than SVM on five circuits (32, 32, 36, 52, 52/56) and
  the same 64 on the rest. It never uses more. The three QV circuits where V2 is
  at 64 are the three weakest results in the corpus, and they are also the three
  that spill — see §14.5.

> **The VGPR column above is `rocprofv3`'s, and it disagrees with the binaries.**
> The AMDHSA metadata in the same kernels' `.hsaco` files reports 104–128 VGPRs
> plus 40–64 AGPRs where the profiler reports 52–64 and `Accum_VGPR_Count = 0`.
> Scratch sizes match one-to-one across both sources, so these are the same
> kernels and the profiler is reporting a different quantity, not a different
> kernel. The column is retained here for consistency with the rest of this
> table, which is profiler-sourced. **For register allocation and spilling,
> §14.5 and §10 use the `.hsaco` metadata, and those are authoritative.**

### 15.6 The complete counter table

Instruction counts, both backends, all 26 circuits. MFMA is omitted from the
table because it is **0.0 in all 52 cells** — 26 circuits × 2 backends, none
missing.

| circuit | VALU V2 | VALU SVM | ratio | SALU V2 | SALU SVM | ratio |
|---|---:|---:|---:|---:|---:|---:|
| `circuit_d3_p0.001` | 3.79e+06 | 4.64e+06 | 0.817 | 1.10e+06 | 7.30e+06 | **0.150** |
| `circuit_d5_p0.0005` | 6.16e+08 | 9.69e+08 | 0.636 | 3.74e+08 | 2.40e+09 | 0.156 |
| `circuit_d5_p0.001` | 6.25e+08 | 9.77e+08 | 0.639 | 3.76e+08 | 2.40e+09 | 0.156 |
| `circuit_d5_p0.002` | 6.40e+08 | 9.92e+08 | 0.646 | 3.79e+08 | 2.41e+09 | 0.158 |
| `circuit_d5_p0.003` | 6.55e+08 | 1.00e+09 | 0.652 | 3.83e+08 | 2.41e+09 | 0.159 |
| `circuit_d5_p0.005` | 6.82e+08 | 1.03e+09 | 0.663 | 3.89e+08 | 2.42e+09 | 0.161 |
| `cultivation_d5` | 1.36e+09 | 2.06e+09 | 0.663 | 7.78e+08 | 4.83e+09 | 0.161 |
| `four_t` | 6.76e+04 | 1.56e+05 | 0.432 | 2.10e+04 | 1.90e+05 | **0.110** |
| `frame_h` | 4.13e+04 | 9.63e+04 | 0.429 | 2.10e+04 | 9.63e+04 | 0.218 |
| `qv10` | 5.02e+08 | 1.13e+09 | 0.445 | 1.20e+08 | 6.05e+08 | 0.198 |
| `qv20_L8_seed42` | 1.28e+11 | 1.54e+11 | 0.830 | 5.39e+09 | 1.57e+10 | 0.344 |
| `qv20_seed42` | 1.74e+11 | 2.11e+11 | 0.821 | 4.79e+09 | 1.39e+10 | 0.345 |
| `qv21_L8_seed42` | 2.40e+11 | 2.88e+11 | 0.835 | 1.01e+10 | 2.93e+10 | 0.345 |
| `qv22_L6_seed42` | 1.96e+11 | 2.30e+11 | 0.855 | 8.15e+09 | 2.33e+10 | 0.349 |
| `qv23_L5_seed42` | 2.73e+11 | 3.18e+11 | 0.861 | 1.16e+10 | 3.24e+10 | 0.357 |
| `qv24_L4_seed42` | 2.31e+11 | 2.68e+11 | 0.863 | 9.86e+09 | 2.73e+10 | **0.361** |
| `surface_d11_t10` | 4.14e+09 | 6.98e+09 | 0.594 | 2.76e+09 | 1.99e+10 | 0.139 |
| `surface_d11_t15` | 2.09e+09 | 4.34e+09 | 0.482 | 1.38e+09 | 9.99e+09 | 0.138 |
| `surface_d11_t19` | 2.19e+09 | 4.46e+09 | 0.490 | 1.40e+09 | 1.01e+10 | 0.138 |
| `surface_d7_t10` | 1.11e+09 | 1.82e+09 | 0.609 | 7.26e+08 | 5.00e+09 | 0.145 |
| `surface_d7_t15` | 1.15e+09 | 1.91e+09 | 0.602 | 7.56e+08 | 5.36e+09 | 0.141 |
| `surface_d7_t19` | 5.89e+08 | 1.18e+09 | 0.500 | 3.74e+08 | 2.63e+09 | 0.142 |
| `surface_d7_t5` | 2.62e+07 | 3.56e+07 | 0.735 | 9.15e+06 | 5.50e+07 | 0.167 |
| `surface_d9_t10` | 2.30e+09 | 3.83e+09 | 0.601 | 1.52e+09 | 1.07e+10 | 0.142 |
| `surface_d9_t15` | 1.17e+09 | 2.38e+09 | 0.492 | 7.64e+08 | 5.44e+09 | 0.140 |
| `surface_d9_t19` | 1.22e+09 | 2.54e+09 | 0.482 | 7.87e+08 | 5.75e+09 | 0.137 |

The SALU column is the chapter's punchline in raw form: **every value is below
the VALU value in the same row, on all 26 rows.** The range 0.110–0.361 is the
corpus-scale statement of §14.3.

### 15.7 Cache and activity counters

| circuit | L2 hit V2 | L2 hit SVM | busy ratio | GRBM ratio | time ratio |
|---|---:|---:|---:|---:|---:|
| `surface_d11_t19` | 97.4 % | 79.3 % | 0.255 | 0.254 | 0.256 |
| `surface_d11_t15` | 98.7 % | 98.3 % | 0.257 | 0.257 | 0.258 |
| `surface_d9_t19` | 97.4 % | 90.0 % | 0.253 | 0.254 | 0.264 |
| `surface_d9_t15` | 98.2 % | 98.6 % | 0.267 | 0.269 | 0.275 |
| `surface_d7_t19` | 97.0 % | 98.8 % | 0.281 | 0.288 | 0.310 |
| `qv10` | 99.6 % | 99.8 % | 0.291 | 0.289 | 0.319 |
| `surface_d11_t10` | 98.7 % | 99.1 % | 0.465 | 0.465 | 0.467 |
| `surface_d9_t10` | 98.4 % | 99.1 % | 0.479 | 0.481 | 0.491 |
| `circuit_d3_p0.001` | 95.3 % | 98.1 % | 0.505 | 0.442 | 0.509 |
| `surface_d7_t15` | 97.7 % | 99.1 % | 0.485 | 0.489 | 0.510 |
| `surface_d7_t10` | 97.7 % | 99.0 % | 0.501 | 0.504 | 0.534 |
| `four_t` | **52.3 %** | 74.4 % | 0.496 | 0.533 | 0.587 |
| `surface_d7_t5` | 99.2 % | 98.9 % | 0.587 | 0.595 | 0.608 |
| `qv20_L8_seed42` | 69.8 % | 69.8 % | 0.687 | 0.683 | 0.732 |
| `qv21_L8_seed42` | 69.4 % | 69.5 % | 0.692 | 0.686 | 0.736 |
| `frame_h` | **52.2 %** | 66.8 % | 0.712 | 0.391 | 0.742 |
| `cultivation_d5` | 90.9 % | 98.9 % | 0.645 | 0.650 | 0.786 |
| `circuit_d5_p0.005` | 91.1 % | 97.8 % | 0.642 | 0.654 | 0.815 |
| `circuit_d5_p0.003` | 91.4 % | 97.6 % | 0.629 | 0.641 | 0.838 |
| `circuit_d5_p0.002` | 91.6 % | 97.7 % | 0.623 | 0.638 | 0.846 |
| `circuit_d5_p0.001` | 91.9 % | 98.0 % | 0.617 | 0.644 | 0.847 |
| `qv20_seed42` | 69.9 % | 69.9 % | 0.769 | 0.765 | 0.850 |
| `circuit_d5_p0.0005` | 92.1 % | 98.4 % | 0.615 | 0.643 | 0.856 |
| `qv24_L4_seed42` | 68.3 % | 68.5 % | 0.885 | 0.879 | 0.882 |
| `qv22_L6_seed42` | 68.9 % | 69.0 % | 0.970 | 0.983 | 0.980 |
| `qv23_L5_seed42` | 68.0 % | 68.1 % | 1.004 | 1.002 | 0.990 |

**`SQ_BUSY_CYCLES` and `GRBM_GUI_ACTIVE` agree with each other almost
everywhere** — two independent activity counters, from the same pass, tracking
within a few percent on 24 of 26 circuits. They agree with kernel time at
Pearson r = 0.942 (§14.7).

The two circuits where the *activity counters* disagree with *each other* are
`frame_h` (busy 0.712, GRBM 0.391) and `circuit_d3` (0.505 / 0.442) — the two
shortest kernels in the corpus at 12 µs and 221 µs, where `GRBM_GUI_ACTIVE`
includes ramp-up the SQ counter does not.

**The d5 anomaly, stated precisely.** On all six d5-family circuits both
activity counters say ~0.62–0.65 while kernel time says 0.79–0.86. The GPU is
*less busy* on V2 and yet takes *longer* than that reduced busy-ness predicts.
This is the same set of six circuits that carry the entire busy-cycle residual
(§14.7) and the residual L2 deficit (§14.8, 91 % vs 98 %). Three counters, one
population, one unexplained mechanism. §16 records it as the chapter's main open
question.

### 15.8 Reproduction

```bash
sbatch V2_performance/tools/bench_all.sh report-final-allfixtures
```

The label is positional; partition, `--gpus=1` and the 01:55:00 walltime are
`#SBATCH` directives in the script itself. `V2_SPECIALIZE=1` is set by the
script rather than left to the caller, because unset it measures the bytecode
interpreter and reports a fake 2–3× regression.

The script is in-tree for a reason worth repeating. Its header records that it
"produced `20260726T182433Z_report-final-postdust` and every run before it, but
was never committed — it lived in `/tmp` and was lost." Every number in this
chapter is reproducible only because the harness that produced it is now
versioned alongside the results.

Two further guarantees: the run aborts rather than continuing past a missing
fixture (§14.1), and `manifest.json` records `dirty` — **every figure above is
void if it reads `true`.** For job 50793 it reads `false`. The expected
denominator is 26 and `summary.md` states the actual one.
