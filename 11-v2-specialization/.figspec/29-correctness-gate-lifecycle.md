# Figure 6.4 — `diagrams/correctness-gate-lifecycle.svg`

**Caption it must serve:** "Specialization is a compiler, and compilers have
bugs — so V2 does not ask you to trust it. Each freshly compiled `.hsaco` is
validated against the interpreter on a shot sample before it is allowed to run,
and on failure (or on any thrown exception in emission or compilation) the
circuit silently falls back to the interpreter. That is what makes byte-exactness
a guarantee for *every* circuit rather than for the ones that happened to be
tested. The load-bearing detail is *where the verdict is cached*: when it lived
only in-process, the gate's own validation dispatches re-ran inside every
profiled region and were summed into the kernel time — a correctness mechanism
reported as a 2–4× performance regression."

**Title:** Compile Once, Validate Once, Then Choose
**Subtitle:** And why a correctness check on the measurement path becomes a performance number

## What it must show

**Two halves, stacked.** The top half is the state machine. The bottom half is
what went wrong when one of its caches was missing.

### Top half — the lifecycle (left to right)

Five stages, with the three caches drawn as distinct storage glyphs, because
"the verdict caches in three places, not one" is the report's own emphasis.

1. **shape key** (cyan) — `tname + "_r" + peak_rank + "_n" + num_instrs`
   (`v2_kernel.cc:366-367`). Callout: the key is the circuit **shape**, not its
   name — `circuit_d5_p0.001` and `cultivation_d5` compile to the same
   `coop_r10_n1720` key, share one `.hsaco` and one verdict. *That is why
   §11.1's failure took out a whole family at once.*

2. **emit + compile** → `.hsaco` (this is **cache 3**, the binary itself).

3. **the gate** (`v2_kernel.cc:370-398`), drawn as a decision:
   `specialized_matches_interpreter(program, spath, spec_sym)` — run the circuit
   **both ways on a shot sample and compare bytes**.

4. **persist** → `<hsaco>.gate` (**cache 2**, on disk) — with the source comment
   quoted verbatim as the reason it exists:
   *"NEVER re-run during a profiled sample dispatch — the gate's own validation
   dispatches would otherwise pollute rocprofv3 kernel traces."*
   Plus the in-process `static std::map` keyed by circuit shape (**cache 1**).

5. **select** — two outcomes, drawn as two exits:
   - verdict `1` → **`clifft_v2_spec`** (green)
   - verdict `0`, *or any thrown exception in emission or compilation*
     (`v2_kernel.cc:399-403`) → **the interpreter** (orange, not red — this is
     the safe path, and it is what makes the six unsupported opcodes safe rather
     than fatal)

On a later run, a dashed fast-path arrow from **shape key** straight to
**select**, labelled *disk cache hit — the verdict is computed once ever*.

Beside the exits, the current verdict state: **16 of 16 pass**
(`build-v2-nohip/v2_spec_cache/`, 2026-07-27), against the quarantined
pre-fence cache at **36 verdicts, 2 failures** — with the caution that the two
are **not comparable on count**: 36 vs 16 is a cache rebuilt from scratch after
`009df59` changed the key, not a shrinking corpus.

### Bottom half — the pollution incident (red band)

Title: **WHEN THE VERDICT LIVED ONLY IN-PROCESS**.

Left: a `rocprofv3` invocation box with the note *spawns a fresh process per
invocation*, so the in-process map is always empty → the gate re-runs **inside
the profiled region**.

Middle: the kernel trace, verbatim, with the three dispatches drawn as three
blocks on one timeline and a brace summing them:

    noise-fenced-gated   frame_h  clifft_v2_register x1 (19.1us) + clifft_v2_spec x2 (28.8us)
                         qv10     clifft_v2_coop     x1 (1270.2us) + clifft_v2_spec x2 (1667.2us)
    noise-specialized    frame_h  clifft_v2_spec     x1 (10.5us)
                         qv10     clifft_v2_spec     x1 (1340.0us)

Label the three blocks: *one interpreter run + two specialized runs — the gate
executing the circuit both ways to compare them, plus the real sampling
dispatch.* **Three dispatches where every other run has one.**

Right: the apparent ratios, each an arrow from the true value to the inflated
one, in red:

    frame_h     0.612 -> 2.859
    circuit_d3  1.116 -> 2.117
    qv10        0.252 -> 0.675

with a hard label: **nothing had actually slowed down.**

Fix pill (green): commit `bbb5e42` persisted the verdict to `<hsaco>.gate`, so
it is computed once ever and the numbers returned to trend.

### Bottom strip

**"A correctness mechanism that runs on the measurement path becomes a
performance number."** Followed by: three of these eight circuits would have
been reported as 2–4× regressions by anyone reading the table without the commit
history.

## DATA — verbatim, invent nothing

- Shape key: `tname + "_r" + peak_rank + "_n" + num_instrs`
  (`v2_kernel.cc:366-367`). Gate: `v2_kernel.cc:370-398`. Fallback on exception:
  `v2_kernel.cc:399-403`.
- Current verdicts (2026-07-27, `build-v2-nohip/v2_spec_cache/`): **16 of 16
  pass**. Quarantined pre-fence cache
  (`V2_performance/history/stale_spec_cache_20260725/`): **36** verdicts,
  **2** failures — `coop_r10_n1720_977e1e83…`, `coop_r10_n1720_cadfdf19…`.
- Kernel trace, exactly as quoted above (19.1us / 28.8us / 1270.2us / 1667.2us /
  10.5us / 1340.0us).
- Apparent ratios: `frame_h` **0.612 → 2.859**, `circuit_d3` **1.116 → 2.117**,
  `qv10` **0.252 → 0.675**.
- Fix commit: **`bbb5e42`**. Key-change commit: **`009df59`**.
- The two source comments, quoted verbatim (the "NEVER re-run" comment and
  `bbb5e42`'s message: *"that polluted kernel traces with clifft_v2_coop+spec
  dispatches summed by the digester → inflated times."*)

## Notes

- The interpreter fallback is **orange, not red**. It is the safe path by
  design, not a failure.
- Do not draw the three caches as one. "The verdict caches in three places, not
  one" is the report's phrasing and the incident is precisely about which of the
  three was missing.
- Do not claim the gate proves correctness for all inputs — it validates on a
  **shot sample**. The report's claim is that it guarantees byte-exactness for
  every circuit *because the fallback is safe*, which is a different and weaker
  statement than "the gate proves the compiler correct".
