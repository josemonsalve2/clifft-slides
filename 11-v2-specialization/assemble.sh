#!/usr/bin/env bash
# Concatenate parts/*.md into REPORT.md in chapter order.
# The parts are split only so chapters can be edited and audited independently;
# REPORT.md is the artifact readers see and is regenerated, never hand-edited.
set -eu
cd "$(dirname "$0")"
ORDER=(p1_intro p2_svm p3_gpu_svm p4_v1 p5_v2_arch p6_specialization
       p7_lowering p8_optimizations p9_rank26 p10_pitfalls p11_gap p12_hsa
       p13_perf p14_benchall p15_conclusions)
{
  for p in "${ORDER[@]}"; do
    cat "parts/$p.md"
    printf '\n'
  done
} > REPORT.md
echo "REPORT.md: $(wc -l < REPORT.md) lines from ${#ORDER[@]} parts"
