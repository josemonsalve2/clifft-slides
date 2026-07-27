#!/usr/bin/env bash
# Concatenate deck_parts/*.html into index.html in slide order.
# Same rationale as assemble.sh: the parts exist so chapters can be edited
# independently; index.html is the artifact and is regenerated, never hand-edited.
set -eu
cd "$(dirname "$0")"
ORDER=(d00_head d01_title d02_svm d03_gpu_svm d05_v1 d06_v2arch
       d07_spec d08_lowering d09_opt d10_rank26 d11_pitfalls d12_gap d13_hsa
       d14_perf d15_benchall d16_conclusions d99_foot)
{
  for p in "${ORDER[@]}"; do
    cat "deck_parts/$p.html"
    printf '\n'
  done
} > index.html
n=$(grep -c '<section' index.html || true)
echo "index.html: $(wc -l < index.html) lines, ${n} slides from ${#ORDER[@]} parts"
