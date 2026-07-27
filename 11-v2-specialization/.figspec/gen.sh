#!/usr/bin/env bash
# Drive codex to author one diagram SVG from its spec file.
# usage: gen.sh <spec-basename-without-.md> <output-svg-name>
set -eu
cd "$(dirname "$0")/.."
SPEC="$1"; OUT="$2"
mkdir -p .figspec/log
codex exec --skip-git-repo-check --ephemeral \
  -c model_reasoning_effort=high \
  -s workspace-write -C "$PWD" \
  "You are authoring one standalone SVG diagram for a technical report.

READ FIRST, in this order:
  1. .figspec/STYLE.md          — the house style. Non-negotiable.
  2. diagrams/memory-hierarchy-tiers.svg and diagrams/persistent-kernel.svg
                                — the reference implementations. Match them.
  3. .figspec/${SPEC}.md        — the spec for THIS figure, including its DATA.

Then write the file diagrams/${OUT}.

Rules:
  - Every number in the SVG must come verbatim from the spec's DATA blocks.
    Invent nothing. If the spec gives no number for something, draw it
    qualitatively without a number.
  - Hand-write the SVG. No scripts, no external assets, no web fonts.
  - Do your own layout arithmetic so nothing overflows the viewBox and no two
    text elements overlap. Estimate text width as 0.55*font-size per character
    for Arial and 0.6*font-size for monospace, and size every box to fit.
  - After writing, re-read the file you produced and check it against the
    spec's DATA line by line. Fix anything that drifted.
  - Write ONLY diagrams/${OUT}. Do not modify any other file.
Reply when done with a one-line confirmation and nothing else." \
  < /dev/null > ".figspec/log/${SPEC}.log" 2>&1
echo "[$SPEC] exit=$? -> diagrams/${OUT} $(test -f diagrams/${OUT} && wc -c < diagrams/${OUT} || echo MISSING)"
