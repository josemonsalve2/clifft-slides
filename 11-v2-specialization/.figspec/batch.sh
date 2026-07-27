#!/usr/bin/env bash
# Run several gen.sh jobs concurrently. usage: batch.sh "spec:out" "spec:out" ...
set -u
cd "$(dirname "$0")/.."
pids=()
for pair in "$@"; do
  ./.figspec/gen.sh "${pair%%:*}" "${pair##*:}" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
