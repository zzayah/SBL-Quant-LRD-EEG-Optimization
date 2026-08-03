#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:?Usage: bash eeg-parzen/run/run-test-best.sh RUN_ID}"
SEED="$(python eeg-parzen/control-logic.py --print-seed)"
RUN_ROOT="data/seed-${SEED}/eeg-parzen"

python eeg-parzen/test-best.py \
    --bnci-trials "$RUN_ROOT/bnci2014_001/$RUN_ID/${RUN_ID}_bnci2014_001.jsonl" \
    --lee-trials "$RUN_ROOT/lee2019_mi/$RUN_ID/${RUN_ID}_lee2019_mi.jsonl" \
    --physionet-trials "$RUN_ROOT/physionet_mi/$RUN_ID/${RUN_ID}_physionet_mi.jsonl" \
    --output-dir "$RUN_ROOT/test-best" \
    --run-id "$RUN_ID" \
    --seed "$SEED"
