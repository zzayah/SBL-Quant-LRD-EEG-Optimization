#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate

SEED="${1:?Usage: bash eeg-parzen/run/run-calibrate.sh SEED RUN_ID}"
RUN_ID="${2:?Usage: bash eeg-parzen/run/run-calibrate.sh SEED RUN_ID}"
RUN_ROOT="data/seed-${SEED}/eeg-parzen"

python eeg-parzen/calibrate-final-training.py \
    --bnci-trials "$RUN_ROOT/bnci2014_001/$RUN_ID/${RUN_ID}_bnci2014_001.jsonl" \
    --lee-trials "$RUN_ROOT/lee2019_mi/$RUN_ID/${RUN_ID}_lee2019_mi.jsonl" \
    --physionet-trials "$RUN_ROOT/physionet_mi/$RUN_ID/${RUN_ID}_physionet_mi.jsonl" \
    --output-dir "$RUN_ROOT/calibration" \
    --run-id "$RUN_ID" \
    --seed "$SEED"
