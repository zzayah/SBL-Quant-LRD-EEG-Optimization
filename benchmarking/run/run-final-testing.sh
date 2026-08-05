#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate

SEED="${1:?Usage: bash benchmarking/run/run-final-testing.sh SEED RUN_ID EPOCHS}"
RUN_ID="${2:?Usage: bash benchmarking/run/run-final-testing.sh SEED RUN_ID EPOCHS}"
EPOCHS="${3:?Usage: bash benchmarking/run/run-final-testing.sh SEED RUN_ID EPOCHS}"
RUN_ROOT="data/seed-${SEED}/benchmarking"

python benchmarking/final-testing.py \
    --model-dir "$RUN_ROOT/final-training/$RUN_ID/epochs-$EPOCHS" \
    --output-dir "$RUN_ROOT/final-testing" \
    --run-id "$RUN_ID" \
    --seed "$SEED" \
    --epochs "$EPOCHS"
