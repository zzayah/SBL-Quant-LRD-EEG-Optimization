#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate

SEED="${1:?Usage: bash eeg-parzen/run/run-nas.sh SEED [RUN_ID]}"
RUN_ID="${2:-nas-seed${SEED}-$(python -c 'import secrets; print(secrets.token_hex(4))')}"

echo "Run ID: $RUN_ID"
EEG_SEED="$SEED" python eeg-parzen/control-logic.py \
    --output-dir "data/seed-${SEED}/eeg-parzen" \
    --session-id "$RUN_ID"
