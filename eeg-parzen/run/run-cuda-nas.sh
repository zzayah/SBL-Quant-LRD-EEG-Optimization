#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"'
python data/prepare_data.py

RUN_ID="$(python -c 'import secrets; print(secrets.token_hex(4))')"
echo "Run ID: $RUN_ID"

python eeg-parzen/control-logic.py \
    --output-dir data/seed-0/eeg-parzen \
    --session-id "$RUN_ID"
