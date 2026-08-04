#!/bin/bash
set -e
source .venv/bin/activate
python benchmarking/run-baselines.py --baseline wide_eegnet shallow_temporal dilated_temporal --session-id baseline-seed0
