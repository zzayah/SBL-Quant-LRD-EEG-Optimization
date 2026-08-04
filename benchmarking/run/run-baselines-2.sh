#!/bin/bash
set -e
source .venv/bin/activate
python benchmarking/run-baselines.py --baseline compact_eegnet deep_separable --session-id baseline-seed0
