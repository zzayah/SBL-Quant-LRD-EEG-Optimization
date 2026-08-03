# EEG data preparation

From the repository root, run:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python data/prepare_data.py
```

This one command creates `seed-0/splits_seed0.json`, downloads the original MOABB
files, exports one normalized NPZ and metadata CSV per subject, and audits all
subjects, shapes, labels, sampling rates, channels, and finite signal values.

The import is resumable: completed subject files are skipped. Use
`--dataset DATASET --subjects ID ...` for a smoke test or targeted retry. Use
`--overwrite` only when an existing processed subject must be regenerated.

The normalized representation is left-versus-right motor imagery, 8-35 Hz,
128 Hz, and 3 seconds (384 samples). Labels are `0 = left_hand` and
`1 = right_hand`.

## PyTorch loading

`eeg_dataset.py` loads complete subject partitions from
`seed-0/splits_seed0.json`.
Channel means and standard deviations are computed exclusively from the
training subjects and applied to validation data. The locked test set is only
loaded when `include_test=True` is explicitly requested.

```python
from data.eeg_dataset import make_data_loaders

loaders, info = make_data_loaders(
    "physionet_mi",
    batch_size=64,
    seed=0,
)
```

BNCI uses `fold=0`, `fold=1`, or `fold=2`. Run the loader tests from the
repository root with `pytest -q data/test_eeg_dataset.py`.
