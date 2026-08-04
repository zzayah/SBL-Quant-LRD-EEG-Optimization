# EEG baselines

This directory contains five fixed CNN baselines for comparison with EEG
Parzen NAS. All models use the same seed, subject splits, normalization,
optimizer, batch size, epoch limit, and early stopping configuration as the
NAS. Calibration allows at most 100 epochs with patience 15 and records the
best validation-balanced-accuracy epoch. The runner loads only training and
validation subjects; it does not access the locked test subjects.

The fixed models are compact and wide EEGNet-style networks, a shallow
log-variance temporal network, a deep separable network, and a dilated temporal
network. The other four retain pooled temporal positions with flatten-and-dense
classifiers. None use global temporal averaging. They are dataset-independent
except for the channel-spanning spatial input layer.

Do not run baseline training concurrently with NAS on the same accelerator.
After NAS finishes, run the baselines from two terminals on the CUDA node:

```bash
bash benchmarking/run/run-baselines-2.sh
bash benchmarking/run/run-baselines-3.sh
```

Each script trains its assigned baselines sequentially across all three
datasets under `data/seed-0/benchmarking/<dataset>/baseline-seed0/`. NAS data
remains separately under `data/seed-0/eeg-parzen/`. The runner is resumable
and skips completed entries.

To run one dataset:

```bash
python benchmarking/run-baselines.py --dataset bnci2014_001
```

Results are written to
`data/seed-<seed>/benchmarking/<dataset>/<run-id>/baseline_results.jsonl`.
The metrics include validation loss, accuracy, balanced accuracy, macro F1,
class precision and recall, confusion matrices, per-subject accuracy, model
size, parameter count, training time, epochs, inference latency, and
throughput. `figures.ipynb` reads these JSONL files and can later incorporate
the selected NAS architecture's results.

After reviewing the validation results, retrain all baselines on combined
development subjects for their median best validation epoch and evaluate the
locked tests once:

```bash
bash benchmarking/run/run-test-baselines.sh RUN_ID
```

This produces 15 final baseline models: five fixed architectures for each of
the three datasets. The runner refuses to overwrite existing locked-test
results.
