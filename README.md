# EEG-Parzen: Neural Architecture Search with Tree-Structured Parzen Estimation for Motor-Imagery Classification

## Introduction

Electroencephalography (EEG) measures electrical activity across the scalp and has historically supported the development of sophisticated brain-computer interfaces that interpret neural signals. The selected domain of neural activity detection is motor imagery, whose decoding focuses on recognizing imagined movements, such as left- or right-hand movement, from multichannel EEG. Convolutional neural networks (CNNs) are well suited to this task because they can learn both temporal patterns and relationships between electrode channels.¹ ² However, EEG signals contain substantial noise, vary between people, and are often available only in small datasets.¹ ³ CNN architectures are therefore difficult to design without becoming unnecessarily large or overfitting to the training subjects. This study proposes neural architecture search (NAS) as a practical method for identifying promising CNN architectures capable of generalizing to unseen subjects. Specifically, the NAS uses a tree-structured Parzen estimator (TPE) to optimize architectures for left- versus right-hand motor-imagery classification.⁴ It searches EEG-specific temporal and spatial convolutional architectures independently on BNCI2014_001, Lee2019_MI, and PhysionetMI using subject-held-out balanced accuracy, then retrains selected models and evaluates them on locked test subjects. Fixed CNN baselines and cross-dataset transfer experiments provide comparisons of classification performance, model size, and inference efficiency.

## EEG-Parzen Workflow

Run experiment commands from the repository root. The scripts under
`eeg-parzen/run/` execute the complete experiment pipeline for one seed:

1. Read the seed and create or reuse a run ID.
2. Create subject-held-out splits for the three EEG datasets.
3. Run an independent Optuna TPE architecture search for each dataset.
4. Select one architecture per dataset and measure the cross-dataset development learning curves.
5. Retrain the nine architecture-target pairs from scratch on all development subjects.
6. Evaluate the frozen models once on the locked test subjects.

With the environment configured and data prepared, begin the architecture
search with:

```bash
bash eeg-parzen/run/run-nas.sh SEED [RUN_ID]
```

If no run ID is supplied, the script generates one. Supplying that run ID
again resumes the same SQLite studies toward the configured trial count.

After the search finishes, run the remaining stages in order with the same
seed and run ID:

```bash
bash eeg-parzen/run/run-calibrate.sh SEED RUN_ID
bash eeg-parzen/run/run-final-training.sh SEED RUN_ID EPOCHS
bash eeg-parzen/run/run-final-testing.sh SEED RUN_ID EPOCHS
```

Calibration measures development learning curves without loading the locked
test subjects. After reviewing those curves, choose one epoch count for final
training. Final testing should only be run after the resulting models are
frozen.

The primary experiment settings are source constants: `SEED` and `N_TRIALS`
in `eeg-parzen/control-logic.py`, and the search space and training settings in
`eeg-parzen/cnn.py`. Classification metrics are implemented in
`eeg-parzen/evaluation.py`.

## Motor-Imagery Data

Install the dependencies, then test one subject from each dataset before
beginning the full download:

```bash
python -m pip install -r requirements.txt
python data/prepare_data.py --dataset bnci2014_001 --subjects 1
python data/prepare_data.py --dataset lee2019_mi --subjects 1
python data/prepare_data.py --dataset physionet_mi --subjects 1
```

To download and prepare every supported subject, run:

```bash
python data/prepare_data.py
```

This creates `data/seed-0/splits_seed0.json`, downloads the MOABB datasets,
exports the subject arrays, and validates the complete result. The operation
is resumable. Raw MOABB data is cached under `data/moabb_raw`, while normalized
subject arrays and metadata are written under `data/moabb_processed`. Both
large locations are ignored by Git.

## References

¹ Craik et al., “Deep learning for electroencephalogram (EEG) classification tasks: a review,” J. Neural Eng., 2019.

² Lawhern et al., “EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces,” J. Neural Eng., 2018.

³ Jayaram and Barachant, “MOABB: trustworthy algorithm benchmarking for BCIs,” J. Neural Eng., 2018.

⁴ Bergstra et al., “Algorithms for Hyper-Parameter Optimization,” Advances in Neural Information Processing Systems, 2011.

## Credit

This repository adapts the Accuracy-NAS structure and Optuna workflow from
[NAS-in-the-Loop](https://github.com/zzayah/NAS-in-the-Loop). NAS-in-the-Loop
was developed by Zayah Cortright, Prateek Ganguli, Tingan Zhu, and Samarjit
Chakraborty.
