"""Create subject splits, download, export, and validate all EEG datasets."""

import argparse
import json
import os
from pathlib import Path

# Keep MNE configuration local to the process instead of modifying ~/.mne.
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import moabb
import numpy as np
from moabb.datasets import BNCI2014_001, Lee2019_MI, PhysionetMI
from moabb.paradigms import LeftRightImagery

from eeg_dataset import create_subject_splits


def make_lee2019_mi() -> Lee2019_MI:
    dataset = Lee2019_MI(
        train_run=True,
        test_run=False,
        resting_state=False,
    )
    # MOABB 1.5.0 stores Lee sessions as "0"/"1" but filters them using
    # "1"/"2", silently dropping the first session.
    dataset._selected_sessions = None
    return dataset


DATASET_FACTORIES = {
    "bnci2014_001": BNCI2014_001,
    "lee2019_mi": make_lee2019_mi,
    "physionet_mi": lambda: PhysionetMI(imagined=True, executed=False),
}
LABEL_TO_ID = {"left_hand": 0, "right_hand": 1}
TARGET_SFREQ = 128.0
WINDOW_SECONDS = 3.0
TARGET_SAMPLES = int(TARGET_SFREQ * WINDOW_SECONDS)
SEED = 0
DATA_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_FACTORIES, "all"],
        default="all",
        help="Dataset to export (default: all).",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        help="Subject IDs to export (default: every subject).",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=DATA_ROOT / "moabb_raw",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATA_ROOT / "moabb_processed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing subject exports.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not run the complete integrity audit after exporting.",
    )
    return parser.parse_args()


def selected_subjects(dataset, requested: list[int] | None) -> list[int]:
    available = list(dataset.subject_list)
    if requested is None:
        subjects = available
    else:
        invalid = sorted(set(requested) - set(available))
        if invalid:
            raise ValueError(f"Invalid subject IDs: {invalid}; available: {available}")
        subjects = requested

    return subjects


def export_subject(paradigm, dataset, dataset_name: str, subject: int, output_dir: Path) -> dict:
    epochs, labels, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=[subject],
        return_epochs=True,
    )
    signals = epochs.get_data(copy=True).astype(np.float32)[..., :TARGET_SAMPLES]
    if signals.shape[-1] != TARGET_SAMPLES:
        raise RuntimeError(
            f"{dataset_name} subject {subject}: expected {TARGET_SAMPLES} samples, "
            f"received {signals.shape[-1]}"
        )

    labels = np.asarray(labels, dtype="U")
    unexpected = sorted(set(labels) - set(LABEL_TO_ID))
    if unexpected:
        raise ValueError(f"{dataset_name} subject {subject}: unexpected labels {unexpected}")
    targets = np.asarray([LABEL_TO_ID[label] for label in labels], dtype=np.int64)

    subject_id = str(subject).zfill(3)
    output_file = output_dir / f"subject_{subject_id}.npz"
    np.savez_compressed(
        output_file,
        X=signals,
        y=targets,
        labels=labels,
        channels=np.asarray(epochs.ch_names, dtype="U"),
        sfreq=np.asarray(TARGET_SFREQ, dtype=np.float32),
    )

    metadata = metadata.copy()
    metadata["label"] = labels
    metadata["target"] = targets
    metadata.to_csv(output_dir / f"subject_{subject_id}_metadata.csv", index=False)

    return {
        "subject": subject,
        "file": output_file.name,
        "trials": int(signals.shape[0]),
        "channels": int(signals.shape[1]),
        "samples": int(signals.shape[2]),
        "channel_names": list(epochs.ch_names),
        "class_counts": {
            label: int(np.sum(labels == label)) for label in LABEL_TO_ID
        },
    }


def export_dataset(
    dataset_name: str,
    output_root: Path,
    requested_subjects: list[int] | None,
    overwrite: bool,
) -> None:
    dataset = DATASET_FACTORIES[dataset_name]()
    subjects = selected_subjects(dataset, requested_subjects)
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    paradigm = LeftRightImagery(
        fmin=8.0,
        fmax=35.0,
        tmin=0.0,
        tmax=WINDOW_SECONDS,
        resample=TARGET_SFREQ,
    )
    manifest_file = output_dir / "manifest.json"
    records_by_subject = {}
    if manifest_file.exists():
        with manifest_file.open(encoding="utf-8") as file:
            existing_manifest = json.load(file)
        records_by_subject = {
            record["subject"]: record for record in existing_manifest.get("subjects", [])
        }

    def write_manifest() -> None:
        manifest = {
            "dataset": dataset_name,
            "label_mapping": LABEL_TO_ID,
            "filter_hz": [8.0, 35.0],
            "sampling_rate_hz": TARGET_SFREQ,
            "window_seconds": WINDOW_SECONDS,
            "samples": TARGET_SAMPLES,
            "subjects": [records_by_subject[key] for key in sorted(records_by_subject)],
        }
        with manifest_file.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

    for subject in subjects:
        output_file = output_dir / f"subject_{str(subject).zfill(3)}.npz"
        if output_file.exists() and not overwrite:
            print(f"Skipping existing {output_file}")
            continue
        print(f"Exporting {dataset_name} subject {subject}")
        record = export_subject(paradigm, dataset, dataset_name, subject, output_dir)
        records_by_subject[subject] = record
        write_manifest()
        print(f"Saved {output_file} with {record['trials']} trials")

    write_manifest()


def validate_dataset(dataset_name: str, output_root: Path) -> None:
    """Validate every exported subject against its manifest and metadata."""
    dataset_dir = output_root / dataset_name
    manifest_file = dataset_dir / "manifest.json"
    if not manifest_file.exists():
        raise RuntimeError(f"Missing manifest: {manifest_file}")
    with manifest_file.open(encoding="utf-8") as file:
        manifest = json.load(file)

    expected_subjects = set(DATASET_FACTORIES[dataset_name]().subject_list)
    records = {record["subject"]: record for record in manifest["subjects"]}
    missing = sorted(expected_subjects - set(records))
    if missing:
        raise RuntimeError(f"{dataset_name}: missing exported subjects {missing}")

    total_trials = 0
    for subject in sorted(expected_subjects):
        record = records[subject]
        subject_id = str(subject).zfill(3)
        npz_file = dataset_dir / record["file"]
        metadata_file = dataset_dir / f"subject_{subject_id}_metadata.csv"
        if not metadata_file.exists():
            raise RuntimeError(f"Missing metadata: {metadata_file}")
        with np.load(npz_file) as data:
            signals = data["X"]
            targets = data["y"]
            labels = data["labels"]
            channels = data["channels"]
            sfreq = float(data["sfreq"])
            expected_shape = (record["trials"], record["channels"], TARGET_SAMPLES)
            if signals.shape != expected_shape:
                raise RuntimeError(f"{npz_file}: shape does not match manifest")
            if targets.shape != labels.shape or len(targets) != len(signals):
                raise RuntimeError(f"{npz_file}: trial and label counts differ")
            if len(channels) != signals.shape[1]:
                raise RuntimeError(f"{npz_file}: channel count differs")
            if sfreq != TARGET_SFREQ:
                raise RuntimeError(f"{npz_file}: sampling rate is {sfreq}")
            if not np.isfinite(signals).all():
                raise RuntimeError(f"{npz_file}: signal contains NaN or infinity")
            if set(targets.tolist()) != {0, 1}:
                raise RuntimeError(f"{npz_file}: expected both binary classes")
        total_trials += record["trials"]
    print(f"Validated {dataset_name}: {len(expected_subjects)} subjects, {total_trials} trials")


def main() -> None:
    args = parse_args()
    split_file = create_subject_splits(SEED)
    print(f"Saved deterministic subject splits to {split_file}")
    download_root = args.download_root.resolve()
    output_root = args.output_root.resolve()
    download_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    moabb.set_download_dir(str(download_root))
    moabb.set_log_level("info")

    dataset_names = DATASET_FACTORIES if args.dataset == "all" else [args.dataset]
    if args.subjects and args.dataset == "all":
        raise ValueError("--subjects requires selecting one --dataset")

    for dataset_name in dataset_names:
        export_dataset(
            dataset_name,
            output_root,
            args.subjects,
            args.overwrite,
        )
        if not args.skip_validation and args.subjects is None:
            validate_dataset(dataset_name, output_root)


if __name__ == "__main__":
    main()
