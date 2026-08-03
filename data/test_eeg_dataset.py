import numpy as np
import pytest
import torch

from eeg_dataset import (
    compute_channel_stats,
    make_data_loaders,
    make_final_data_loaders,
    subject_split,
)


@pytest.mark.parametrize(
    ("dataset_name", "fold", "channels"),
    [
        ("bnci2014_001", 0, 22),
        ("lee2019_mi", None, 62),
        ("physionet_mi", None, 64),
    ],
)
def test_data_loaders(dataset_name, fold, channels):
    loaders, info = make_data_loaders(
        dataset_name,
        batch_size=16,
        seed=0,
        fold=fold,
    )
    signals, targets = next(iter(loaders["train"]))
    assert signals.shape == (16, channels, 384)
    assert signals.dtype == torch.float32
    assert targets.dtype == torch.int64
    assert torch.isfinite(signals).all()
    assert set(loaders) == {"train", "validation"}
    assert info["classes"] == 2
    assert not set(info["subjects"]["train"]) & set(info["subjects"]["validation"])


def test_channel_stats_use_requested_subjects():
    split = subject_split("bnci2014_001", fold=0)
    mean, std = compute_channel_stats("bnci2014_001", split["train"])
    assert mean.shape == std.shape == (22,)
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()
    assert (std > 0).all()


def test_bnci_requires_fold():
    with pytest.raises(ValueError, match="requires fold"):
        subject_split("bnci2014_001")


@pytest.mark.parametrize("dataset_name", ["bnci2014_001", "lee2019_mi", "physionet_mi"])
def test_final_loaders_combine_development_and_lock_test(dataset_name):
    loaders, info = make_final_data_loaders(dataset_name, batch_size=16, seed=0)
    assert set(loaders) == {"train", "test"}
    assert not set(info["subjects"]["train"]) & set(info["subjects"]["test"])
    split = subject_split(dataset_name, fold=0 if dataset_name == "bnci2014_001" else None)
    if dataset_name != "bnci2014_001":
        assert set(split["validation"]) <= set(info["subjects"]["train"])
