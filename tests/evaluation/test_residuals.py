import json

import numpy as np
import pytest

from usctbench.evaluation import make_data_split, residual_statistics


def test_complex_residual_retains_imaginary_component():
    stats = residual_statistics(np.array([1 + 3j]), np.array([1 + 1j]))
    assert stats["residual_norm"] == 2
    assert stats["rmse"] == 2
    assert stats["relative_residual"] == pytest.approx(np.sqrt(2))
    json.dumps(stats, allow_nan=False)


def test_zero_norm_empty_and_bad_prediction_are_distinct():
    assert residual_statistics(np.ones(2), np.zeros(2))["relative_residual"] is None
    assert (
        residual_statistics(np.ones(2), np.ones(2), mask=[False, False])["status"]
        == "empty"
    )
    with pytest.raises(FloatingPointError):
        residual_statistics(np.array([np.nan, 1]), np.ones(2))
    assert (
        residual_statistics(np.array([np.nan, 1]), np.array([np.nan, 1]))["num_samples"]
        == 1
    )
    with pytest.raises(ValueError):
        residual_statistics(np.ones(2), np.ones(2), weights=[-1, 1])


def test_group_holdouts_disjoint_and_reciprocity_safe():
    positions = np.column_stack([np.arange(5), np.zeros(5)])
    obs = np.ones((3, 5, 5), dtype=complex)
    split = make_data_split(
        obs,
        receiver_indices=[1],
        frequency_indices=[2],
        tx_positions=positions,
        rx_positions=positions,
    )
    assert not split.train[:, :, 1].any()
    assert not split.train[:, 1, :].any()
    assert not split.train[2].any()
    assert split.receiver[:2, :, 1].all()
    assert split.joint[2, :, 1].all()
    masks = [split.train, split.receiver, split.frequency, split.joint, split.unused]
    np.testing.assert_array_equal(sum(m.astype(int) for m in masks), np.ones(obs.shape))
    assert split.metadata["reciprocal_tx_excluded"] == [1]
    with pytest.raises(ValueError):
        split.train[0, 0, 0] = False
    json.dumps(split.metadata, allow_nan=False)


def test_random_split_reproducible_and_no_truth_argument():
    kwargs = dict(
        receiver_fraction=0.25,
        frequency_fraction=0.25,
        seed=82,
        exclude_reciprocal=False,
    )
    a = make_data_split(np.ones((4, 6, 8)), **kwargs)
    b = make_data_split(np.ones((4, 6, 8)), **kwargs)
    assert a.metadata == b.metadata
    assert a.receiver.any() and a.frequency.any() and a.joint.any()
    assert not (a.train & a.validation).any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"receiver_indices": [0, 0]},
        {"receiver_indices": [9]},
        {"receiver_fraction": 1},
        {"receiver_indices": [0, 1, 2]},
        {"receiver_indices": [0.5]},
        {"frequency_indices": [0]},
        {"receiver_indices": [0], "receiver_fraction": 0.2},
    ],
)
def test_invalid_splits_fail(kwargs):
    with pytest.raises(ValueError):
        make_data_split(np.ones((3, 3)), exclude_reciprocal=False, **kwargs)


def test_weighted_residual_and_missing_data():
    obs = np.array([[1, 2, np.nan], [3, 4, 5]])
    split = make_data_split(obs, weights=[[4, 0, 1], [1, 1, 1]])
    stats = split.evaluate(np.nan_to_num(obs) + 1, obs)["train"]
    assert stats["num_samples"] == 4
    assert stats["weighted_residual_norm"] == pytest.approx(np.sqrt(7))
