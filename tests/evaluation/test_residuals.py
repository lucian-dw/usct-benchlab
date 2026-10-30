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


def test_prepared_statistics_reuse_norms_and_preserve_complex_holdouts(monkeypatch):
    obs = np.arange(24).reshape(2, 3, 4).astype(complex) + 2j
    obs[0, 0, 0] = np.nan
    weights = np.linspace(0, 2, obs.size).reshape(obs.shape)
    split = make_data_split(
        obs,
        weights=weights,
        receiver_indices=[1],
        frequency_indices=[1],
        exclude_reciprocal=False,
    )
    prediction = obs + 0.5j
    expected = {
        name: residual_statistics(
            prediction, obs, mask=getattr(split, name), weights=weights
        )
        for name in ("train", "validation", "receiver", "frequency", "joint")
    }
    norm_calls = []
    real_norm = np.linalg.norm

    def norm(values, *args, **kwargs):
        norm_calls.append(1)
        return real_norm(values, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", norm)
    for name in ("train", "validation"):
        assert split.statistics(prediction, name) == expected[name]
    assert len(norm_calls) == 4  # Only residual and weighted residual per partition.
    assert split.evaluate(prediction) == {
        name: expected[name] for name in ("train", "receiver", "frequency", "joint")
    }
    np.testing.assert_array_equal(
        split.train_weights, np.where(split.train, weights, 0)
    )
    assert not split.validation.flags.writeable
    assert not split.train_weights.flags.writeable


def test_explicit_contrast_observations_prepare_new_norms_without_changing_split():
    obs = np.array([[1 + 2j, 4], [2j, 3]])
    split = make_data_split(obs, receiver_indices=[1], exclude_reciprocal=False)
    predicted = obs + 1j
    original = split.evaluate(predicted)
    offset = np.full(obs.shape, 1 + 1j)
    contrast = split.evaluate(predicted - offset, obs - offset)
    assert (
        contrast["train"]["weighted_observed_norm"]
        != original["train"]["weighted_observed_norm"]
    )
    assert (
        contrast["train"]["weighted_residual_norm"]
        == original["train"]["weighted_residual_norm"]
    )
    assert split.evaluate(predicted) == original
    zero = split.evaluate(predicted - obs, np.zeros_like(obs))
    assert zero["train"]["relative_residual"] is None


def test_prepared_statistics_keep_empty_zero_norm_and_failure_distinct():
    obs = np.array([[0.0, 0.0, np.nan]])
    split = make_data_split(obs, weights=[[1, 0, 1]])
    assert (
        split.statistics(np.array([[1, np.nan, np.nan]]), "train")["relative_residual"]
        is None
    )
    assert (
        split.statistics(np.full(obs.shape, np.nan), "validation")["status"] == "empty"
    )
    with pytest.raises(FloatingPointError, match="non-finite prediction"):
        split.statistics(np.full(obs.shape, np.nan), "train")
    with pytest.raises(ValueError, match="same shape"):
        split.statistics(np.ones(3), "train")
