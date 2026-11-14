"""USCT contracts and real small Eikonal neural reconstructions."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from usctbench import describe, run
from usctbench.core.config import prepare_config
from usctbench.core.schema import MeasurementSpec, USCTCase, GroundTruthSpec
from usctbench.data.synthetic import make_grid, make_ring_geometry
from usctbench.operators.eikonal import EikonalForward


def make_case(shape=(9, 11), absolute=False):
    grid = make_grid(shape)
    geometry = make_ring_geometry(8, radius_m=0.009)
    speed = np.full(shape, 1500.0)
    speed[2:6, 3:7] = 1430
    forward = EikonalForward(grid, geometry)
    times = forward.forward(1 / speed).reshape(8, 8)
    distance = np.linalg.norm(
        geometry.tx_pos_m[:, None] - geometry.rx_pos_m[None], axis=-1
    )
    return USCTCase(
        case_id="independent_fttt_test",
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(
            domain="features",
            valid_mask=distance > 0,
            **(
                {"tof_s": times}
                if absolute
                else {"delta_tof_s": times - distance / 1500}
            ),
        ),
        ground_truth=GroundTruthSpec(sound_speed_mps=speed),
        metadata={"reference_sound_speed_mps": 1500.0},
    )


def test_discovery_does_not_import_torch():
    program = """
import builtins, json, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError("blocked optional torch", name='torch')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from usctbench import describe, list_algorithms
from usctbench.core.config import prepare_config
assert len(list_algorithms()) == 7
for variant in ('dip','inr','dps'):
    assert describe('fttt', variant)['variant'] == variant
    prepare_config('fttt', {'parameters': {'variant': variant}})
assert 'torch' not in sys.modules
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_variant_parameters_and_sampling_policy():
    for variant in ("dip", "inr", "dps"):
        assert describe("fttt", variant)["variant"] == variant
        with pytest.raises(ValueError):
            prepare_config("fttt", {"parameters": {"variant": variant, "typo": 1}})
    config = prepare_config(
        "fttt", {"parameters": {"variant": "dps", "sampling_steps": 7}}
    )
    assert config.run_controls.max_iterations == 7
    capped = prepare_config("fttt", config, host_limits={"max_iterations": 2})
    assert capped.run_controls.max_iterations == 2
    assert capped.parameters.sampling_steps == 7
    with pytest.raises(ValueError, match="plateau"):
        prepare_config(
            "fttt",
            {
                "parameters": {"variant": "dps"},
                "run_controls": {"objective_rtol": 0.001},
            },
        )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("variant", ["dip", "inr"])
@pytest.mark.parametrize("absolute", [False, True])
def test_real_reconstruction_and_odd_grid(variant, absolute, torch, tmp_path):
    case = make_case(absolute=absolute)
    parameters = {
        "variant": variant,
        "seed": 41,
        "learning_rate": 0.002,
        **(
            {"base_channels": 4, "input_channels": 2}
            if variant == "dip"
            else {"hidden_features": 24, "hidden_layers": 2}
        ),
    }
    result = run(
        "fttt",
        case,
        {"parameters": parameters, "run_controls": {"max_iterations": 18}},
        tmp_path / variant,
        score=False,
        preview=False,
    )
    assert result.status == "budget_exhausted", result.failure_reason
    assert result.sound_speed_mps.shape == case.grid.shape
    assert np.isfinite(result.sound_speed_mps).all()
    assert result.sound_speed_mps.min() >= 1300 and result.sound_speed_mps.max() <= 1700
    history = result.metrics["iteration_history"]
    assert history[-1]["objective"] < history[0]["objective"]
    assert result.metrics["stopping"]["work"]["adjoint_calls"] == 18
    assert result.metrics["stopping"]["completed_iterations"] == 18
    saved = json.loads((tmp_path / variant / "run.json").read_text())
    assert saved["variant"] == variant


@pytest.mark.parametrize(
    "budget",
    [
        {"max_iterations": 0},
        {"max_forward_calls": 0},
        {"max_adjoint_calls": 0},
        {"max_elapsed_s": 0},
    ],
)
def test_budgets_and_no_unrecorded_final_iterate(torch, budget):
    case = make_case()
    result = run(
        "fttt",
        case,
        {
            "parameters": {"variant": "inr", "hidden_features": 8, "hidden_layers": 1},
            "run_controls": {"max_iterations": 5, **budget},
        },
        score=False,
        preview=False,
    )
    assert result.status == "budget_exhausted", result.failure_reason
    stopping = result.metrics["stopping"]
    for counter in ("forward_calls", "adjoint_calls"):
        if "max_" + counter in budget:
            assert stopping["work"][counter] <= budget["max_" + counter]
    if stopping["has_complete_checkpoint"]:
        history = result.metrics["iteration_history"]
        native = EikonalForward(case.grid, case.geometry)
        distance = np.linalg.norm(
            case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None], axis=-1
        )
        pred = (
            native.forward(1 / result.sound_speed_mps).reshape(8, 8) - distance / 1500
        )
        mask = case.measurement.valid_mask
        residual = np.linalg.norm((pred - case.measurement.delta_tof_s)[mask])
        assert residual == pytest.approx(history[-1]["residual_norm"], rel=1e-5)


def test_masked_loss_excludes_nan_and_holdouts(torch):
    from usctbench.algorithms.fttt.adapter import WeightedTravelTimeLoss

    mask = np.array([[True, False], [False, True]])
    control = SimpleNamespace(
        split=SimpleNamespace(train=mask), precision=np.array([[1.0, 0], [0, 4.0]])
    )
    loss = WeightedTravelTimeLoss(control, 2.0, "cpu")
    prediction = torch.tensor(
        [[[[3.0, float("nan")], [99.0, 6.0]]]], requires_grad=True
    )
    observed = torch.tensor([[[[1.0, float("nan")], [float("nan"), 2.0]]]])
    loss(prediction, observed).backward()
    assert torch.isfinite(prediction.grad).all()
    np.testing.assert_allclose(prediction.grad.numpy(), [[[[0.1, 0], [0, 0.8]]]])


def test_rng_restored_and_holdout_not_used_for_training(torch):
    case = make_case()
    request = {
        "parameters": {"variant": "inr", "hidden_features": 12, "hidden_layers": 1},
        "run_controls": {"max_iterations": 2},
        "evaluation": {"receiver_indices": [0]},
    }
    before = torch.random.get_rng_state().clone()
    first = run("fttt", case, request, score=False, preview=False)
    assert first.status != "failed", first.failure_reason
    assert torch.equal(before, torch.random.get_rng_state())
    altered = case.model_copy(deep=True)
    altered.measurement.delta_tof_s[:, 0] += 0.001
    altered.measurement.delta_tof_s[0, :] += 0.001
    altered.ground_truth.sound_speed_mps[:] = 9999
    second = run("fttt", altered, request, score=False, preview=False)
    assert second.status != "failed", second.failure_reason
    np.testing.assert_array_equal(first.sound_speed_mps, second.sound_speed_mps)


def test_missing_dps_weights_fail_before_physics(torch, monkeypatch):
    case = make_case((16, 16))
    import usctbench.operators.eikonal as module

    monkeypatch.setattr(
        module.EikonalForward,
        "linearize",
        lambda *args: pytest.fail("physics before checkpoint validation"),
    )
    result = run(
        "fttt", case, {"parameters": {"variant": "dps"}}, score=False, preview=False
    )
    assert result.status == "failed"
    assert "pretrained checkpoint" in result.failure_reason


def test_schedule_ends_at_clean_sample_and_caps_are_not_rescheduling(torch):
    from usctbench._vendor.inv_framework.solvers.diffusion.scheduler import VPSchedule

    schedule = VPSchedule(8)
    schedule.set_timesteps(8)
    assert len(set(schedule.timesteps.tolist())) == 8
    x = torch.randn(1, 1, 4, 4, requires_grad=True)
    eps = torch.ones_like(x)
    previous, clean = schedule.step(eps, schedule.timesteps[-1], x)
    torch.testing.assert_close(previous, clean, rtol=0, atol=0)
    with pytest.raises(ValueError):
        schedule.set_timesteps(9)


def test_trained_prior_roundtrip_and_dps(torch, tmp_path):
    from usctbench.algorithms.fttt.train_prior import train_prior
    from usctbench.algorithms.fttt.prior import load_prior

    checkpoint = tmp_path / "small.pt"
    metadata = train_prior(output=checkpoint, steps=2, batch_size=2)
    assert metadata["trained_steps"] == 2
    _, _, provenance = load_prior(checkpoint, (16, 16), (1300, 1700), "cpu")
    assert len(provenance["sha256"]) == 64
    with pytest.raises(ValueError, match="image_shape"):
        load_prior(checkpoint, (8, 8), (1300, 1700), "cpu")
    with pytest.raises(ValueError, match="bounds"):
        load_prior(checkpoint, (16, 16), (1400, 1600), "cpu")
    case = make_case((16, 16))
    result = run(
        "fttt",
        case,
        {
            "parameters": {
                "variant": "dps",
                "checkpoint": str(checkpoint),
                "sampling_steps": 4,
            },
            "run_controls": {"max_iterations": 2},
        },
        score=False,
        preview=False,
    )
    assert result.status == "budget_exhausted", result.failure_reason
    assert result.metrics["stopping"]["completed_iterations"] == 2
    assert result.metrics["pretrained_prior"]["sha256"] == provenance["sha256"]
    assert result.metrics["stopping"]["work"]["adjoint_calls"] == 2
    payload = torch.load(checkpoint, weights_only=True)
    payload["metadata"]["training_timesteps"] = 64
    incompatible = tmp_path / "nonterminal.pt"
    torch.save(payload, incompatible)
    with pytest.raises(ValueError, match="near-Gaussian"):
        load_prior(incompatible, (16, 16), (1300, 1700), "cpu")
