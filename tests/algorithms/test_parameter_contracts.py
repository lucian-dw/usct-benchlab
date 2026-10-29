"""One typed parameter boundary for Python, YAML and public execution."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from usctbench.algorithms.parameters import CGLSParameters, SIRTParameters
from usctbench.core.config import parameter_model, prepare_config
from usctbench.benchmark.runner import load_algorithm_config
from usctbench.core.registry import get_algorithm
from usctbench.core.schema import AlgorithmConfig
from usctbench.data.synthetic import make_sound_speed_case

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "name",
    [
        "straight_cgls",
        "straight_sirt",
        "straight_sart",
        "bent_ray_gn",
        "rwave_adapter",
        "fwi_wust",
    ],
)
def test_models_have_metadata_and_reject_unknown(name, tmp_path):
    model = parameter_model(name)
    for field in model.model_json_schema()["properties"].values():
        assert field["description"]
        assert field["exposure"] in {"agent", "advanced", "internal"}
        assert "units" in field
    config = AlgorithmConfig(name=name, parameters={"typo": 1})
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    case = make_sound_speed_case(shape=(8, 8), n_transducers=8)
    for admit in (
        lambda: prepare_config(name, config),
        lambda: load_algorithm_config(path),
        lambda: get_algorithm(name).run(case, config),
    ):
        with pytest.raises(ValueError):
            admit()


@pytest.mark.parametrize(
    "variant",
    ["wkb_fixed", "wkb_nonlinear", "full_green_fixed", "full_green_nonlinear"],
)
def test_born_limits_and_variant_specific_parameters(variant):
    model = parameter_model("rwave_adapter", {"variant": variant})
    config = prepare_config(
        "rwave_adapter",
        AlgorithmConfig(
            parameters={
                "variant": variant,
                "max_cache_bytes": 0,
                "green_solver_rtol": 0.5,
            },
            run_controls={"max_iterations": 2},
        ),
    )
    assert isinstance(config.parameters, model)
    assert prepare_config("rwave_adapter", config) == config
    for field, value in (
        ("max_cache_bytes", -1),
        ("green_solver_rtol", 0),
        ("green_solver_rtol", 1),
        ("green_solver_rtol", 2),
    ):
        with pytest.raises(ValueError):
            prepare_config(
                "rwave_adapter",
                AlgorithmConfig(parameters={"variant": variant, field: value}),
            )
    if variant.endswith("fixed"):
        for field in ("inner_iterations", "max_update_mps"):
            with pytest.raises(ValueError):
                model.model_validate({"variant": variant, field: 3})


@pytest.mark.parametrize(
    "values",
    [
        {"lambda": 0.2},
        {"damping": 0.04},
        {"iterations": 3},
        {"regularization_lambda": -1},
        {"regularization_lambda": float("inf")},
        {"sound_speed_bounds_mps": [1500, 1400]},
        {"inner_iterations": 10},
        {"stopping": {"max_iterations": 2}},
        {"evaluation": {}},
    ],
)
def test_removed_aliases_and_invalid_parameters_fail(values):
    with pytest.raises(ValueError):
        CGLSParameters.model_validate(values)


def test_nested_fields_and_unused_parameters_fail():
    with pytest.raises(ValueError):
        prepare_config(
            "bent_ray_gn",
            AlgorithmConfig(
                parameters={"inner_options": {"atol": 1e-8, "matlab_bin": "bad"}}
            ),
        )
    for field in ("regularization_lambda", "subsets", "damping"):
        with pytest.raises(ValueError):
            SIRTParameters.model_validate({field: 0.1})


def test_repository_configs_and_idempotent_resolution():
    for path in sorted((ROOT / "configs/algorithms").glob("*.yaml")):
        config = load_algorithm_config(path)
        assert prepare_config(config.name, config) == config, path


def test_yaml_python_and_typed_object_share_validation(tmp_path):
    parameters = {"regularization_lambda": 0.02, "regularization": "laplacian"}
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "straight_cgls",
                "parameters": parameters,
                "run_controls": {"max_iterations": 2},
            }
        )
    )
    loaded = load_algorithm_config(path)
    direct = prepare_config(
        "straight_cgls",
        AlgorithmConfig(
            name="straight_cgls",
            parameters=CGLSParameters.model_validate(parameters),
            run_controls={"max_iterations": 2},
        ),
    )
    assert loaded == direct
    capped = prepare_config("straight_cgls", direct, host_limits={"max_iterations": 1})
    assert capped.run_controls.max_iterations == 1
    assert direct.run_controls.max_iterations == 2


@pytest.mark.parametrize(
    "name", ["straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn"]
)
def test_typed_boundary_preserves_reconstruction(name):
    case = make_sound_speed_case(shape=(8, 8), n_transducers=8, inclusion_mps=1490)
    config = AlgorithmConfig(
        parameters={},
        run_controls={"max_iterations": 2, "update_rtol": None, "objective_rtol": None},
    )
    algorithm = get_algorithm(name)
    raw = algorithm.run(case, config)
    typed = algorithm.run(case, prepare_config(name, config))
    assert raw.status == typed.status == "success", typed.failure_reason
    np.testing.assert_array_equal(raw.sound_speed_mps, typed.sound_speed_mps)
    np.testing.assert_array_equal(
        raw.metrics["residual_curve"], typed.metrics["residual_curve"]
    )
    assert (
        raw.metrics["stopping"]["resolved_policy"]
        == typed.metrics["stopping"]["resolved_policy"]
    )
