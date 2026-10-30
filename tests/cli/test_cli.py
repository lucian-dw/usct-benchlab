"""Agent schemas are executable permissions, not just a filtered help page."""

from usctbench.core.config import prepare_config

import json

import numpy as np
import pytest

from usctbench.cli import main
from usctbench.core.algorithm_specs import (
    SPECS,
    case_capabilities,
    make_agent_config,
)
from usctbench.core.io import write_case_hdf5
from usctbench.core.registry import get_algorithm, get_algorithm_entry
from usctbench.core.schema import AlgorithmConfig
from usctbench.data.synthetic import make_sound_speed_case


@pytest.mark.parametrize("algorithm_id", sorted(SPECS))
def test_schema_defaults_permissions_and_variant_identity(algorithm_id):
    entry = get_algorithm_entry(algorithm_id)
    for variant in entry.specification.variants:
        description = entry.describe(variant.id)
        schema = description["config_schema"]
        defaults = description["default_parameters"]
        assert (
            set(description["allowed_parameters"])
            == set(defaults)
            == set(schema["properties"])
        )
        assert schema["additionalProperties"] is False
        assert description["algorithm_id"] == algorithm_id
        assert description["family"] == SPECS[algorithm_id].family
        for name, value in schema["properties"].items():
            assert value["exposure"] == "agent"
            assert defaults[name] == value["default"]
        config = make_agent_config(algorithm_id, defaults, variant=variant.id)
        assert config.name == algorithm_id
        assert "iterations" not in description["allowed_parameters"]
        assert "update_rtol" not in json.dumps(description["compute_budget"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("regularization_lambda", 0.1),
        ("lambda", 0.1),
        ("damping", 0.1),
        ("inner_options", {"atol": 0.1}),
        ("iterations", 10),
        ("_run_output_dir", "/tmp/not-allowed"),
        ("matlab_bin", "anything"),
        ("pipeline_args", ["--something"]),
        ("source_spectrum", [1]),
    ],
)
def test_agent_cannot_submit_hidden_or_unknown_controls(field, value):
    with pytest.raises(ValueError, match="not exposed"):
        make_agent_config("straight_cgls", {field: value})


def test_agent_types_enums_cross_checks_and_budget_boundary():
    for values in (
        {"regularization": "roughness"},
        {"robust_loss": "irls"},
        {"sound_speed_bounds_mps": [1600, 1400]},
    ):
        with pytest.raises(ValueError):
            make_agent_config("straight_cgls", values)
    with pytest.raises(ValueError):
        make_agent_config("straight_sirt", {"relaxation": "0.3"})
    with pytest.raises(ValueError):
        make_agent_config("straight_cgls", run_controls={"update_rtol": 0.1})
    config = make_agent_config(
        "straight_cgls",
        run_controls={"max_iterations": 3},
    )
    case = make_sound_speed_case(shape=(8, 8), n_transducers=8)
    assert config.run_controls.max_iterations == 3
    config = prepare_config(
        "straight_cgls", config, case=case, host_limits={"max_iterations": 1}
    )
    result = get_algorithm("straight_cgls").run(case, config)
    assert result.status in {"completed", "budget_exhausted"}, result.failure_reason
    assert result.metrics["iterations"] <= 1
    assert result.metrics["stopping"]["resolved_policy"]["update_rtol"] is None


def test_external_fwi_agent_admission_stays_canonical_and_strict():
    for values in (
        {"c_init": None},
        {"sos_freqs_mhz": 0.3},
        {"sos_frequencies_hz": [300000.0]},
        {"baseline_sound_speed_mps": 1490.0},
        {"initial_sound_speed_mps": "1490"},
        {"sound_speed_bounds_mps": 1500.0},
    ):
        with pytest.raises(ValueError):
            make_agent_config("fwi_wust", values)
    config = make_agent_config(
        "fwi_wust",
        {
            "initialization": "scalar",
            "initial_sound_speed_mps": 1490.0,
            "sound_speed_bounds_mps": [1400.0, 1600.0],
        },
    )
    assert config.parameters.initial_sound_speed_mps == 1490.0
    assert "baseline_sound_speed_mps" not in type(config.parameters).model_fields
    with pytest.raises(ValueError, match="initialization must be one of"):
        make_agent_config(
            "fwi_wust",
            {"initialization": "map"},
            trusted_parameters={"initial_map_mps": [[1500.0]]},
        )


def test_registry_capabilities_are_authoritative_without_name_patches():
    from dataclasses import replace
    from usctbench.core.run_controls import RunControls

    entry = replace(get_algorithm_entry("fwi_wust"), name="external_reference")
    description = entry.describe()
    assert description["algorithm_id"] == "external_reference"
    budget = description["compute_budget"]
    assert set(budget["run_controls_schema"]["properties"]) == {
        "max_iterations",
        "max_elapsed_s",
    }
    assert budget["run_controls_schema"] == budget["host_limits_schema"]
    assert budget["numerical_convergence_stopping_supported"] is False
    entry.validate_controls(RunControls(max_iterations=0, max_elapsed_s=1))
    for controls in ({"max_forward_calls": 1}, {"objective_rtol": 1e-3}):
        with pytest.raises(ValueError, match="does not support run controls"):
            entry.validate_controls(RunControls(**controls))


def test_describe_builds_only_requested_variant_schema(monkeypatch):
    from unittest.mock import Mock
    from usctbench.core import registry
    from usctbench.algorithms.parameters import FixedBornParameters

    schema = Mock(wraps=registry.agent_schema)
    monkeypatch.setattr(registry, "agent_schema", schema)
    result = get_algorithm_entry("born").describe("full_green_fixed")
    assert schema.call_count == 1
    assert schema.call_args.args[0] is FixedBornParameters
    assert "inner_options" not in json.dumps(result["config_schema"])
    result["default_parameters"]["regularization"] = "changed"
    assert (
        get_algorithm_entry("born").describe("full_green_fixed")["default_parameters"][
            "regularization"
        ]
        == "identity"
    )


def test_observation_and_runtime_variants_are_not_guessed_from_names():
    born = get_algorithm_entry("born").describe("full_green_nonlinear")
    assert born["family"] == "ray_born"
    assert born["required_observation_domains"] == ["frequency"]
    assert "freq_data" in born["required_observations"]["all_of"]
    assert born["mathematical_model"] == "distorted_born_volume_integral"
    fixed = get_algorithm_entry("born").describe("wkb_fixed")
    assert "max_update_mps" not in fixed["allowed_parameters"]
    assert fixed["compute_budget"]["default_max_iterations"] == 30
    fwi = get_algorithm_entry("fwi_wust").describe()
    assert fwi["family"] == "full_wave"
    assert fwi["compute_budget"]["online_controls_supported"] is True
    assert fwi["compute_budget"]["numerical_convergence_stopping_supported"] is False
    assert fwi["required_observations"]["any_of"] == []
    with pytest.raises(ValueError):
        make_agent_config(
            "born",
            {"variant": "full_green_nonlinear"},
            variant="wkb_nonlinear",
        )
    with pytest.raises(ValueError, match="unsupported Agent run control"):
        make_agent_config(
            "fwi_wust",
            run_controls={"max_forward_calls": 1},
        )


def test_case_bound_facts_and_policy_are_separate():
    case = make_sound_speed_case(shape=(8, 8), n_transducers=8)
    case.measurement.frequencies_hz = np.array([150000.0, 250000.0])
    static = get_algorithm_entry("straight_cgls").describe()
    config = AlgorithmConfig(
        parameters={},
        run_controls={"max_iterations": 2, "update_rtol": None, "min_iterations": 1},
    )
    bound = case_capabilities(
        "straight_cgls",
        case,
    )
    assert "available_frequencies_hz" not in static
    assert bound["available_frequencies_hz"] == [150000, 250000]
    assert bound["runtime_available"] is None
    assert "resolved_budgets" not in bound and "resolved_run_policy" not in bound
    prepared = prepare_config("straight_cgls", config, case=case)
    assert prepared.run_controls.max_iterations == 2
    assert prepared.run_controls.min_iterations == 1


def test_json_stdout_cli_and_case_capabilities(capsys, tmp_path):
    assert main(["list-algorithms", "--json"]) == 0
    output = capsys.readouterr()
    entries = json.loads(output.out)
    assert output.err == ""
    assert len(entries) == 6
    ids = {entry["algorithm_id"] for entry in entries}
    assert ids == {
        "straight_cgls",
        "straight_sirt",
        "straight_sart",
        "bent_ray_gn",
        "born",
        "fwi_wust",
    }
    path = tmp_path / "case.h5"
    write_case_hdf5(make_sound_speed_case(shape=(8, 8), n_transducers=8), path)
    assert main(["describe-algorithm", "straight_cgls", "--case", str(path)]) == 0
    assert "case_capabilities" in json.loads(capsys.readouterr().out)
    with pytest.raises(SystemExit):
        main(["describe-algorithm", "unknown"])
    output = capsys.readouterr()
    assert output.out == ""
    assert "unknown algorithm" in output.err
    with pytest.raises(SystemExit):
        main(["describe-algorithm", "born", "--json"])
    assert "unrecognized arguments: --json" in capsys.readouterr().err
    assert main(["list-algorithms"]) == 0
    assert "born:" in capsys.readouterr().out


def test_cli_reconstructs_one_case_and_reports_observation_units(tmp_path, capsys):
    path = tmp_path / "case.h5"
    write_case_hdf5(make_sound_speed_case(shape=(8, 8), n_transducers=8), path)
    config = tmp_path / "cgls.yaml"
    config.write_text("name: straight_cgls\nrun_controls:\n  max_iterations: 2\n")
    out = tmp_path / "result"
    assert (
        main(
            [
                "run",
                "straight_cgls",
                "--case",
                str(path),
                "--config",
                str(config),
                "--out",
                str(out),
                "--no-preview",
            ]
        )
        == 0
    )
    record = json.loads(capsys.readouterr().out)
    assert record["status"] != "failed"
    assert (out / "result.h5").is_file()
    assert main(["evaluate", "--run", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["records"][0]["observation_domain"] == "travel_time"
    assert report["records"][0]["observation_units"] == "s"


def test_pressure_cli_writes_reader_result_once(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from usctbench import cli

    case = SimpleNamespace(
        case_id="pressure", measurement=SimpleNamespace(time_data=np.zeros((2, 3, 8)))
    )
    reader = Mock(return_value=case)
    writer = Mock()
    monkeypatch.setattr(cli, "read_kwave_pressure_mat", reader)
    monkeypatch.setattr(cli, "write_case_hdf5", writer)
    output = str(tmp_path / "case.h5")
    assert (
        main(
            [
                "data",
                "import-pressure",
                "--input",
                "pressure.mat",
                "--out",
                output,
                "--channel-axes",
                "tx,rx,time",
                "--reference-sound-speed-mps",
                "1500",
            ]
        )
        == 0
    )
    assert reader.call_count == 1
    assert reader.call_args.args == ("pressure.mat",)
    writer.assert_called_once_with(case, output)
    assert json.loads(capsys.readouterr().out)["time_shape"] == [2, 3, 8]
