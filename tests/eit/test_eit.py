import numpy as np
import pytest
from pydantic import ValidationError

import eitbench as eit
from eitbench.algorithms.lbfgsb_tv import Objective
from eitbench.core.run_controls import WorkLedger, BudgetExhausted
from eitbench.operators.fem import disk_mesh
from eitbench.operators.forward import ForwardModel
from eitbench.solvers.lbfgsb import optimize


@pytest.fixture
def case():
    mesh = disk_mesh(electrodes=4, rings=2)
    currents = 1e-3 * np.array([[1, 0, -1], [-1, 1, 0], [0, -1, 1], [0, 0, 0.0]])
    measurement = np.eye(4) - np.roll(np.eye(4), 1, axis=1)
    forward = ForwardModel(mesh, currents, measurement)
    background = 0.03
    reference, _, _ = forward.solve(np.full(len(mesh.nodes), background))
    sigma = background * (
        1 + 0.4 * np.exp(-np.sum((mesh.nodes - [0.04, 0.02]) ** 2, axis=1) / 0.003)
    )
    measured, _, _ = forward.solve(sigma)
    return eit.EITCase(
        mesh,
        dict(
            nodes=mesh.nodes,
            triangles=mesh.triangles,
            boundary_edges=mesh.boundary_edges,
            electrode_ids=mesh.electrode_ids,
            currents_A=currents,
            measurement_matrix=measurement,
            voltages_V=measured,
            reference_voltages_V=reference,
            measurement_mask=np.ones_like(measured, dtype=bool),
            thickness_m=np.array(0.07),
            contact_impedance_ohm_m2=np.full(4, 0.001),
            background_conductivity_S_m=np.array(background),
            case_id=np.array("synthetic"),
        ),
    )


@pytest.mark.parametrize("variant", ["cem", "continuum"])
def test_forward_physics_and_objective_gradient(case, variant):
    d, mesh = case.data, case.mesh
    f = ForwardModel(mesh, d["currents_A"], d["measurement_matrix"], model=variant)
    m = np.random.default_rng(4).uniform(0.8, 1.2, len(mesh.nodes))
    predicted, state, lu = f.solve(0.03 * m)
    np.testing.assert_allclose(f.matrix(0.03 * m) @ state, f.rhs, atol=1e-12)
    if variant == "cem":
        np.testing.assert_allclose(state[f.n : f.n + f.l].sum(axis=0), 0, atol=1e-12)
    else:
        np.testing.assert_allclose(f.gauge @ state[: f.n], 0, atol=1e-12)
    objective = Objective(
        f,
        d["voltages_V"] - d["reference_voltages_V"],
        d["measurement_mask"],
        0.03,
        baseline=f.solve(np.full(f.n, 0.03))[0],
    )
    value, gradient, _ = objective(m)
    direction = np.random.default_rng(7).normal(size=m.shape)
    delta = 1e-6
    fd = (objective(m + delta * direction)[0] - objective(m - delta * direction)[0]) / (
        2 * delta
    )
    assert np.isfinite(predicted).all() and value > 0
    np.testing.assert_allclose(fd, gradient @ direction, rtol=2e-5, atol=1e-6)
    tv, tv_gradient = mesh.tv(m, 0.1)
    fd_tv = (
        mesh.tv(m + delta * direction, 0.1)[0] - mesh.tv(m - delta * direction, 0.1)[0]
    ) / (2 * delta)
    np.testing.assert_allclose(fd_tv, tv_gradient @ direction, rtol=1e-6)


@pytest.mark.parametrize("variant", ["cem", "continuum"])
@pytest.mark.parametrize("mode", ["difference", "absolute"])
def test_four_combinations(case, tmp_path, variant, mode):
    cfg = dict(
        parameters=dict(variant=variant, mode=mode), run_controls=dict(max_iterations=4)
    )
    result = eit.run(case=case, config=cfg, out=tmp_path / "run")
    assert result.status in {"max_iterations", "converged"}
    assert result.calibration["status"] in {"converged", "max_iterations"}
    assert (
        result.optimization["final_objective"]
        <= result.optimization["initial_objective"]
    )
    assert np.isfinite(result.conductivity_S_m).all()
    assert result.work["forward_calls"] > result.optimization["evaluations"]
    metrics = eit.evaluate(tmp_path / "run")
    assert metrics["n_measurements"] == case.data["measurement_mask"].sum()
    assert not metrics["conductivity_ground_truth_available"]
    assert (tmp_path / "run/index.json").is_file()


@pytest.mark.parametrize(
    "field", ["max_forward_calls", "max_adjoint_calls", "max_elapsed_s"]
)
def test_zero_global_budget_before_calibration(case, tmp_path, field):
    result = eit.run(
        case=case, config={"run_controls": {field: 0}}, out=tmp_path / field
    )
    assert result.status == "resource_exhausted"
    assert result.conductivity_S_m is None
    assert result.metadata["stopped_phase"] == "calibration"
    assert not (tmp_path / field / "index.json").exists()
    assert not (tmp_path / field / "result.h5").exists()
    if field.endswith("calls"):
        assert result.work[field.removeprefix("max_")] == 0


def test_interrupted_trial_never_becomes_checkpoint():
    ledger = WorkLedger(eit.RunControls(max_forward_calls=2))
    calls = []

    def objective(x):
        ledger.charge("forward")
        calls.append(x.copy())
        if len(calls) > 1:
            raise BudgetExhausted("adjoint_calls_budget")
        return float(np.sum(x * x)), 2 * x, {"predicted": x.copy()}

    record, info = optimize(
        objective,
        np.array([2.0]),
        [(0.0, 5.0)],
        ledger,
        maxiter=10,
        ftol=1e-8,
        gtol=1e-6,
    )
    assert info["status"] == "resource_exhausted"
    np.testing.assert_array_equal(record.x, [2.0])
    np.testing.assert_array_equal(record.payload["predicted"], [2.0])
    assert info["iterations"] == 0


def test_checkpoint_after_accepted_updates():
    ledger = WorkLedger(eit.RunControls(max_forward_calls=4))

    def objective(x):
        ledger.charge("forward")
        value = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        gradient = np.array(
            [
                -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                200 * (x[1] - x[0] ** 2),
            ]
        )
        return value, gradient, {"predicted": x.copy()}

    record, info = optimize(
        objective,
        np.array([-1.2, 1.0]),
        [(-5, 5)] * 2,
        ledger,
        maxiter=50,
        ftol=1e-12,
        gtol=1e-10,
    )
    assert info["status"] == "resource_exhausted"
    assert info["final_objective"] == info["history"][-1]["objective"]
    np.testing.assert_array_equal(record.payload["predicted"], record.x)
    assert ledger.forward_calls == 4


@pytest.mark.parametrize("suffix", [".npz", ".h5"])
def test_input_roundtrip(case, tmp_path, suffix):
    path = eit.save_case(case, tmp_path / ("case" + suffix))
    restored = eit.load_case(path)
    assert restored.case_id == case.case_id
    np.testing.assert_array_equal(restored.data["voltages_V"], case.data["voltages_V"])


def test_reject_float_indices_and_bad_mask(case, tmp_path):
    path = tmp_path / "bad.npz"
    data = dict(case.data, triangles=case.mesh.triangles.astype(float))
    np.savez(path, **data)
    with pytest.raises(ValueError, match="integer"):
        eit.load_case(path)
    data = dict(case.data, measurement_mask=case.data["measurement_mask"].astype(int))
    np.savez(path, **data)
    with pytest.raises(ValueError, match="boolean"):
        eit.load_case(path)


@pytest.mark.parametrize(
    "config",
    [
        {"parameters": {"variant": "bad"}},
        {"parameters": {"alpha": -1}},
        {"parameters": {"tv_epsilon": 0}},
        {"parameters": {"lower": 1, "upper": 1}},
        {"run_controls": {"max_iterations": -1}},
        {"run_controls": {"update_rtol": 0.1}},
        {"deployment": {"device": "cuda"}},
        {"parameters": {"ftol": float("nan")}},
    ],
)
def test_config_rejects_invalid_and_unsupported(config):
    with pytest.raises(ValidationError):
        eit.prepare_config(config)


def test_budget_intersection_and_discovery():
    config = eit.prepare_config(
        {"run_controls": {"max_iterations": 5, "max_forward_calls": 10}},
        host_caps={"max_iterations": 2, "max_forward_calls": 20},
    )
    assert (
        config.run_controls.max_iterations == 2
        and config.run_controls.max_forward_calls == 10
    )
    assert eit.list_algorithms() == ["lbfgsb_tv"]
    assert (
        eit.describe(variant="continuum")["default_parameters"]["variant"]
        == "continuum"
    )
    with pytest.raises(KeyError):
        eit.describe("dip")


def test_zero_iterations_is_valid_initialization(case):
    result = eit.run(case=case, config={"run_controls": {"max_iterations": 0}})
    assert result.status == "max_iterations" and result.optimization["iterations"] == 0
    assert result.optimization["success"] is False
    np.testing.assert_allclose(
        result.conductivity_S_m, result.calibration["background_conductivity_S_m"]
    )


def test_reference_only_calibration_does_not_read_target(case):
    from eitbench.algorithms.calibration import calibrate
    from eitbench.algorithms.parameters import Parameters

    parameters = Parameters(calibration_max_iterations=5)
    before, info = calibrate(case, parameters, WorkLedger(eit.RunControls()))
    case.data["voltages_V"] *= 2
    after, changed_info = calibrate(case, parameters, WorkLedger(eit.RunControls()))
    np.testing.assert_array_equal(before.x, after.x)
    assert info["history"] == changed_info["history"]


def test_cem_reciprocity_conservation_and_contact_gradient(case):
    mesh, data = case.mesh, case.data
    currents = data["currents_A"]
    measurement = currents.T / 0.001
    f = ForwardModel(mesh, currents, measurement)
    sigma = np.full(f.n, 0.03)
    voltage, state, lu = f.solve(sigma)
    np.testing.assert_allclose(voltage, voltage.T, atol=1e-12)
    electrode = state[f.n : f.n + f.l]
    recovered = (
        f.thickness
        / f.contact[:, None]
        * (f.lengths[:, None] * electrode - f.integrals.T @ state[: f.n])
    )
    np.testing.assert_allclose(recovered, currents, atol=1e-12)
    q = np.random.default_rng(3).normal(size=voltage.shape)
    _, adjoint = f.adjoint_gradient(state, lu, q)
    direction = np.random.default_rng(2).normal(size=f.l)
    gradient = f.contact_log_gradient(state, adjoint)

    def loss(step):
        changed = ForwardModel(
            mesh, currents, measurement, contact=f.contact * np.exp(step * direction)
        )
        return np.sum(changed.solve(sigma)[0] * q)

    fd = (loss(1e-5) - loss(-1e-5)) / 2e-5
    np.testing.assert_allclose(fd, gradient @ direction, rtol=1e-5, atol=1e-8)


def test_continuum_converges_to_analytic_disk(case):
    errors = []
    for rings in (3, 6):
        mesh = disk_mesh(electrodes=4, rings=rings)
        f = ForwardModel(
            mesh,
            case.data["currents_A"],
            case.data["measurement_matrix"],
            model="continuum",
        )
        predicted = f.continuum_flux(np.full(f.n, 0.03), mesh.nodes[:, 0] / 0.14)
        exact = mesh.nodes[:, 0] / (0.03 * 0.07)
        errors.append(np.linalg.norm(predicted - exact) / np.linalg.norm(exact))
    assert errors[1] < errors[0] and errors[1] < 0.01


def test_numerical_failure_retains_accepted_initial_state():
    ledger = WorkLedger(eit.RunControls())
    evaluations = 0

    def objective(x):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 1:
            raise FloatingPointError("invalid trial")
        return float(x @ x), 2 * x, {}

    record, info = optimize(
        objective,
        np.array([2.0]),
        [(0.0, 5.0)],
        ledger,
        maxiter=10,
        ftol=1e-8,
        gtol=1e-6,
    )
    assert info["status"] == "numerical_error"
    np.testing.assert_array_equal(record.x, [2.0])


def test_metered_target_interruption_saves_matching_accepted_prediction(case):
    # Determine calibration work once, then impose a cap during target line search.
    initial = eit.run(case=case, config={"run_controls": {"max_iterations": 0}})
    cap = initial.work["forward_calls"] + 2
    result = eit.run(
        case=case,
        config={"run_controls": {"max_forward_calls": cap, "max_iterations": 100}},
    )
    assert result.status == "resource_exhausted" and result.work["forward_calls"] == cap
    assert result.conductivity_S_m is not None
    d, cal = case.data, result.calibration
    f = ForwardModel(
        case.mesh,
        d["currents_A"],
        d["measurement_matrix"],
        contact=cal["contact_impedance_ohm_m2"],
    )
    np.testing.assert_allclose(
        f.solve(result.conductivity_S_m)[0], result.predicted_voltages_V, rtol=1e-12
    )


def test_optional_preview_failure_preserves_numerical_artifacts(
    case, tmp_path, monkeypatch
):
    from eitbench.benchmark import artifacts

    def unavailable(*args):
        raise ImportError("Pillow unavailable")

    monkeypatch.setattr(artifacts, "save_preview", unavailable)
    result = eit.run(
        case=case,
        config={"run_controls": {"max_iterations": 0}},
        out=tmp_path / "run",
        preview=True,
    )
    assert result.conductivity_S_m is not None
    assert "Pillow unavailable" in result.metadata["preview_error"]
    assert all(
        (tmp_path / "run" / name).is_file()
        for name in ("result.h5", "index.json", "metrics.json", "run.json")
    )
