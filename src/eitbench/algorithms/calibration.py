"""Fit background and contacts using reference voltages exclusively."""

import numpy as np
from ..operators.forward import ForwardModel
from ..solvers.lbfgsb import optimize


def calibrate(case, parameters, ledger):
    mesh, data = case.mesh, case.data
    measured = data["reference_voltages_V"]
    norm2 = float(np.sum(measured**2))
    if norm2 <= 0:
        raise ValueError("zero reference voltage norm")
    cem = parameters.variant == "cem"
    contact0 = np.broadcast_to(
        data["contact_impedance_ohm_m2"], (len(data["currents_A"]),)
    )

    def objective(x):
        background = float(np.exp(x[0]))
        contact = np.exp(x[1:]) if cem else contact0
        forward = ForwardModel(
            mesh,
            data["currents_A"],
            data["measurement_matrix"],
            float(data["thickness_m"]),
            parameters.variant,
            contact,
            ledger,
        )
        predicted, state, lu = forward.solve(np.full(len(mesh.nodes), background))
        residual = predicted - measured
        gradient, adjoint = forward.adjoint_gradient(state, lu, residual / norm2)
        g = [np.sum(gradient) * background]
        if cem:
            g.extend(forward.contact_log_gradient(state, adjoint))
        value = 0.5 * float(np.sum(residual**2)) / norm2
        return value, np.asarray(g), dict(background=background, contact=contact.copy())

    initial = [np.log(float(data["background_conductivity_S_m"]))]
    bounds = [(np.log(0.001), np.log(0.3))]
    if cem:
        initial.extend(np.log(contact0))
        bounds += [(np.log(1e-7), np.log(0.1))] * len(contact0)
    record, info = optimize(
        objective,
        np.array(initial),
        bounds,
        ledger,
        maxiter=parameters.calibration_max_iterations,
        ftol=1e-13,
        gtol=1e-8,
        maxls=30,
        maxcor=10,
    )
    info["parameter_source"] = "reference voltages only; no target or photo used"
    if record:
        info.update(
            background_conductivity_S_m=record.payload["background"],
            contact_impedance_ohm_m2=record.payload["contact"].tolist(),
            reference_relative_voltage_error=float(np.sqrt(2 * record.value)),
        )
    return record, info
