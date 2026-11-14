import numpy as np
from ..core.run_controls import BudgetExhausted, WorkLedger
from ..core.schema import ReconstructionResult
from ..operators.forward import ForwardModel
from ..solvers.lbfgsb import optimize
from .calibration import calibrate


class Objective:
    def __init__(
        self,
        forward,
        measured,
        mask,
        background,
        alpha=0.03,
        tv_epsilon=0.1,
        baseline=None,
    ):
        self.forward, self.measured, self.mask = forward, measured, mask
        self.background, self.alpha, self.tv_epsilon = background, alpha, tv_epsilon
        self.baseline = np.zeros_like(measured) if baseline is None else baseline
        self.norm2 = float(np.sum(measured[mask] ** 2))
        if self.norm2 < 1e-24:
            raise ValueError("measurement norm is too small for normalized inversion")

    def __call__(self, m):
        predicted, state, lu = self.forward.solve(m * self.background)
        residual = predicted - self.baseline - self.measured
        weighted = np.where(self.mask, residual / self.norm2, 0.0)
        gradient, _ = self.forward.adjoint_gradient(state, lu, weighted)
        value = 0.5 * float(np.sum(residual[self.mask] ** 2)) / self.norm2
        tv, tv_grad = self.forward.mesh.tv(m, self.tv_epsilon)
        return (
            value + self.alpha * tv,
            self.background * gradient + self.alpha * tv_grad,
            dict(predicted=predicted, state=state),
        )


class LBFGSBTV:
    def run(self, case, config):
        parameters, controls = config.parameters, config.run_controls
        ledger = WorkLedger(controls)
        calibration, cal_info = calibrate(case, parameters, ledger)
        if calibration is None or cal_info["status"] in {
            "resource_exhausted",
            "numerical_error",
        }:
            return ReconstructionResult(
                None,
                None,
                None,
                None,
                cal_info["status"],
                cal_info["stopping_reason"],
                calibration=cal_info,
                work=ledger.snapshot(),
                metadata={"stopped_phase": "calibration"},
            )
        background, contact = (
            calibration.payload["background"],
            calibration.payload["contact"],
        )
        data = case.data
        forward = ForwardModel(
            case.mesh,
            data["currents_A"],
            data["measurement_matrix"],
            float(data["thickness_m"]),
            parameters.variant,
            contact,
            ledger,
        )
        try:
            baseline, _, _ = forward.solve(np.full(forward.n, background))
        except (BudgetExhausted, FloatingPointError, RuntimeError) as exc:
            status = (
                "resource_exhausted"
                if isinstance(exc, BudgetExhausted)
                else "numerical_error"
            )
            return ReconstructionResult(
                None,
                None,
                None,
                None,
                status,
                str(exc),
                calibration=cal_info,
                work=ledger.snapshot(),
                metadata={"stopped_phase": "baseline"},
            )
        difference = parameters.mode == "difference"
        measured = (
            data["voltages_V"] - data["reference_voltages_V"]
            if difference
            else data["voltages_V"]
        )
        objective = Objective(
            forward,
            measured,
            data["measurement_mask"],
            background,
            parameters.alpha,
            parameters.tv_epsilon,
            baseline if difference else None,
        )
        record, info = optimize(
            objective,
            np.ones(forward.n),
            [(parameters.lower, parameters.upper)] * forward.n,
            ledger,
            maxiter=controls.max_iterations,
            ftol=parameters.ftol,
            gtol=parameters.gtol,
        )
        conductivity = None if record is None else background * record.x
        if record:
            info.update(
                hit_lower_fraction=float(np.mean(record.x <= parameters.lower + 1e-6)),
                hit_upper_fraction=float(np.mean(record.x >= parameters.upper - 1e-6)),
            )
        return ReconstructionResult(
            conductivity,
            record.payload["predicted"] if record else None,
            baseline,
            record.payload["state"][: forward.n] if record else None,
            info["status"],
            info["stopping_reason"],
            cal_info,
            info,
            ledger.snapshot(),
            metadata={"stopped_phase": "reconstruction"},
        )
