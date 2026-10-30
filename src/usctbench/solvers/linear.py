"""Real-parameter linearized least squares with explicit weights and ROI.

Solve [sqrt(W) J; sqrt(damping) L] d =
[weighted_residual/sqrt(W); -sqrt(damping) L current].
Complex data are real-stacked. RHS, operator and column scales are each applied
once to both the data and penalty terms; they never change the objective.
"""

from usctbench.operators.base import LinearSolveInfo
import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import LinearOperator, lsmr, lsqr
from usctbench.operators.regularization import regularizer, normal_regularizer


def solve_linearized(
    jacobian,
    residual,
    current,
    control,
    *,
    weights,
    iterations,
    damping=0.0,
    regularization="identity",
    roi=None,
    method="lsmr",
    rtol=1e-7,
    atol=1e-10,
    btol=1e-10,
    conlim=1e8,
    preconditioner="none",
    preconditioner_probes=8,
    preconditioner_seed=0,
):
    """Return a physical step and solver status, not a nonlinear convergence claim.

    residual is W * (observed - current prediction). weights are precisions.
    control supplies budgeted call() and work counts, not mathematical settings.
    """
    if method == "normal_cg":
        if preconditioner != "none":
            raise ValueError("normal_cg does not support column_rms preconditioning")
        return _normal_step(
            jacobian,
            residual,
            current,
            control,
            weights=weights,
            iterations=iterations,
            damping=damping,
            regularization=regularization,
            roi=roi,
            rtol=rtol,
        )
    if method not in {"lsmr", "lsqr"}:
        raise ValueError("inner solver must be normal_cg, lsmr or lsqr")
    return _augmented_step(
        jacobian,
        residual,
        current,
        control,
        weights=weights,
        iterations=iterations,
        damping=damping,
        regularization=regularization,
        roi=roi,
        method=method,
        rtol=rtol,
        atol=atol,
        btol=btol,
        conlim=conlim,
        preconditioner=preconditioner,
        preconditioner_probes=preconditioner_probes,
        preconditioner_seed=preconditioner_seed,
    )


def _normal_step(
    jacobian,
    residual,
    current,
    control,
    *,
    weights,
    iterations,
    damping,
    regularization,
    roi,
    rtol,
):
    precision = np.asarray(weights, dtype=float)
    active = (
        np.ones(current.shape, dtype=bool)
        if roi is None
        else np.asarray(roi, dtype=bool)
    )
    rhs = control.call(
        "adjoint", jacobian.adjoint, np.where(precision > 0, residual, 0)
    )
    rhs = np.where(
        active, rhs - damping * normal_regularizer(current, regularization), 0.0
    )
    if not np.isfinite(rhs).all():
        raise FloatingPointError("nonfinite normal-system right-hand side")
    initial_norm = float(norm(rhs.ravel()))
    if not np.isfinite(initial_norm):
        raise FloatingPointError("nonfinite normal-system right-hand side norm")
    solution = np.zeros_like(current)
    direction = rhs.copy()
    rr = float(np.vdot(rhs, rhs).real)
    if not np.isfinite(rr):
        raise FloatingPointError("nonfinite squared normal-system right-hand side norm")
    initial_rr = rr

    def normal_product(image):
        projected = control.call("jacobian", jacobian.forward, image).reshape(
            precision.shape
        )
        weighted = np.where(precision > 0, projected, 0) * precision
        if not np.isfinite(weighted).all():
            raise FloatingPointError("nonfinite training Jacobian product")
        q = control.call("adjoint", jacobian.adjoint, weighted)
        return np.where(
            active, q + damping * normal_regularizer(image, regularization), 0.0
        )

    initial_rhs = rhs.copy()
    completed = 0
    for _ in range(iterations):
        if rr <= max(np.finfo(float).tiny, initial_rr * rtol**2):
            break
        q = normal_product(direction)
        denom = float(np.vdot(direction, q).real)
        if not np.isfinite(denom) or denom <= 0:
            raise FloatingPointError("nonpositive or nonfinite normal-system curvature")
        alpha = rr / denom
        solution += alpha * direction
        rhs -= alpha * q
        next_rr = float(np.vdot(rhs, rhs).real)
        if not np.isfinite(next_rr):
            raise FloatingPointError("nonfinite normal-system residual")
        direction = rhs + (next_rr / rr) * direction
        rr = next_rr
        completed += 1
        control.work.counts["inner_iterations"] = (
            control.work.counts.get("inner_iterations", 0) + 1
        )
    # Recompute rather than reporting only the recursively updated CG residual.
    # An iteration cap is not a certificate that the GN subproblem was solved.
    true_residual = initial_rhs - normal_product(solution) if completed else initial_rhs
    true_norm = float(norm(true_residual.ravel()))
    if not np.isfinite(true_norm):
        raise FloatingPointError("nonfinite recomputed normal-system residual")
    recursive_norm = float(norm(rhs.ravel()))
    denominator = initial_norm if initial_norm > 0 else 1.0
    relative = true_norm / denominator
    # Keep the unscaled CG recurrence, but never certify a squared-norm underflow.
    squared_residual_underflow = (initial_rr == 0 and initial_norm > 0) or (
        rr == 0 and recursive_norm > 0
    )
    converged = bool(not squared_residual_underflow and relative <= rtol)
    if converged:
        stop_reason = "normal_residual_target"
    elif squared_residual_underflow or rr <= np.finfo(float).tiny:
        stop_reason = "arithmetic_precision_limit"
    else:
        stop_reason = "iteration_limit"
    return solution, LinearSolveInfo(
        method="normal_cg",
        iterations=completed,
        stop_reason=stop_reason,
        relative_residual=relative,
        residual_norm=true_norm,
        converged=converged,
    )


def _augmented_step(
    jacobian,
    residual,
    current,
    control,
    *,
    weights,
    iterations,
    damping=0.0,
    regularization="identity",
    roi=None,
    method="lsmr",
    rtol=1e-7,
    atol=1e-10,
    btol=1e-10,
    conlim=1e8,
    preconditioner="none",
    preconditioner_probes=8,
    preconditioner_seed=0,
):
    current = np.asarray(current)
    if np.iscomplexobj(current) or not np.isfinite(current).all():
        raise ValueError("current must be a finite real model")
    current = current.astype(float, copy=False)
    active = np.ones(current.shape, bool) if roi is None else np.asarray(roi, bool)
    if active.shape != current.shape:
        raise ValueError("ROI shape must match model")
    precision = np.asarray(weights, dtype=float)
    if not np.isfinite(precision).all() or np.any(precision < 0):
        raise ValueError("precision must match observations and be finite/nonnegative")
    residual = np.asarray(residual).reshape(precision.shape)
    selected = precision > 0
    sqrt_w = np.sqrt(precision[selected])
    complex_data = np.iscomplexobj(residual) or np.iscomplexobj(residual)
    ndata, nmodel = sqrt_w.size, int(active.sum())
    data_rows = ndata * (2 if complex_data else 1)
    sqrt_damping = np.sqrt(damping)
    reg_current = (
        np.asarray(regularizer(current, regularization)) if damping else np.empty(0)
    )
    reg_shape = reg_current.shape
    nreg = reg_current.size

    def checked(value):
        value = np.asarray(value)
        if np.iscomplexobj(value) or not np.isfinite(value).all():
            raise FloatingPointError("nonfinite or nonreal augmented operator product")
        return value.astype(float, copy=False)

    checked(reg_current)

    def expand(free):
        image = np.zeros(current.shape)
        image[active] = np.asarray(free).reshape(-1)
        return image

    def matvec(free):
        image = expand(free)
        prediction = np.asarray(
            control.call("jacobian", jacobian.forward, image)
        ).reshape(precision.shape)
        # Index BEFORE arithmetic: excluded channels may legitimately be NaN.
        data = sqrt_w * prediction[selected]
        parts = [data.real, data.imag] if complex_data else [data]
        if damping:
            parts.append(
                sqrt_damping * np.asarray(regularizer(image, regularization)).ravel()
            )
        return checked(np.concatenate(parts))

    def rmatvec(vector):
        vector = checked(vector).ravel()
        data = vector[:ndata]
        if complex_data:
            data = data + 1j * vector[ndata:data_rows]
        full = np.zeros(precision.shape, dtype=complex if complex_data else float)
        full[selected] = sqrt_w * data
        image = checked(control.call("adjoint", jacobian.adjoint, full)).reshape(
            current.shape
        )
        if damping:
            reg_data = vector[data_rows:].reshape(reg_shape)
            reg_adj = (
                regularization.adjoint(reg_data)
                if hasattr(regularization, "adjoint")
                else regularizer(reg_data, regularization)
            )
            image = image + sqrt_damping * checked(reg_adj)
        return checked(image[active])

    target_data = residual[selected] / sqrt_w
    parts = [target_data.real, target_data.imag] if complex_data else [target_data]
    if damping:
        parts.append(-sqrt_damping * reg_current.ravel())
    target = checked(np.concatenate(parts))
    operator = LinearOperator(
        (data_rows + nreg, nmodel), matvec=matvec, rmatvec=rmatvec, dtype=float
    )
    # Exact binary RHS rescaling protects M* h before it can underflow.
    rhs_exponent = int(np.frexp(np.max(np.abs(target), initial=0))[1])
    scaled_target = np.ldexp(target, -rhs_exponent)
    initial_gradient = operator.rmatvec(scaled_target) if nmodel else np.empty(0)
    initial_norm = norm(initial_gradient)
    if initial_norm == 0:
        return np.zeros_like(current), LinearSolveInfo(
            method, 0, "stationary_subproblem", 0.0, float(norm(target)), True
        )
    # Scale M from an operator product, not ||h||/||M*h||: a large
    # left-nullspace component of h must not amplify the entire system.
    probe_norm = norm(operator.matvec(initial_gradient / initial_norm))
    if not np.isfinite(probe_norm) or probe_norm == 0:
        raise FloatingPointError("inconsistent adjoint or unresolved operator scale")
    operator_exponent = int(np.frexp(probe_norm)[1])
    normalized = LinearOperator(
        operator.shape,
        matvec=lambda z: checked(np.ldexp(operator.matvec(z), -operator_exponent)),
        rmatvec=lambda v: checked(np.ldexp(operator.rmatvec(v), -operator_exponent)),
        dtype=float,
    )
    initial_gradient = np.ldexp(initial_gradient, -operator_exponent)
    initial_norm = norm(initial_gradient)
    scale = np.ones(nmodel)
    if preconditioner == "column_rms":
        rng = np.random.default_rng(preconditioner_seed)
        columns = np.zeros(nmodel)
        for _ in range(preconditioner_probes):
            probe = rng.choice([-1.0, 1.0], size=operator.shape[0])
            product = normalized.rmatvec(probe)
            with np.errstate(over="raise", invalid="raise"):
                columns += product**2 / preconditioner_probes
        positive = columns[columns > 0]
        if positive.size:
            # A floor bounds scaling of poorly sampled/unseen columns.
            floor = max(float(np.median(positive)) * 1e-3, np.finfo(float).tiny)
            scale = 1 / np.sqrt(np.maximum(columns, floor))
            scale /= np.median(scale)
            scale = np.clip(scale, 1e-3, 1e3)
    scaled = LinearOperator(
        operator.shape,
        matvec=lambda z: normalized.matvec(scale * z),
        rmatvec=lambda v: scale * normalized.rmatvec(v),
        dtype=float,
    )
    options = dict(atol=atol, btol=btol, conlim=conlim)
    if method == "lsmr":
        result = lsmr(scaled, scaled_target, maxiter=iterations, **options)
        z, code, completed, _, _, _, acond, _ = result
    else:
        result = lsqr(scaled, scaled_target, iter_lim=iterations, **options)
        z, code, completed, _, _, _, acond, _ = result[:8]
    control.work.counts["inner_iterations"] = control.work.counts.get(
        "inner_iterations", 0
    ) + int(completed)
    normalized_free = checked(scale * z)
    with np.errstate(over="raise", invalid="raise"):
        free = checked(np.ldexp(normalized_free, rhs_exponent - operator_exponent))

    true_residual = normalized.matvec(normalized_free) - scaled_target
    normal_norm = norm(normalized.rmatvec(true_residual))
    relative = float(normal_norm / initial_norm)
    converged = bool(code in {0, 1, 2, 4, 5} and relative <= rtol)
    reasons = {
        0: "zero_solution",
        1: "compatible_backward_error",
        2: "least_squares_backward_error",
        3: "condition_limit",
        4: "machine_compatible_error",
        5: "machine_least_squares_error",
        6: "machine_condition_limit",
        7: "iteration_limit",
    }
    reason = "normal_residual_target" if converged else reasons[int(code)]
    return expand(free), LinearSolveInfo(
        method,
        int(completed),
        reason,
        relative,
        float(np.ldexp(norm(true_residual), rhs_exponent)),
        converged,
        int(code),
        float(acond),
    )
