"""One stopping policy and compute budget, independent of model parameters."""

from __future__ import annotations

from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field

NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonnegativeFloat = Annotated[float, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]

BUDGET_FIELDS = (
    "max_iterations",
    "max_elapsed_s",
    "max_forward_calls",
    "max_adjoint_calls",
)


class RunControls(BaseModel):
    """Budgets are checked between charged work units; the host owns hard deadlines.

    Relative update tolerances are opt-in and interpreted in the algorithm's
    declared variable, units, norm scope and normalization. GT is never a stop rule.
    """

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_default=True
    )
    max_iterations: NonnegativeInt | None = Field(
        None,
        description="Complete updates; null inherits the algorithm default, zero performs no updates.",
    )
    max_elapsed_s: NonnegativeFloat | None = Field(
        None,
        description="Cooperative seconds; WUST also has a supervised hard deadline.",
    )
    max_forward_calls: NonnegativeInt | None = Field(
        None, description="Charged forward, setup and Jacobian operations."
    )
    max_adjoint_calls: NonnegativeInt | None = Field(
        None, description="Charged adjoint operations, including normalization."
    )
    min_iterations: NonnegativeInt = 3
    update_rtol: NonnegativeFloat | None = None
    update_patience: PositiveInt = 2
    target_relative_residual: NonnegativeFloat | None = None
    noise_norm: NonnegativeFloat | None = None
    discrepancy_factor: Annotated[float, Field(strict=True, ge=1)] = 1.05
    objective_rtol: NonnegativeFloat | None = None
    objective_patience: PositiveInt = 5
    validation_patience: PositiveInt | None = None
    validation_rtol: NonnegativeFloat = 1e-4
    restore_best_validation: bool = False

    def capped(
        self,
        host_limits: RunControls | dict | None = None,
        *,
        default_iterations: int | None = None,
    ) -> RunControls:
        """Resolve the method default before intersecting independent host caps."""
        limits = RunControls.model_validate(host_limits or {})
        values = self.model_dump()
        if values["max_iterations"] is None:
            if default_iterations is None:
                raise ValueError(
                    "default_iterations is required for an inherited budget"
                )
            values["max_iterations"] = default_iterations
        for key in BUDGET_FIELDS:
            cap = getattr(limits, key)
            if cap is not None:
                values[key] = cap if values[key] is None else min(values[key], cap)
        return RunControls.model_validate(values)
