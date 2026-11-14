"""Budgets cover reference calibration and target inversion together."""

from time import perf_counter
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Count = Annotated[int, Field(strict=True, ge=0)]


class RunControls(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    max_iterations: Count = 500
    max_elapsed_s: Annotated[float, Field(ge=0)] | None = None
    max_forward_calls: Count | None = None
    max_adjoint_calls: Count | None = None

    def capped(self, caps=None):
        values = self.model_dump()
        for key, value in (caps or {}).items():
            if key not in values:
                raise ValueError(f"unsupported budget: {key}")
            if value is not None:
                values[key] = value if values[key] is None else min(values[key], value)
        return type(self).model_validate(values)


class BudgetExhausted(RuntimeError):
    pass


class WorkLedger:
    def __init__(self, controls):
        self.controls = controls
        self.started = perf_counter()
        self.forward_calls = self.adjoint_calls = 0

    def check(self):
        limit = self.controls.max_elapsed_s
        if limit is not None and perf_counter() - self.started >= limit:
            raise BudgetExhausted("time_budget")

    def charge(self, operation):
        self.check()
        name = operation + "_calls"
        limit = getattr(self.controls, "max_" + name)
        if limit is not None and getattr(self, name) >= limit:
            raise BudgetExhausted(operation + "_calls_budget")
        setattr(self, name, getattr(self, name) + 1)

    def snapshot(self):
        return dict(
            forward_calls=self.forward_calls,
            adjoint_calls=self.adjoint_calls,
            elapsed_s=perf_counter() - self.started,
        )
