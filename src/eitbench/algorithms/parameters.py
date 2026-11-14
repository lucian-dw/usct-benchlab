from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    variant: Literal["cem", "continuum"] = "cem"
    mode: Literal["difference", "absolute"] = "difference"
    alpha: Annotated[float, Field(ge=0)] = 0.03
    tv_epsilon: Annotated[float, Field(gt=0)] = 0.1
    lower: Annotated[float, Field(gt=0, le=1)] = 0.05
    upper: Annotated[float, Field(ge=1)] = 5.0
    ftol: Annotated[float, Field(gt=0)] = 1e-8
    gtol: Annotated[float, Field(gt=0)] = 1e-6
    calibration_max_iterations: Annotated[int, Field(strict=True, gt=0)] = 400

    @model_validator(mode="after")
    def bounds(self):
        if self.lower >= self.upper:
            raise ValueError("lower must be less than upper")
        return self
