from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..algorithms.parameters import Parameters
from .run_controls import RunControls


class Deployment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: Literal["cpu"] = "cpu"


class AlgorithmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["lbfgsb_tv"] = "lbfgsb_tv"
    parameters: Parameters = Field(default_factory=Parameters)
    run_controls: RunControls = Field(default_factory=RunControls)
    deployment: Deployment = Field(default_factory=Deployment)


def prepare_config(config=None, *, host_caps=None):
    if isinstance(config, (str, Path)):
        config = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    resolved = (
        config
        if isinstance(config, AlgorithmConfig)
        else AlgorithmConfig.model_validate(config or {})
    )
    return resolved.model_copy(
        update={"run_controls": resolved.run_controls.capped(host_caps)}
    )
