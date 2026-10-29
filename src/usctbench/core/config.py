"""Small config-loading helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


def parameter_model(name, parameters=None):
    from usctbench.algorithms.parameters import (
        CGLSParameters,
        SIRTParameters,
        SARTParameters,
        BentParameters,
        BornParameters,
        FixedBornParameters,
    )
    from usctbench.algorithms.fwi.parameters import WUSTParameters

    models = {
        "straight_cgls": CGLSParameters,
        "straight_sirt": SIRTParameters,
        "straight_sart": SARTParameters,
        "bent_ray_gn": BentParameters,
        "rwave_adapter": BornParameters,
        "fwi_wust": WUSTParameters,
    }
    variant = (
        parameters.get("variant", "wkb_nonlinear")
        if isinstance(parameters, dict)
        else getattr(parameters, "variant", "wkb_nonlinear")
    )
    if name == "rwave_adapter" and variant in ("wkb_fixed", "full_green_fixed"):
        return FixedBornParameters
    try:
        return models[name]
    except KeyError as exc:
        raise ValueError(f"unknown algorithm {name!r}") from exc


def default_iterations(name, parameters=None, case=None):
    if name == "rwave_adapter":
        return (
            30
            if parameter_model(name, parameters).__name__ == "FixedBornParameters"
            else 4
        )
    if name == "fwi_wust":
        schedule = (
            parameters.get("frequency_schedule_hz")
            if isinstance(parameters, dict)
            else parameters.frequency_schedule_hz
        )
        if schedule is not None:
            return len(schedule)
        if case is None:
            return None
        frequencies = case.measurement.frequencies_hz
        return 0 if frequencies is None else len(frequencies)
    return {
        "straight_cgls": 30,
        "straight_sirt": 50,
        "straight_sart": 10,
        "bent_ray_gn": 4,
    }[name]


def prepare_config(name, config, *, case=None, host_limits=None, deployment=None):
    """Parse each scientific payload once; intersect resolved request/host budgets."""
    from usctbench.core.schema import AlgorithmConfig
    from usctbench.core.execution import Deployment

    if isinstance(config, (str, Path)):
        config = AlgorithmConfig.model_validate(expand_config_value(load_yaml(config)))
    elif not isinstance(config, AlgorithmConfig):
        config = AlgorithmConfig.model_validate(config)
    name = name or config.name
    if name is None:
        raise ValueError("algorithm name is required")
    if config.name is not None and config.name != name:
        raise ValueError(f"algorithm/config mismatch: {name!r} != {config.name!r}")
    values = expand_config_value(config.parameters)
    if isinstance(values, dict):
        # Python callers may supply an image array; typed payloads serialize lists.
        import numpy as np

        values = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in values.items()
        }
    model = parameter_model(name, values)
    parameters = values if isinstance(values, model) else model.model_validate(values)
    controls = config.run_controls
    count = default_iterations(name, parameters, case)
    # WUST's inherited frequency plan needs a case; YAML loading can stay static.
    if count is not None or controls.max_iterations is not None:
        controls = controls.capped(host_limits, default_iterations=count)
    elif host_limits is not None:
        raise ValueError("case required to resolve inherited WUST frequency budget")
    if name == "fwi_wust":
        controls.validate_for_wust()
    context = config.deployment
    if deployment is not None:
        overrides = Deployment.model_validate(deployment).model_dump(exclude_unset=True)
        context = Deployment.model_validate({**context.model_dump(), **overrides})
    return config.model_copy(
        update={
            "name": name,
            "parameters": parameters,
            "run_controls": controls,
            "deployment": context,
        }
    )


def prepared_run(function):
    """Public algorithm calls use the same preparation as YAML/runner execution."""
    from functools import wraps

    @wraps(function)
    def run(self, case, config):
        return function(self, case, prepare_config(self.name, config, case=case))

    return run


_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return payload


def expand_config_value(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env(value)
    if isinstance(value, list):
        return [expand_config_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_config_value(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_config_value(item) for key, item in value.items()}
    return value


def coerce_bool(value: Any) -> bool:
    """Parse config booleans from YAML scalars or environment-string values."""

    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def _expand_env(value: str) -> str:
    def replace_default(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default)

    return os.path.expanduser(
        os.path.expandvars(_ENV_DEFAULT_RE.sub(replace_default, value))
    )
