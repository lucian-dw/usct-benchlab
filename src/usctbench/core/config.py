"""Small config-loading helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


def prepare_config(name, config, *, case=None, host_limits=None, deployment=None):
    """Parse each scientific payload once; intersect resolved request/host budgets."""
    from usctbench.core.schema import AlgorithmConfig
    from usctbench.core.execution import Deployment
    from usctbench.core.registry import get_algorithm_entry

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
    entry = get_algorithm_entry(name)
    model = entry.parameter_model(values)
    parameters = values if isinstance(values, model) else model.model_validate(values)
    controls = config.run_controls
    count = entry.default_iterations(parameters, case)
    # Case-dependent defaults remain unresolved during static YAML inspection.
    if count is not None or controls.max_iterations is not None:
        controls = controls.capped(host_limits, default_iterations=count)
    elif host_limits is not None:
        raise ValueError(f"case required to resolve inherited {name} iteration budget")
    entry.validate_controls(controls)
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


def _expand_env(value: str) -> str:
    def replace_default(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default)

    return os.path.expanduser(
        os.path.expandvars(_ENV_DEFAULT_RE.sub(replace_default, value))
    )
