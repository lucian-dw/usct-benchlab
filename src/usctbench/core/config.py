"""Small config-loading helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

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

    return os.path.expandvars(_ENV_DEFAULT_RE.sub(replace_default, value))
