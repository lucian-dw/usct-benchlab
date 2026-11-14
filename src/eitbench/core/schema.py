"""EIT quantities retain their own domains and SI units."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

from ..operators.fem import Mesh


@dataclass
class EITCase:
    mesh: Mesh
    data: dict[str, np.ndarray]
    source: Path | None = None

    def __post_init__(self):
        from .io import validate_data

        validate_data(self.mesh, self.data)

    @property
    def case_id(self):
        return str(self.data["case_id"].item())


@dataclass
class ReconstructionResult:
    conductivity_S_m: np.ndarray | None
    predicted_voltages_V: np.ndarray | None
    baseline_voltages_V: np.ndarray | None
    potential_nodes_V: np.ndarray | None
    status: str
    stopping_reason: str
    calibration: dict = field(default_factory=dict)
    optimization: dict = field(default_factory=dict)
    work: dict = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
