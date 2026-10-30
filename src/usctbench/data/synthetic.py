"""Small synthetic fixtures for local tests and smoke runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.core.io import write_case_hdf5
from usctbench.core.provenance import MeasurementProvenance, stamp_measurement_metadata
from usctbench.core.schema import (
    GeometrySpec,
    GridSpec,
    GroundTruthSpec,
    MeasurementSpec,
    USCTCase,
)


def make_grid(
    shape: tuple[int, int] = (32, 32),
    spacing_m: tuple[float, float] = (1.0e-3, 1.0e-3),
) -> GridSpec:
    """Create a centered Cartesian image grid."""

    ny, nx = shape
    dy, dx = spacing_m
    origin_m = (-(ny * dy) / 2.0, -(nx * dx) / 2.0)
    return GridSpec(
        shape=shape,
        spacing_m=spacing_m,
        origin_m=origin_m,
        roi_mask=np.ones(shape, dtype=bool),
    )


def make_ring_geometry(n_transducers: int = 32, radius_m: float = 0.03) -> GeometrySpec:
    """Create a ring geometry using every transducer as both source and receiver."""

    if n_transducers < 4:
        raise ValueError("n_transducers must be at least 4")
    angles = np.linspace(0.0, 2.0 * np.pi, n_transducers, endpoint=False)
    positions = np.column_stack((radius_m * np.sin(angles), radius_m * np.cos(angles)))
    return GeometrySpec(
        type="ring", tx_pos_m=positions, rx_pos_m=positions.copy(), radius_m=radius_m
    )


def circular_sound_speed(
    grid: GridSpec,
    *,
    background_mps: float = 1500.0,
    inclusion_mps: float = 1450.0,
    radius_m: float = 0.006,
    center_m: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Create a circular sound-speed phantom."""

    y0, x0 = grid.origin_m
    dy, dx = grid.spacing_m
    ny, nx = grid.shape
    y = y0 + (np.arange(ny) + 0.5) * dy
    x = x0 + (np.arange(nx) + 0.5) * dx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    mask = (yy - center_m[0]) ** 2 + (xx - center_m[1]) ** 2 <= radius_m**2
    image = np.full(grid.shape, background_mps, dtype=float)
    image[mask] = inclusion_mps
    return image


def build_surrogate_tof_case(
    grid, geometry, sound_speed_mps, reference_sound_speed_mps, case_id, *, source
):
    """Generate model-matched Siddon observations from an explicit medium.

    Images use [y,x], distances meters, sound speed m/s and delay seconds.
    Truth is retained for evaluation, never read by a reconstruction algorithm.
    """
    speed = np.asarray(sound_speed_mps, dtype=float)
    if speed.shape != grid.shape or not np.isfinite(speed).all() or np.any(speed <= 0):
        raise ValueError("sound speed must be positive, finite and match the grid")
    if not np.isfinite(reference_sound_speed_mps) or reference_sound_speed_mps <= 0:
        raise ValueError("reference sound speed must be positive and finite")
    projector = StraightRayProjector.from_grid_geometry(grid, geometry)
    delta = projector.forward(1 / speed - 1 / reference_sound_speed_mps).reshape(
        projector.ray_shape
    )
    valid = (
        np.linalg.norm(geometry.tx_pos_m[:, None] - geometry.rx_pos_m[None], axis=-1)
        > 0
    )
    return USCTCase(
        case_id=case_id,
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(
            domain="features", delta_tof_s=delta, valid_mask=valid
        ),
        ground_truth=GroundTruthSpec(sound_speed_mps=speed),
        metadata={**source, "reference_sound_speed_mps": reference_sound_speed_mps},
    )


def make_sound_speed_case(
    *,
    case_id: str = "synthetic_circular_sos",
    shape: tuple[int, int] = (32, 32),
    n_transducers: int = 32,
    background_mps: float = 1500.0,
    inclusion_mps: float = 1450.0,
    inclusion_radius_m: float = 0.006,
    inclusion_center_m: tuple[float, float] = (0.0, 0.0),
) -> USCTCase:
    """Create a feature-domain case with straight-ray travel-time differences."""

    grid = make_grid(shape=shape)
    geometry = make_ring_geometry(n_transducers=n_transducers)
    sound_speed = circular_sound_speed(
        grid,
        background_mps=background_mps,
        inclusion_mps=inclusion_mps,
        radius_m=inclusion_radius_m,
        center_m=inclusion_center_m,
    )
    return build_surrogate_tof_case(
        grid,
        geometry,
        sound_speed,
        background_mps,
        case_id,
        source=stamp_measurement_metadata(
            {
                "case_type": "synthetic_oracle",
                "synthetic": True,
                "feature_provenance": "oracle_straight_ray_forward_from_ground_truth_sound_speed",
                "reference_sound_speed_mps": background_mps,
                "roi_source": "full_grid",
                "roi_use": "reconstruction_domain",
            },
            measurement_provenance=MeasurementProvenance.ORACLE_TRAVEL_TIME,
            benchmark_type="oracle_travel_time",
            forward_model="straight_ray_oracle_from_ground_truth",
            feature_source="oracle_straight_ray_delta_tof",
            extra={
                "measurement_limitations": [
                    "fast debug / solver sanity benchmark",
                    "travel-time data are generated directly from ground truth",
                    "not a formal wavefield inversion benchmark",
                ],
            },
        ),
    )


def prepare_synthetic_demo(
    out_root: str | Path,
    *,
    shape: tuple[int, int] = (24, 24),
    n_transducers: int = 24,
    background_mps: float = 1500.0,
    inclusion_mps: float = 1450.0,
    inclusion_radius_m: float = 0.008,
) -> list[dict[str, object]]:
    """Write a deterministic circular phantom with model-matched ToF (no RNG)."""
    root = Path(out_root)
    case = make_sound_speed_case(
        shape=shape,
        n_transducers=n_transducers,
        background_mps=background_mps,
        inclusion_mps=inclusion_mps,
        inclusion_radius_m=inclusion_radius_m,
    )
    path = root / "cases" / f"{case.case_id}.h5"
    write_case_hdf5(case, path)
    records = [{"case_id": case.case_id, "path": str(path)}]
    settings = {
        "shape": list(shape),
        "n_transducers": n_transducers,
        "background_mps": background_mps,
        "inclusion_mps": inclusion_mps,
        "inclusion_radius_m": inclusion_radius_m,
        "noise": "none",
    }
    (root / "manifest.json").write_text(
        json.dumps({"cases": records, "generation": settings}, indent=2),
        encoding="utf-8",
    )
    return records
