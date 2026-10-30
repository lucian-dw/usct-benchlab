"""First-arrival bent-ray physics using a Cartesian fast-marching method.

Solve |grad T| = s with positive, cell-centered slowness in s/m. The accepted
upwind dependencies form a causal tape. Differentiating this *same discrete
solve* gives the Jacobian; reverse accumulation gives its exact transpose.
The default is first order; optional mixed second-order upwinding retains its
own exact discrete derivative. Point-source singularities and interfaces can
still reduce global convergence order. This is not r-Wave's shooting solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.operators.base import Linearization


def interpolation(shape: tuple[int, int], coordinates: np.ndarray):
    """Bilinear interpolation on cell-center indices, and its scatter weights."""
    points = np.asarray(coordinates, dtype=float).reshape(-1, 2)
    if not np.all(np.isfinite(points)):
        raise ValueError("interpolation coordinates must be finite")
    if np.any(points < 0) or np.any(points > np.array(shape) - 1):
        raise ValueError("interpolation points lie outside the computational grid")
    low = np.minimum(np.floor(points).astype(int), np.array(shape) - 2)
    high_weight = points - low
    indices, weights = [], []
    for iy, ix in ((0, 0), (0, 1), (1, 0), (1, 1)):
        indices.append((low[:, 0] + iy) * shape[1] + low[:, 1] + ix)
        wy = high_weight[:, 0] if iy else 1 - high_weight[:, 0]
        wx = high_weight[:, 1] if ix else 1 - high_weight[:, 1]
        weights.append(wy * wx)
    return np.array(indices).T, np.array(weights).T


@dataclass(frozen=True)
class MarchingTape:
    times: np.ndarray
    order: np.ndarray
    parents: np.ndarray
    weights: np.ndarray
    local_derivative: np.ndarray

    def tangent(self, perturbation: np.ndarray) -> np.ndarray:
        from usctbench.operators._marching import tangent_accumulate

        ds = np.asarray(perturbation, dtype=float).ravel()
        return tangent_accumulate(
            self.order, self.parents, self.weights, self.local_derivative, ds
        )


def fast_march(
    slowness: np.ndarray,
    spacing: tuple[float, float],
    source: np.ndarray,
    *,
    compiled=True,
    spatial_order=1,
    budget_check=None,
    work_counts=None,
):
    """Causal first- or mixed second-order marching with off-grid source seeding.

    Four surrounding nodes receive local constant-slowness source values. Their
    analytic derivatives are retained. Source seeding and the grid error converge
    under refinement; no straight-ray projector is used anywhere in this solve.
    """
    return PreparedMarching(
        slowness,
        spacing,
        compiled=compiled,
        spatial_order=spatial_order,
        budget_check=budget_check,
        work_counts=work_counts,
    ).solve(source)


class PreparedMarching:
    """One validated slowness/spacing pair shared by a source loop."""

    def __init__(
        self,
        slowness,
        spacing,
        *,
        compiled=True,
        spatial_order=1,
        budget_check=None,
        work_counts=None,
    ):
        if (
            isinstance(spatial_order, (bool, np.bool_))
            or not isinstance(spatial_order, (int, np.integer))
            or spatial_order not in (1, 2)
        ):
            raise ValueError("spatial_order must be 1 or 2")
        model = np.asarray(slowness, dtype=float)
        if model.ndim != 2 or min(model.shape) < 2:
            raise ValueError("fast marching requires a 2-D grid of at least 2 by 2")
        if not np.all(np.isfinite(model)) or np.any(model <= 0):
            raise ValueError("slowness must be finite and positive")
        h = np.asarray(spacing, dtype=float)
        if h.shape != (2,) or np.any(h <= 0) or not np.all(np.isfinite(h)):
            raise ValueError("spacing must be finite and positive")
        from usctbench.operators._marching import marching_arrays

        self.model, self.spacing, self.spatial_order = model, h, spatial_order
        self.march = (
            marching_arrays
            if compiled
            else getattr(marching_arrays, "py_func", marching_arrays)
        )
        self.budget_check, self.work_counts = budget_check, work_counts

    def solve(self, source):
        source = np.asarray(source, dtype=float)
        seeds, _ = interpolation(self.model.shape, source.reshape(1, 2))
        if self.budget_check is not None:
            self.budget_check()
        if self.work_counts is not None:
            self.work_counts["eikonal_solves"] = (
                self.work_counts.get("eikonal_solves", 0) + 1
            )
        return MarchingTape(
            *self.march(
                self.model, self.spacing, source, np.unique(seeds), self.spatial_order
            )
        )


class EikonalJacobian:
    """Frozen-active-set derivative with respect to slowness, not sound speed."""

    data_dtype = np.dtype(np.float64)

    def __init__(self, model, tapes):
        self.model = model
        self.grid = model.grid
        self.tapes = tuple(tapes)
        self.n_rays = model.n_rays
        self.ray_shape = model.ray_shape

    def forward(self, image: np.ndarray) -> np.ndarray:
        perturbation = self.model.extend(image, background=0.0)
        result = []
        for tape in self.tapes:
            field = tape.tangent(perturbation)
            result.append(self.model.sample_receivers(field))
        return np.asarray(result).ravel()

    def adjoint(self, data: np.ndarray) -> np.ndarray:
        from usctbench.operators._marching import reverse_accumulate

        model = self.model
        values = np.asarray(data, dtype=float)
        if values.size != model.n_rays:
            raise ValueError("data size must match (n_tx, n_rx)")
        if not np.all(np.isfinite(values)):
            raise ValueError("adjoint data must be finite")
        values = values.reshape(model.ray_shape)
        result = np.zeros(model.shape)
        for tape, receiver_values in zip(self.tapes, values):
            sensitivity = np.zeros(tape.times.size)
            np.add.at(
                sensitivity,
                model.receiver_indices.ravel(),
                (receiver_values[:, None] * model.receiver_weights).ravel(),
            )
            gradient = reverse_accumulate(
                tape.order,
                tape.parents,
                tape.weights,
                tape.local_derivative,
                sensitivity,
            )
            result += gradient.reshape(model.shape)
        return result[model.crop].copy()


class EikonalForward:
    """Nonlinear first-arrival forward map for an arbitrary 2-D acquisition.

    Geometry uses [y, x] meters and GridSpec's *cell-edge* origin. Transducers
    outside the image are included by padding with fixed background water.
    With calibrate=True, homogeneous discretization error is removed using an
    identical numerical water solve. The derivative is unchanged by calibration.
    """

    def __init__(
        self,
        grid: GridSpec,
        geometry: GeometrySpec,
        *,
        background_speed_mps: float = 1500.0,
        calibrate: bool = True,
        spatial_order: int = 1,
        budget_check=None,
        work_counts=None,
    ):
        if not np.isfinite(background_speed_mps) or background_speed_mps <= 0:
            raise ValueError("background_speed_mps must be finite and positive")
        if (
            isinstance(spatial_order, (bool, np.bool_))
            or not isinstance(spatial_order, (int, np.integer))
            or spatial_order not in (1, 2)
        ):
            raise ValueError("spatial_order must be 1 or 2")
        self.spatial_order = spatial_order
        self.budget_check, self.work_counts = budget_check, work_counts
        self.background_speed_mps = float(background_speed_mps)
        positions = np.vstack((geometry.tx_pos_m, geometry.rx_pos_m))
        if not np.all(np.isfinite(positions)):
            raise ValueError("transducer positions must be finite")
        self.grid = grid
        self.geometry = geometry
        self.background_slowness = 1 / background_speed_mps
        self.calibrate = bool(calibrate)
        h = np.array(grid.spacing_m)
        first_center = np.array(grid.origin_m) + h / 2
        coordinates = (positions - first_center) / h
        before = np.maximum(0, 2 - np.floor(coordinates.min(axis=0)).astype(int))
        after = np.maximum(
            0, np.ceil(coordinates.max(axis=0)).astype(int) + 3 - grid.shape
        )
        self.padding = tuple((int(a), int(b)) for a, b in zip(before, after))
        self.shape = tuple(np.array(grid.shape) + before + after)
        self.crop = tuple(slice(int(a), int(a + n)) for a, n in zip(before, grid.shape))
        self.source_coordinates = coordinates[: len(geometry.tx_pos_m)] + before
        self.receiver_indices, self.receiver_weights = interpolation(
            self.shape, coordinates[len(geometry.tx_pos_m) :] + before
        )
        self.ray_shape = (len(geometry.tx_pos_m), len(geometry.rx_pos_m))
        self.n_rays = int(np.prod(self.ray_shape))
        self._reference = None

    def extend(self, image: np.ndarray, *, background: float) -> np.ndarray:
        image = np.asarray(image, dtype=float)
        if image.shape != self.grid.shape:
            raise ValueError("image shape must match the reconstruction grid")
        if not np.all(np.isfinite(image)):
            raise ValueError("image contains non-finite entries")
        return np.pad(image, self.padding, constant_values=background)

    def sample_receivers(self, field: np.ndarray) -> np.ndarray:
        return np.sum(
            np.asarray(field).ravel()[self.receiver_indices] * self.receiver_weights,
            axis=1,
        )

    def _raw(self, slowness, *, retain_tapes=True):
        extended = self.extend(slowness, background=self.background_slowness)
        prepared = PreparedMarching(
            extended,
            self.grid.spacing_m,
            spatial_order=self.spatial_order,
            budget_check=self.budget_check,
            work_counts=self.work_counts,
        )
        values, tapes = [], []
        for source in self.source_coordinates:
            tape = prepared.solve(source)
            values.append(self.sample_receivers(tape.times))
            if retain_tapes:
                tapes.append(tape)
            del tape
        values = np.asarray(values).ravel()
        return values, tapes

    def linearize(self, slowness: np.ndarray) -> Linearization:
        values, tapes = self._raw(slowness)
        values = self._calibrate(values)
        return Linearization(values, EikonalJacobian(self, tapes))

    def _calibrate(self, values):
        if self.calibrate:
            if self._reference is None:
                water = np.full(self.grid.shape, self.background_slowness)
                raw, _ = self._raw(water, retain_tapes=False)
                distance = np.linalg.norm(
                    self.geometry.tx_pos_m[:, None] - self.geometry.rx_pos_m[None, :],
                    axis=-1,
                ).ravel()
                self._reference = raw
                self._water_tof = distance / self.background_speed_mps
            # Subtract matching numerical fields before adding geometric ToF.
            # Homogeneous water then cancels exactly, without a second large
            # subtraction manufacturing a roundoff-only inverse problem.
            values = (values - self._reference) + self._water_tof
        return values

    def forward(self, slowness: np.ndarray) -> np.ndarray:
        values, _ = self._raw(slowness, retain_tapes=False)
        return self._calibrate(values)
