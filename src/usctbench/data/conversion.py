"""Shared image resampling and physical-coordinate conversion."""

from __future__ import annotations
import numpy as np
from usctbench.core.schema import GridSpec


def resample_bilinear(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample cell centers while preserving the full cell-edge FOV."""

    array = np.asarray(image, dtype=float)
    ny, nx = array.shape
    out_y, out_x = output_shape
    if out_y <= 0 or out_x <= 0:
        raise ValueError("output_shape must be positive")
    if (ny, nx) == (out_y, out_x):
        return array.astype(float, copy=False)
    y_src, x_src = _resampling_coordinates(array.shape, output_shape)
    y0 = np.floor(y_src).astype(int)
    x0 = np.floor(x_src).astype(int)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    wy = (y_src - y0)[:, None]
    wx = (x_src - x0)[None, :]
    top = (1.0 - wx) * array[np.ix_(y0, x0)] + wx * array[np.ix_(y0, x1)]
    bottom = (1.0 - wx) * array[np.ix_(y1, x0)] + wx * array[np.ix_(y1, x1)]
    return ((1.0 - wy) * top + wy * bottom).astype(float, copy=False)


def resample_labels_nearest(
    label: np.ndarray, output_shape: tuple[int, int]
) -> np.ndarray:
    """Nearest-neighbor labels on the same cell-edge FOV as continuous maps."""
    label = np.asarray(label)
    y_src, x_src = _resampling_coordinates(label.shape, output_shape)
    return label[np.ix_(np.rint(y_src).astype(int), np.rint(x_src).astype(int))]


def _resampling_coordinates(source_shape, output_shape):
    """Map target cell centers into source indices, preserving physical edges."""
    if len(source_shape) != 2 or len(output_shape) != 2:
        raise ValueError("resampling requires two-dimensional grids")
    if min(source_shape) <= 0 or min(output_shape) <= 0:
        raise ValueError("output_shape must be positive")
    return tuple(
        np.clip((np.arange(dst) + 0.5) * src / dst - 0.5, 0, src - 1)
        for src, dst in zip(source_shape, output_shape)
    )


def grid_from_coordinates(
    shape: tuple[int, int] | None, *, xi: np.ndarray | None, yi: np.ndarray | None
) -> GridSpec:
    """Build pixel-edge geometry from uniformly sampled physical cell centers."""
    if xi is not None and yi is not None and np.size(xi) > 1 and np.size(yi) > 1:
        axes = [np.asarray(a, dtype=float).reshape(-1) for a in (yi, xi)]
        steps = [float(np.mean(np.diff(a))) for a in axes]
        if any(
            not np.isfinite(a).all() or d <= 0 or not np.allclose(np.diff(a), d)
            for a, d in zip(axes, steps)
        ):
            raise ValueError(
                "source coordinates must be finite, uniform and increasing"
            )
        shape = tuple(len(a) for a in axes) if shape is None else tuple(shape)
        return GridSpec(
            shape=shape,
            spacing_m=tuple(d * len(a) / n for a, d, n in zip(axes, steps, shape)),
            origin_m=tuple(a[0] - d / 2 for a, d in zip(axes, steps)),
        )
    raise ValueError("xi_orig and yi_orig cell-center coordinates are required")


def _safe_case_id(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "kwave_case"
