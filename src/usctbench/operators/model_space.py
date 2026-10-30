"""Explicit reduced slowness parameterization with an exact transpose.

Fine-grid propagation and measurements are unchanged: x_fine = B x_coarse.
B is nonnegative, constant-preserving cell-center bilinear interpolation with
edge replication. It is a model-space prior, not post-reconstruction filtering.
Regularization remains on the original fine grid, avoiding a hidden change of
physical penalty when the coefficient-grid size changes.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_array, kron
from usctbench.operators.regularization import prepare_regularizer

from usctbench.core.schema import GridSpec


class BilinearBasis:
    def __init__(self, grid: GridSpec, shape):
        raw = np.asarray(shape)
        if (
            raw.shape != (2,)
            or raw.dtype.kind not in "iu"
            or np.any(raw < 2)
            or np.any(raw > grid.shape)
        ):
            raise ValueError(
                "model_grid_shape must contain two integers between 2 and the fine shape"
            )
        self.fine_grid = grid
        self.shape = tuple(int(n) for n in raw)
        self.grid = GridSpec(
            shape=self.shape,
            spacing_m=tuple(
                h * n / k for h, n, k in zip(grid.spacing_m, grid.shape, self.shape)
            ),
            origin_m=grid.origin_m,
        )
        maps = []
        for n, k in zip(grid.shape, self.shape):
            coordinate = np.clip((np.arange(n) + 0.5) * k / n - 0.5, 0, k - 1)
            lower = np.minimum(np.floor(coordinate).astype(int), k - 2)
            high_weight = coordinate - lower
            maps.append(
                csr_array(
                    (
                        np.column_stack((1 - high_weight, high_weight)).ravel(),
                        (
                            np.repeat(np.arange(n), 2),
                            np.column_stack((lower, lower + 1)).ravel(),
                        ),
                    ),
                    shape=(n, k),
                )
            )
        self.matrix = kron(maps[0], maps[1], format="csr")
        self.matrix.eliminate_zeros()

    def forward(self, coefficients):
        a = np.asarray(coefficients)
        return np.asarray(self.matrix @ a.ravel()).reshape(self.fine_grid.shape)

    def adjoint(self, fine_image):
        a = np.asarray(fine_image)
        return np.asarray(self.matrix.T @ a.ravel()).reshape(self.shape)

    def metadata(self):
        return {
            "kind": "cell_center_bilinear",
            "coefficient_shape": list(self.shape),
            "propagation_shape": list(self.fine_grid.shape),
            "boundary": "edge_replication",
            "penalty_grid": "original_fine_grid",
        }


class ReducedLinearOperator:
    """Generic A B and B^T A^T, independent of the measurement layout."""

    def __init__(self, operator, basis):
        self.operator, self.basis = operator, basis
        self.data_dtype = operator.data_dtype
        self.grid, self.n_pixels = basis.grid, int(np.prod(basis.shape))

    def forward(self, image):
        return self.operator.forward(self.basis.forward(image))

    def adjoint(self, values):
        return self.basis.adjoint(self.operator.adjoint(values))


class FineGridRegularizer:
    """L B with transpose B^T L^T, including rectangular fine-grid penalties."""

    def __init__(self, basis, kind):
        self.basis, self.penalty = basis, prepare_regularizer(kind)

    def forward(self, coefficients):
        return self.penalty.forward(self.basis.forward(coefficients))

    def adjoint(self, values):
        return self.basis.adjoint(self.penalty.adjoint(values))
