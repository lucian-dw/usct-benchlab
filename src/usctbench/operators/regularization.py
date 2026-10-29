"""Discrete image penalties and their Euclidean adjoints."""

import numpy as np
from scipy.sparse import csr_array, diags, eye, kron, vstack
from usctbench.core.schema import GridSpec


def regularizer(image, kind="identity"):
    if hasattr(kind, "forward") and hasattr(kind, "adjoint"):
        return kind.forward(image)
    if kind in {"identity", "l2"}:
        return np.asarray(image)
    if kind in {"laplacian", "roughness"}:
        a = np.pad(image, 1, mode="edge")
        return a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * image
    if isinstance(kind, tuple) and len(kind) == 2:
        # Dimensionless ell^2 * physical Laplacian; edge replication supplies
        # zero-normal-flux boundaries and preserves a symmetric discrete map.
        a = np.pad(image, 1, mode="edge")
        return kind[0] * (a[:-2, 1:-1] + a[2:, 1:-1] - 2 * image) + kind[1] * (
            a[1:-1, :-2] + a[1:-1, 2:] - 2 * image
        )
    raise ValueError("regularization must be identity or laplacian")


def normal_regularizer(image, kind="identity"):
    if hasattr(kind, "forward") and hasattr(kind, "adjoint"):
        return kind.adjoint(kind.forward(image))
    return (
        image
        if kind in {"identity", "l2"}
        else regularizer(regularizer(image, kind), kind)
    )


class SpatialGradient:
    """Dimensionless ell * gradient on interior edges, with its exact transpose.

    Edges use [row=y, col=x] spacing. No differences against an artificial
    zero-valued exterior are added, so a constant image has zero penalty.
    """

    def __init__(self, grid: GridSpec, length_m: float):
        if not np.isfinite(length_m) or length_m <= 0:
            raise ValueError("gradient length_m must be finite and positive")
        self.grid, self.length_m = grid, float(length_m)
        ny, nx = grid.shape

        def difference(n, spacing):
            if n == 1:
                return csr_array((0, 1))
            return diags(
                [-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)
            ) * (length_m / spacing)

        self.matrix = csr_array(
            vstack(
                [
                    kron(difference(ny, grid.spacing_m[0]), eye(nx)),
                    kron(eye(ny), difference(nx, grid.spacing_m[1])),
                ],
                format="csr",
            )
        )

    def forward(self, image):
        a = np.asarray(image)
        if a.shape != self.grid.shape or np.iscomplexobj(a) or not np.isfinite(a).all():
            raise ValueError("gradient input must be a finite real image")
        return np.asarray(self.matrix @ a.ravel())

    def adjoint(self, edges):
        a = np.asarray(edges)
        if (
            a.shape != (self.matrix.shape[0],)
            or np.iscomplexobj(a)
            or not np.isfinite(a).all()
        ):
            raise ValueError("gradient adjoint input must be a finite real edge vector")
        return np.asarray(self.matrix.T @ a).reshape(self.grid.shape)
