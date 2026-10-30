"""Discrete image penalties and their Euclidean adjoints."""

import numpy as np
from types import SimpleNamespace
from scipy.sparse import csr_array, diags, eye, kron, vstack
from usctbench.core.schema import GridSpec


def prepare_regularizer(kind):
    """Resolve L/L* once; built-in cell penalties are symmetric."""
    if hasattr(kind, "forward") and hasattr(kind, "adjoint"):
        return kind
    if kind == "identity":
        return SimpleNamespace(forward=np.asarray, adjoint=np.asarray)
    if kind == "laplacian":
        coefficients = (1.0, 1.0)
    elif isinstance(kind, tuple) and len(kind) == 2:
        coefficients = kind
    else:
        raise ValueError("regularization must be identity or laplacian")

    def apply(image):
        a = np.pad(image, 1, mode="edge")
        return coefficients[0] * (
            a[:-2, 1:-1] + a[2:, 1:-1] - 2 * image
        ) + coefficients[1] * (a[1:-1, :-2] + a[1:-1, 2:] - 2 * image)

    return SimpleNamespace(forward=apply, adjoint=apply)


def regularizer(image, kind="identity"):
    return prepare_regularizer(kind).forward(image)


def normal_regularizer(image, kind="identity"):
    penalty = prepare_regularizer(kind)
    return penalty.adjoint(penalty.forward(image))


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
