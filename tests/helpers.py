"""Shared independent numerical test helpers."""

import numpy as np


class DenseJacobian:
    """Independent dense real-model/complex-data operator."""

    def __init__(self, matrix, shape):
        self.matrix, self.shape = matrix, shape
        self.data_dtype = matrix.dtype

    def forward(self, x):
        return self.matrix @ x.ravel()

    def adjoint(self, y):
        return (self.matrix.conj().T @ y).real.reshape(self.shape)


def adjoint_error(operator, image: np.ndarray, data: np.ndarray) -> float:
    """Relative real-inner-product error, with no dimensional absolute floor."""
    left = float(np.vdot(operator.forward(image), data).real)
    right = float(np.vdot(image, operator.adjoint(data)).real)
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)
