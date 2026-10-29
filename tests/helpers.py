"""Shared independent numerical test helpers."""

import numpy as np


def adjoint_error(operator, image: np.ndarray, data: np.ndarray) -> float:
    """Relative real-inner-product error, with no dimensional absolute floor."""
    left = float(np.vdot(operator.forward(image), data).real)
    right = float(np.vdot(image, operator.adjoint(data)).real)
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)
