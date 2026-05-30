from __future__ import annotations

import numpy as np

from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.data.synthetic import make_grid, make_ring_geometry


def test_straight_projector_adjoint_dot_product():
    grid = make_grid(shape=(12, 10), spacing_m=(1.0e-3, 1.0e-3))
    geometry = make_ring_geometry(n_transducers=18, radius_m=0.02)
    projector = StraightRayProjector.from_grid_geometry(grid, geometry)

    rng = np.random.default_rng(123)
    image = rng.normal(size=grid.shape)
    ray_values = rng.normal(size=projector.n_rays)

    lhs = float(np.vdot(projector.forward(image), ray_values))
    rhs = float(np.vdot(image, projector.adjoint(ray_values)))

    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-12)
    assert rel_err < 1.0e-12

