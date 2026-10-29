"""Siddon geometry and exact discrete adjoint, without a second production path."""

import numpy as np

from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.operators.straight_ray import StraightRayProjector
from tests.helpers import adjoint_error


def test_rectangular_siddon_integrals_empty_rays_and_adjoint():
    grid = GridSpec(shape=(2, 3), spacing_m=(0.002, 0.001))
    geometry = GeometrySpec(
        tx_pos_m=[[0.001, -0.001], [0.01, -0.001]],
        rx_pos_m=[[0.001, 0.004], [0.01, 0.004]],
    )
    op = StraightRayProjector.from_grid_geometry(grid, geometry)
    # Horizontal segment crosses the three first-row cells; bottom ray misses.
    np.testing.assert_allclose(
        op.matrix.toarray()[0], [0.001] * 3 + [0] * 3, atol=1e-17
    )
    np.testing.assert_allclose(op.matrix.toarray()[3], 0)
    np.testing.assert_allclose(
        op.forward(np.full(grid.shape, 1 / 1500))[0], 0.003 / 1500
    )
    assert op.matrix.shape == (4, 6)
    rng = np.random.default_rng(723)
    assert adjoint_error(op, rng.normal(size=grid.shape), rng.normal(size=4)) < 2e-14
