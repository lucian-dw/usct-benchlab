"""Numerical checks for the native-physics/autograd boundary."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Import the bridge only after checking its optional PyTorch dependency.
from usctbench.core.schema import GeometrySpec, GridSpec  # noqa: E402
from usctbench.operators.eikonal import EikonalForward  # noqa: E402
from usctbench.operators.torch_eikonal import TorchEikonalOperator  # noqa: E402
from usctbench.algorithms.fttt.adapter import NormalizedSlownessOperator  # noqa: E402


def physics(order=1):
    grid = GridSpec(shape=(7, 9), spacing_m=(0.001, 0.0008))
    geometry = GeometrySpec(
        tx_pos_m=[[-0.001, 0.0012], [0.0032, -0.002]],
        rx_pos_m=[[0.0092, 0.01], [0.011, 0.0082]],
    )
    return EikonalForward(grid, geometry, spatial_order=order)


@pytest.mark.parametrize("order", [1, 2])
def test_value_vjp_chain_rule_and_saved_linearizations(order):
    native = physics(order)
    op = TorchEikonalOperator(native)
    rng = np.random.default_rng(83)
    s = (1 + 0.02 * rng.random(native.grid.shape)) / 1500
    tensor = torch.tensor(s[None, None], requires_grad=True)
    data = torch.tensor(rng.normal(size=(1, 1, 2, 2)))
    result = op.forward(tensor)
    lin = native.linearize(s)
    np.testing.assert_allclose(result.detach().numpy().ravel(), lin.value, rtol=1e-13)
    # A second forward must not replace the first graph's Jacobian.
    other = op.forward(tensor * 1.01)
    result.backward(data)
    np.testing.assert_allclose(
        tensor.grad.numpy()[0, 0],
        lin.jacobian.adjoint(data.numpy().ravel()),
        rtol=1e-12,
        atol=1e-15,
    )
    assert other.shape == result.shape and tensor.grad.dtype == tensor.dtype
    direction = rng.normal(size=s.shape) / 1500
    eps = 1e-6
    fd = (native.forward(s + eps * direction) - native.forward(s - eps * direction)) / (
        2 * eps
    )
    np.testing.assert_allclose(
        np.sum(fd * data.numpy().ravel()),
        np.sum(tensor.grad.numpy()[0, 0] * direction),
        rtol=3e-5,
        atol=1e-12,
    )
    normalized = NormalizedSlownessOperator(op, (1300, 1700), 1500)
    u = torch.tensor(
        ((s - 1 / 1700) / (1 / 1300 - 1 / 1700))[None, None], requires_grad=True
    )
    normalized.forward(u).backward(data)
    np.testing.assert_allclose(
        u.grad.numpy(),
        tensor.grad.numpy() * (1 / 1300 - 1 / 1700),
        rtol=1e-11,
        atol=1e-16,
    )


def test_roi_is_fixed_and_has_zero_network_gradient():
    native = physics()
    roi = np.zeros(native.grid.shape, bool)
    roi[2:5, 3:6] = True
    op = NormalizedSlownessOperator(
        TorchEikonalOperator(native), (1300, 1700), 1500, roi
    )
    image = torch.full(
        (1, 1, *native.grid.shape), 0.5, requires_grad=True, dtype=torch.double
    )
    op.forward(image).sum().backward()
    assert torch.count_nonzero(image.grad[0, 0][~torch.tensor(roi)]) == 0
    np.testing.assert_allclose(
        op.slowness(image).detach().numpy()[0, 0][~roi], 1 / 1500
    )
