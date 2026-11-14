"""First-order PyTorch bridge for the existing discrete Eikonal linearization.

Physics stays in NumPy/CPU; tensors and returned gradients retain caller device
and dtype. A saved linearization belongs to exactly one forward invocation.
"""

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from usctbench._vendor.inv_framework.operators.base import ForwardOperator


class _EikonalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, slowness, physics, work):
        if slowness.ndim != 4 or slowness.shape[:2] != (1, 1):
            raise ValueError("Eikonal bridge accepts one case: [1,1,ny,nx]")
        if not slowness.is_floating_point():
            raise ValueError("slowness must be a floating-point tensor")
        image = slowness.detach().cpu().double().numpy()[0, 0]
        if image.shape != tuple(physics.grid.shape):
            raise ValueError("slowness shape must match the Eikonal grid")
        if work is None:
            linearization = physics.linearize(image)
        else:
            linearization = work.call("forward", physics.linearize, image)
        ctx.jacobian, ctx.work = linearization.jacobian, work
        ctx.device, ctx.dtype = slowness.device, slowness.dtype
        shape = (1, 1, *physics.ray_shape)
        return torch.as_tensor(
            np.array(linearization.value, copy=True).reshape(shape),
            device=slowness.device,
            dtype=slowness.dtype,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        values = upstream.detach().cpu().double().numpy().ravel()
        if ctx.work is None:
            gradient = ctx.jacobian.adjoint(values)
        else:
            gradient = ctx.work.call("adjoint", ctx.jacobian.adjoint, values)
        result = torch.as_tensor(
            np.array(gradient, copy=True),
            device=ctx.device,
            dtype=ctx.dtype,
        )[None, None]
        return result, None, None


class TorchEikonalOperator(ForwardOperator):
    """Physical slowness [s/m] to absolute or reference-subtracted ToF [s]."""

    def __init__(self, physics, *, offset=None, work=None):
        self.physics, self.work = physics, work
        self.domain_shape = (1, *physics.grid.shape)
        self.range_shape = (1, *physics.ray_shape)
        self.offset = (
            np.zeros(physics.ray_shape) if offset is None else np.asarray(offset)
        )
        if self.offset.shape != physics.ray_shape or not np.isfinite(self.offset).all():
            raise ValueError("offset must be finite and match [tx,rx]")

    def forward(self, slowness):
        if torch.is_grad_enabled() and slowness.requires_grad:
            prediction = _EikonalFunction.apply(slowness, self.physics, self.work)
        else:
            if slowness.shape != (1, 1, *self.physics.grid.shape):
                raise ValueError("slowness must match [1,1,ny,nx]")
            image = slowness.detach().cpu().double().numpy()[0, 0]
            values = (
                self.physics.forward(image)
                if self.work is None
                else self.work.call("forward", self.physics.forward, image)
            )
            prediction = torch.as_tensor(
                values.reshape(1, 1, *self.physics.ray_shape),
                dtype=slowness.dtype,
                device=slowness.device,
            )
        offset = torch.as_tensor(
            self.offset,
            dtype=slowness.dtype,
            device=slowness.device,
        )
        return prediction - offset[None, None]
