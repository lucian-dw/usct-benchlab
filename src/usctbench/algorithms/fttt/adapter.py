"""USCT observation, physical-variable, execution and result adapters."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from usctbench.algorithms._control import InversionControl
from usctbench.algorithms.ray import (
    configured_ray_weights,
    reference_sound_speed,
    speed_bounds,
)
from usctbench.core.schema import ReconstructionResult
from usctbench.core.stopping import BudgetExhausted
from usctbench.operators.eikonal import EikonalForward
from usctbench.operators.torch_eikonal import TorchEikonalOperator
from usctbench._vendor.inv_framework.operators.base import ForwardOperator
from usctbench._vendor.inv_framework.models.skip_unet import SkipUNet
from usctbench._vendor.inv_framework.solvers.dip import DIPSolver
from usctbench._vendor.inv_framework.solvers.inr import INRSolver
from usctbench._vendor.inv_framework.solvers.diffusion.conditioning import (
    ConditioningMethod,
)
from usctbench._vendor.inv_framework.solvers.diffusion.dps import DPSSolver


class NormalizedSlownessOperator(ForwardOperator):
    """u in [0,1] -> s_min + u*(s_max-s_min); out-of-range DPS estimates clip.

    DIP and INR already have sigmoid output. DPS trains directly on u images;
    its clean estimates use a piecewise differentiable clip at physical bounds.
    """

    def __init__(self, physical, bounds, reference_speed, roi=None):
        self.physical = physical
        self.domain_shape, self.range_shape = (
            physical.domain_shape,
            physical.range_shape,
        )
        self.lower, self.upper = 1 / bounds[1], 1 / bounds[0]
        self.reference = 1 / reference_speed
        self.roi = None if roi is None else np.asarray(roi, dtype=bool)

    def slowness(self, image):
        if image.shape != (1, *self.domain_shape) or not torch.isfinite(image).all():
            raise FloatingPointError("network image must be finite [1,1,ny,nx]")
        value = self.lower + image.clamp(0, 1) * (self.upper - self.lower)
        if self.roi is not None:
            mask = torch.as_tensor(self.roi, device=image.device)
            value = torch.where(mask[None, None], value, self.reference)
        return value

    def forward(self, image):
        return self.physical.forward(self.slowness(image))


class WeightedTravelTimeLoss:
    """Training-only weighted mean square in fixed residual_scale_s units."""

    def __init__(self, control, scale_s, device):
        self.mask = torch.tensor(control.split.train.copy(), device=device)
        self.weights = torch.tensor(
            control.precision[control.split.train].copy(),
            dtype=torch.float32,
            device=device,
        )
        self.weight_sum = self.weights.sum()
        self.scale_s = scale_s

    def residual(self, prediction, measurement):
        # Select BEFORE subtraction: excluded or held-out NaN never reaches autograd.
        values = (
            prediction[0, 0][self.mask] - measurement[0, 0][self.mask]
        ) / self.scale_s
        if not torch.isfinite(values).all():
            raise FloatingPointError("nonfinite prediction on a training observation")
        return self.weights.sqrt() * values / self.weight_sum.sqrt()

    def __call__(self, prediction, measurement):
        return 0.5 * self.residual(prediction, measurement).square().sum()


class WeightedPosteriorSampling(ConditioningMethod):
    """Original DPS residual-norm guidance with USCT masks, weights and units."""

    def __init__(self, operator, loss, scale):
        super().__init__(operator, scale=scale)
        self.loss = loss

    def apply(self, x_t, x_0_hat, measurement, x_prev, **kwargs):
        residual = torch.linalg.vector_norm(
            self.loss.residual(self.operator.forward(x_0_hat), measurement)
        )
        (gradient,) = torch.autograd.grad(residual, x_prev)
        return x_t - self.scale * gradient, residual.detach()


class _PaddedDIP(nn.Module):
    """Pad the internal CNN canvas for pooling/BatchNorm, crop to physical grid."""

    def __init__(self, p):
        super().__init__()
        self.network = SkipUNet(p.input_channels, 1, base=p.base_channels)

    def forward(self, z):
        h, w = z.shape[-2:]
        ph, pw = max(16, (h + 7) // 8 * 8), max(16, (w + 7) // 8 * 8)
        return self.network(F.pad(z, (0, pw - w, 0, ph - h)))[..., :h, :w]


def reconstruct(case, config):
    p = config.parameters
    c0, bounds = reference_sound_speed(case, config), speed_bounds(config)
    if not bounds[0] <= c0 <= bounds[1]:
        raise ValueError("reference sound speed must lie within sound-speed bounds")
    device = torch.device(config.deployment.neural_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested neural CUDA device is unavailable; no fallback")
    differential = case.measurement.delta_tof_s is not None
    reference = float(case.metadata.get("reference_sound_speed_mps", 1500.0))
    if differential and not np.isclose(reference, c0, rtol=1e-12, atol=0):
        raise ValueError(
            "delta_tof_s reference conflicts with reference_sound_speed_mps"
        )
    observed = case.measurement.delta_tof_s if differential else case.measurement.tof_s
    if observed is None:
        raise ValueError(
            "FTTT requires tof_s or delta_tof_s with declared first-arrival semantics"
        )
    observed = np.asarray(observed, dtype=float)
    distance = np.linalg.norm(
        case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None], axis=-1
    )
    valid = distance > 0
    if case.measurement.valid_mask is not None:
        valid &= case.measurement.valid_mask
    physics = EikonalForward(
        case.grid, case.geometry, background_speed_mps=c0, spatial_order=p.eikonal_order
    )
    weights = configured_ray_weights(case, physics, valid.ravel(), config).reshape(
        observed.shape
    )
    control = InversionControl(
        case,
        config,
        observed,
        weights=weights,
        valid_mask=valid,
        iteration_unit=(
            "reverse sampling update" if p.variant == "dps" else "Adam parameter update"
        ),
    )
    physics.budget_check, physics.work_counts = (
        control.work.check_time,
        control.work.counts,
    )
    physical = TorchEikonalOperator(
        physics, offset=distance / c0 if differential else None, work=control.work
    )
    operator = NormalizedSlownessOperator(
        physical, bounds, c0, case.grid.roi_mask if p.roi_update_only else None
    )
    measurement = torch.tensor(observed[None, None], dtype=torch.float32, device=device)
    loss = WeightedTravelTimeLoss(control, p.residual_scale_s, device)
    previous = None
    checkpoint_metadata = None
    fallback = np.full(case.grid.shape, 1 / c0)

    def observe(iteration, image):
        nonlocal previous
        state = operator.slowness(image).detach().cpu().double().numpy()[0, 0]
        prediction = operator.forward(image)
        objective = float(loss(prediction, measurement))
        update = (
            None
            if previous is None
            else float(np.linalg.norm(state - previous) / np.linalg.norm(previous))
        )
        reason = control.observe(
            iteration,
            state,
            prediction.detach().cpu().double().numpy()[0, 0],
            objective=objective,
            update_relative=update,
        )
        previous = state.copy()
        return reason is not None

    # Keep caller RNG state intact; never seed unrelated CUDA devices.
    cuda_ids = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=cuda_ids):
        torch.random.default_generator.manual_seed(p.seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(p.seed)
        try:
            control.work.check_time()
            if p.variant == "dip":
                solver = DIPSolver(
                    model_factory=lambda: _PaddedDIP(p),
                    num_iterations=control.policy.max_iterations,
                    lr=p.learning_rate,
                    input_channels=p.input_channels,
                )
            elif p.variant == "inr":
                solver = INRSolver(
                    num_iterations=control.policy.max_iterations,
                    lr=p.learning_rate,
                    hidden_features=p.hidden_features,
                    hidden_layers=p.hidden_layers,
                )
            else:
                from .prior import load_prior

                checkpoint = p.checkpoint
                bank_record = None
                if p.prior_manifest:
                    from .weight_bank import resolve_manifest

                    checkpoint, bank_record = resolve_manifest(p.prior_manifest)

                model, schedule, checkpoint_metadata = load_prior(
                    checkpoint, case.grid.shape, bounds, device
                )
                if bank_record is not None:
                    checkpoint_metadata["weight_bank"] = bank_record
                if p.sampling_steps > schedule.num_train_timesteps:
                    raise ValueError(
                        "sampling_steps exceeds checkpoint training timesteps"
                    )
                solver = DPSSolver(
                    model,
                    schedule=schedule,
                    num_inference_steps=p.sampling_steps,
                    eta=p.eta,
                    conditioning=WeightedPosteriorSampling(
                        operator, loss, p.guidance_scale
                    ),
                )
            solver.solve(
                measurement,
                operator,
                loss_fn=loss,
                observer=observe,
                before_step=control.work.check_time,
                max_iterations=control.policy.max_iterations,
            )
            if control.monitor.reason is None:
                control.monitor.finish("max_iterations")
        except BudgetExhausted as exc:
            control.monitor.finish(exc.reason)
        except FloatingPointError as exc:
            control.fail(exc)
    state, metrics = control.output(fallback)
    metrics.update(
        forward_model="eikonal",
        backend="numpy_eikonal_pytorch_bridge",
        variant=p.variant,
        seed=p.seed,
        torch_version=str(torch.__version__),
        neural_device=str(device),
        physics_device="cpu",
        eikonal_order=p.eikonal_order,
        observation_domain="travel_time",
        observation_units="s",
        observation_mode="water_relative" if differential else "absolute",
        model_parameterization={
            "kind": "normalized_slowness",
            "interval": [0, 1],
            "sound_speed_bounds_mps": list(bounds),
        },
        residual_scale_s=p.residual_scale_s,
        objective="0.5 * training weighted mean squared residual / residual_scale_s^2",
        dps_guidance=(
            "gradient of training weighted RMS residual / residual_scale_s"
            if p.variant == "dps"
            else None
        ),
        pretrained_prior=checkpoint_metadata,
        upstream_commit="b258fa90d284f0105259aeffbfd43814f4cd3085",
    )
    return ReconstructionResult(
        algorithm="fttt",
        case_id=case.case_id,
        sound_speed_mps=1 / state,
        metrics=metrics,
    )
