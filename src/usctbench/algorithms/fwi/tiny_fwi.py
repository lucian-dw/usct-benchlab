"""Tiny synthetic waveform-inversion proof of life."""

from __future__ import annotations

import numpy as np

from usctbench.algorithms.fwi.losses import gradient_descent, waveform_from_speed
from usctbench.metrics.image import compute_baseline_improvement_metrics, compute_image_metrics
from usctbench.schema import AlgorithmConfig, ReconstructionResult, ResultStatus, USCTCase


class TinyFWIAlgorithm:
    """A low-frequency synthetic gradient/loss-descent smoke test.

    This is not production FWI. It uses a one-dimensional path waveform to
    verify inversion plumbing before adding heavy full-wave solvers.
    """

    name = "fwi_tiny"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        if case.ground_truth.sound_speed_mps is None:
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.SKIPPED,
                failure_reason="fwi_tiny requires sound_speed_mps ground truth for the synthetic proof-of-life",
            )

        truth = np.asarray(case.ground_truth.sound_speed_mps, dtype=float)
        path_truth = _central_path(truth)
        frequencies = np.asarray(config.parameters.get("frequencies_hz", [1.0e5, 1.5e5, 2.0e5]), dtype=float)
        spacing_m = float(config.parameters.get("spacing_m", case.grid.spacing_m[1]))
        initial_speed = float(config.parameters.get("initial_sound_speed_mps", case.metadata.get("reference_sound_speed_mps", 1500.0)))
        steps = int(config.parameters.get("steps", 20))
        learning_rate = float(config.parameters.get("learning_rate", 1.0e6))
        bounds = config.parameters.get("sound_speed_bounds_mps", [1300.0, 1700.0])

        observed = waveform_from_speed(path_truth, frequencies, spacing_m)
        initial = np.full_like(path_truth, initial_speed, dtype=float)
        reconstructed_path, losses = gradient_descent(
            initial,
            observed,
            frequencies,
            spacing_m,
            steps=steps,
            learning_rate=learning_rate,
            bounds_mps=(float(bounds[0]), float(bounds[1])),
        )
        sound_speed = np.tile(reconstructed_path[None, :], (truth.shape[0], 1))
        metrics = {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_decreased": losses[-1] < losses[0],
            "iterations": steps,
        }
        metrics.update(compute_image_metrics(sound_speed, truth, mask=case.grid.roi_mask))
        metrics.update(compute_baseline_improvement_metrics(sound_speed, truth, initial_speed, mask=case.grid.roi_mask))
        return ReconstructionResult(
            algorithm=self.name,
            case_id=case.case_id,
            sound_speed_mps=sound_speed,
            metrics=metrics,
        )


def _central_path(sound_speed_mps: np.ndarray) -> np.ndarray:
    if sound_speed_mps.ndim != 2:
        raise ValueError("fwi_tiny expects a 2-D sound-speed image")
    return np.asarray(sound_speed_mps[sound_speed_mps.shape[0] // 2, :], dtype=float)
