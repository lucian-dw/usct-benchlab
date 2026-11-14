from pathlib import Path
import h5py
import numpy as np


def measurement_metrics(conductivity, predicted, baseline, observed, reference, mask):
    difference = observed - reference
    residual = (predicted - baseline - difference)[mask]
    dn, an = np.linalg.norm(difference[mask]), np.linalg.norm(observed[mask])
    return dict(
        conductivity_ground_truth_available=False,
        image_metrics_reason="Experimental photos are not quantitative conductivity truth.",
        n_measurements=int(mask.sum()),
        difference_voltage_relative_error=(
            float(np.linalg.norm(residual) / dn) if dn > 0 else None
        ),
        absolute_voltage_relative_error=(
            float(np.linalg.norm((predicted - observed)[mask]) / an) if an > 0 else None
        ),
        difference_voltage_rmse_V=float(np.sqrt(np.mean(residual**2))),
        conductivity_min_S_m=float(conductivity.min()),
        conductivity_max_S_m=float(conductivity.max()),
    )


def evaluate(run_dir):
    path = Path(run_dir)
    with h5py.File(path / "result.h5", "r") as h:
        return measurement_metrics(
            *(
                h[key][()]
                for key in (
                    "conductivity_S_m",
                    "predicted_voltages_V",
                    "baseline_voltages_V",
                    "observed_voltages_V",
                    "reference_voltages_V",
                    "measurement_mask",
                )
            )
        )
