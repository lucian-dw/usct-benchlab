"""Compare migrated numerical routines with an extracted original submission."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import eitbench as eit
from eitbench.operators.forward import ForwardModel
from eitbench.algorithms.lbfgsb_tv import Objective


def verify(original, out):
    sys.path.insert(0, str(Path(original).resolve()))
    import eit_core
    import inverse

    case = eit.load_case(Path(original) / "data/example.npz")
    d, mesh = case.data, case.mesh
    original_mesh = eit_core.Mesh(
        mesh.nodes, mesh.triangles, mesh.boundary_edges, mesh.electrode_ids
    )
    report = {}
    for variant in ("cem", "continuum"):
        old_cal = inverse.calibrate(original_mesh, d, variant, maxiter=10)
        for mode in ("difference", "absolute"):
            config = dict(
                parameters=dict(
                    variant=variant, mode=mode, calibration_max_iterations=10
                ),
                run_controls=dict(max_iterations=5),
            )
            migrated = eit.run(case=case, config=config)
            bg, contact, _ = old_cal
            old_f = eit_core.ForwardModel(
                original_mesh,
                d["currents_A"],
                d["measurement_matrix"],
                0.07,
                variant,
                contact,
            )
            new_f = ForwardModel(
                mesh, d["currents_A"], d["measurement_matrix"], 0.07, variant, contact
            )
            old_sigma, old_pred, _, _ = inverse.reconstruct(
                old_f,
                d["voltages_V"],
                d["reference_voltages_V"],
                d["measurement_mask"],
                bg,
                maxiter=5,
                mode=mode,
            )
            np.testing.assert_allclose(
                old_sigma, migrated.conductivity_S_m, rtol=1e-12, atol=1e-14
            )
            np.testing.assert_allclose(
                old_pred, migrated.predicted_voltages_V, rtol=1e-12, atol=1e-14
            )
            m = np.random.default_rng(1).uniform(0.7, 1.3, len(mesh.nodes))
            measured = (
                d["voltages_V"] - d["reference_voltages_V"]
                if mode == "difference"
                else d["voltages_V"]
            )
            baseline = (
                old_f.solve(np.full(old_f.n, bg))[0] if mode == "difference" else None
            )
            a = inverse.Objective(
                old_f, measured, d["measurement_mask"], bg, baseline=baseline
            )(m)
            b = Objective(
                new_f, measured, d["measurement_mask"], bg, baseline=baseline
            )(m)
            np.testing.assert_allclose(a[0], b[0], rtol=1e-13)
            np.testing.assert_allclose(a[1], b[1], rtol=1e-13, atol=1e-13)
            report[variant + "_" + mode] = dict(
                conductivity_relative_difference=float(
                    np.linalg.norm(old_sigma - migrated.conductivity_S_m)
                    / np.linalg.norm(old_sigma)
                ),
                voltage_relative_difference=float(
                    np.linalg.norm(old_pred - migrated.predicted_voltages_V)
                    / np.linalg.norm(old_pred)
                ),
                objective_abs_difference=float(abs(a[0] - b[0])),
                gradient_relative_difference=float(
                    np.linalg.norm(a[1] - b[1]) / np.linalg.norm(a[1])
                ),
                calibration_iterations=10,
                target_iterations=5,
            )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps(dict(passed=True, comparisons=report), indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.original, args.out)
