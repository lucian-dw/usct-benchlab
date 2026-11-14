import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import h5py
import numpy as np

from ..evaluation.metrics import measurement_metrics
from ..viz import rasterize, save_preview


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_result(out, case, config, result, *, preview=False):
    metrics = {}
    files = {}
    if result.conductivity_S_m is not None:
        sigma = result.conductivity_S_m
        if (
            sigma.shape != (len(case.mesh.nodes),)
            or not np.isfinite(sigma).all()
            or np.any(sigma <= 0)
        ):
            raise FloatingPointError("invalid conductivity reconstruction")
        image, mask = rasterize(case.mesh, sigma)
        temporary = out / "result.h5.tmp"
        with h5py.File(temporary, "w") as h:
            h.attrs["schema"] = "eitbench.result.v1"
            h.attrs["primary_quantity"] = "conductivity_S_m at mesh nodes"
            h.attrs["case_id"] = case.case_id
            h.attrs["status"] = result.status
            h.attrs["image_axes"] = "y decreases, x increases; display only"
            arrays = dict(
                conductivity_S_m=sigma,
                potential_nodes_V=result.potential_nodes_V,
                predicted_voltages_V=result.predicted_voltages_V,
                baseline_voltages_V=result.baseline_voltages_V,
                observed_voltages_V=case.data["voltages_V"],
                reference_voltages_V=case.data["reference_voltages_V"],
                measurement_mask=case.data["measurement_mask"],
                nodes=case.mesh.nodes,
                triangles=case.mesh.triangles,
                conductivity_image_S_m=image,
                domain_mask=mask,
            )
            for key, value in arrays.items():
                h.create_dataset(key, data=value)
        temporary.replace(out / "result.h5")
        metrics = measurement_metrics(
            sigma,
            result.predicted_voltages_V,
            result.baseline_voltages_V,
            case.data["voltages_V"],
            case.data["reference_voltages_V"],
            case.data["measurement_mask"],
        )
        files["result"] = "result.h5"
        if preview:
            try:
                save_preview(
                    out / "preview.png",
                    image,
                    mask,
                    result.calibration["background_conductivity_S_m"],
                )
                files["preview"] = "preview.png"
            except (ImportError, OSError, ValueError) as exc:
                # An optional display step must not destroy numerical evidence.
                result.metadata["preview_error"] = f"{type(exc).__name__}: {exc}"
    metrics.update(optimizer_success=result.optimization.get("success", False))
    write_json(out / "metrics.json", metrics)
    record = dict(
        schema="eitbench.run.v1",
        algorithm="lbfgsb_tv",
        case_id=case.case_id,
        status=result.status,
        stopping_reason=result.stopping_reason,
        configuration=config.model_dump(),
        calibration=result.calibration,
        optimization=result.optimization,
        work=result.work,
        metadata=result.metadata,
        budget_scope="calibration + baseline + target reconstruction; batched RHS per call",
        checkpoint_policy="only complete optimizer-accepted iterates; calibration-only is not a reconstruction",
        input={
            "path": str(case.source) if case.source else None,
            "sha256": (
                hashlib.sha256(case.source.read_bytes()).hexdigest()
                if case.source
                else None
            ),
        },
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "h5py", "pydantic")
        },
        python=platform.python_version(),
        code_sha256={
            str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in Path(__file__).resolve().parents[1].rglob("*.py")
        },
        artifacts=files,
    )
    write_json(out / "run.json", record)
    if files:
        write_json(
            out / "index.json",
            dict(
                case_id=case.case_id,
                status=result.status,
                artifacts=files,
                run="run.json",
                metrics="metrics.json",
            ),
        )
    return record
