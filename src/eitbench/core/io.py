from pathlib import Path
import h5py
import numpy as np

from ..operators.fem import Mesh
from ..operators.forward import ForwardModel

PUBLIC_FIELDS = (
    "nodes",
    "triangles",
    "boundary_edges",
    "electrode_ids",
    "currents_A",
    "measurement_matrix",
    "voltages_V",
    "reference_voltages_V",
    "measurement_mask",
    "thickness_m",
    "contact_impedance_ohm_m2",
    "background_conductivity_S_m",
    "case_id",
)


def validate_data(mesh, data):
    for name in PUBLIC_FIELDS:
        if name not in data:
            raise ValueError("input missing " + name)
        if name != "case_id" and np.iscomplexobj(data[name]):
            raise ValueError("real EIT input required: " + name)
    for name in ("nodes", "triangles", "boundary_edges", "electrode_ids"):
        if not np.array_equal(data[name], getattr(mesh, name)):
            raise ValueError("case data and mesh disagree: " + name)
    ForwardModel(
        mesh,
        data["currents_A"],
        data["measurement_matrix"],
        float(data["thickness_m"]),
        contact=data["contact_impedance_ohm_m2"],
    )
    shape = (len(data["measurement_matrix"]), data["currents_A"].shape[1])
    if not all(shape):
        raise ValueError("empty protocol")
    for name in ("voltages_V", "reference_voltages_V"):
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(name + " must be finite [Q,P]")
    mask = data["measurement_mask"]
    if mask.dtype.kind != "b" or mask.shape != shape or not mask.any():
        raise ValueError("measurement_mask must be nonempty boolean [Q,P]")
    bg = data["background_conductivity_S_m"]
    if bg.shape != () or not np.isfinite(bg) or bg <= 0:
        raise ValueError("positive finite scalar background required")
    if np.asarray(data["case_id"]).shape != ():
        raise ValueError("case_id must be scalar")


def load_case(path):
    from .schema import EITCase

    path = Path(path).resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as source:
            data = {
                name: np.array(source[name]) for name in PUBLIC_FIELDS if name in source
            }
    else:
        with h5py.File(path, "r") as source:
            data = {name: source[name][()] for name in PUBLIC_FIELDS if name in source}
        if isinstance(data.get("case_id"), bytes):
            data["case_id"] = data["case_id"].decode()
        data = {key: np.asarray(value) for key, value in data.items()}
    missing = set(PUBLIC_FIELDS) - set(data)
    if missing:
        raise ValueError(f"input missing {sorted(missing)}")
    for name in ("triangles", "boundary_edges", "electrode_ids"):
        if data[name].dtype.kind not in "iu":
            raise ValueError(name + " must contain integer indices")
    mesh = Mesh(
        *(
            data[name]
            for name in ("nodes", "triangles", "boundary_edges", "electrode_ids")
        )
    )
    return EITCase(mesh, data, path)


def save_case(case, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        np.savez_compressed(path, **{name: case.data[name] for name in PUBLIC_FIELDS})
    else:
        with h5py.File(path, "w") as handle:
            handle.attrs["schema"] = "eitbench.case.v1"
            for name in PUBLIC_FIELDS:
                value = (
                    str(case.data[name].item())
                    if name == "case_id"
                    else case.data[name]
                )
                handle.create_dataset(name, data=value)
    return path
