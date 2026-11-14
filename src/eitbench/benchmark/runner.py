from pathlib import Path
from ..core.config import prepare_config
from ..core.io import load_case
from ..core.registry import get_algorithm
from ..core.schema import EITCase
from .artifacts import save_result, write_json


def run(
    algorithm="lbfgsb_tv", *, case, config=None, out=None, host_caps=None, preview=False
):
    method = get_algorithm(algorithm)
    resolved = prepare_config(config, host_caps=host_caps)
    case = case if isinstance(case, EITCase) else load_case(case)
    target = Path(out).resolve() if out is not None else None
    if target:
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"output must be empty: {target}")
        target.mkdir(parents=True, exist_ok=True)
    result = method.run(case, resolved)
    if target:
        save_result(target, case, resolved, result, preview=preview)
    return result


def suite(cases, *, algorithm="lbfgsb_tv", config=None, out, preview=False):
    target = Path(out).resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"suite output must be empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    records, seen = [], set()
    for i, source in enumerate(cases):
        case = source if isinstance(source, EITCase) else load_case(source)
        if case.case_id in seen:
            raise ValueError("duplicate suite case ID: " + case.case_id)
        seen.add(case.case_id)
        # Sequential, independent global budgets. Calibration is never reused
        # across potentially incompatible geometry/protocol/reference data.
        folder = f"{i:03d}"
        result = run(
            algorithm, case=case, config=config, out=target / folder, preview=preview
        )
        records.append(
            dict(case_id=case.case_id, directory=folder, status=result.status)
        )
    write_json(target / "summary.json", {"cases": records})
    return records
