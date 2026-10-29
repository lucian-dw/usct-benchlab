# Parameter contract

Scientific definitions have one source: `algorithms/parameters.py` and
`algorithms/fwi/parameters.py`. YAML files only override those definitions.
Constraints express numerical admissibility, not calibrated clinical ranges.

## Preparation and execution

`core.config.prepare_config` prepares Python objects and YAML through the same
boundary. Direct algorithm `run()` calls use it too. `config.parameters` then
remains a concrete Pydantic object throughout numerical execution. Invalid,
unknown and unused parameters raise `ValueError` before computation; the runner
records that exception at its preparation stage. Inputs are not modified.

```python
from usctbench.algorithms.parameters import CGLSParameters
from usctbench.core.config import prepare_config
from usctbench.core.schema import AlgorithmConfig

config = prepare_config(
    "straight_cgls",
    AlgorithmConfig(
        parameters=CGLSParameters(
            regularization="laplacian", regularization_lambda=0.02
        ),
        run_controls={"max_iterations": 30},
        evaluation={"receiver_fraction": 0.125, "seed": 0},
    ),
    host_limits={"max_iterations": 20},
)
assert config.run_controls.max_iterations == 20
```

The configuration separates:

- `parameters`: mathematical model, regularization, initialization and solver controls.
- `run_controls`: update/call/time budgets and opt-in stopping criteria.
- `evaluation`: receiver/frequency validation split; never an independent test claim.
- `image_evaluation`: post-hoc image region and metric range, never optimization.
- `deployment`: host-owned runtime root, MATLAB executable, backend, GPU visibility
  and scratch root. These are not Agent scientific choices.

The host supplies independent limits to `prepare_config`. Effective budgets are
the intersection with the resolved request. A null iteration request inherits
the method default; zero means zero updates. Relative-update stopping is opt-in
and uses each algorithm's declared variable, units, scope and normalization.

## Current parameter names

Each scientific field declares description, units, default, legal type/range
and `agent`, `advanced` or `internal` exposure. Static discovery generates the
Agent schema from this model and excludes advanced/internal fields. There are
no weak/moderate/strong presets.

`regularization_lambda` is the amplitude of the penalty:

$$
\tfrac12\|W(Ax-b)\|^2 + \tfrac12\lambda^2\|Lx\|^2.
$$

Ray-Born has one `variant`: `wkb_fixed`, `wkb_nonlinear`, `full_green_fixed` or
`full_green_nonlinear`. Fixed variants have no nonlinear inner-iteration or
step controls. All allow `max_cache_bytes=0`; `green_solver_rtol` is in `(0,1)`.
Calibration arrays belong to the measurement, not the parameter dictionary.

WUST scientific parameters describe initialization, bounds, PML and a Hz
schedule. A null schedule uses each prepared frequency once; repeated entries
repeat updates. The runtime resolves the complete schedule before truncating it
to `max_iterations`. Only iteration/time budgets are supported; call caps and
online numerical convergence rules are rejected. A schedule completion is not
a convergence claim. See [fwi.md](fwi.md).

## Migration from the research interface

The refactor removes implicit aliases. Update existing configurations explicitly:

| Previous setting | Current setting |
| --- | --- |
| `parameters.iterations`, `outer_iterations` | `run_controls.max_iterations` |
| Fixed Born `inner_iterations` | `run_controls.max_iterations` |
| Nonlinear Bent/Born `inner_iterations` | Still a scientific inner-solver parameter |
| `parameters.stopping` | `run_controls` |
| `budget_caps` in a request | Independent host `host_limits` argument |
| `parameters.evaluation` / `image_evaluation` | Top-level `evaluation` / `image_evaluation` |
| `lambda`, `damping` | `regularization_lambda`; convert old damping with its square root |
| `l2`, `roughness` | `identity`, `laplacian` |
| `irls`, `huber_irls` | `huber` |
| `mode` plus `green_backend` | One Ray-Born `variant` |
| Runtime, device or scratch fields in FWI parameters | `deployment` |

Historical configs remain in the sibling research archive. They are not accepted
as current runtime APIs. No reconstruction operator or penalty formula changes
as a consequence of these name migrations.

## Public execution and saved output

```python
from usctbench import run, describe, evaluate

specification = describe("straight_cgls")
result = run("straight_cgls", "case.h5", "configs/algorithms/cgls.yaml", "runs/example")
summary = evaluate("runs/example")
```

The output path is the exact case directory, not a parent to which a case ID is
silently appended. The equivalent CLI is `usct run straight_cgls --case case.h5
--config configs/algorithms/cgls.yaml --out runs/example`.

Saved output is `result.h5`, `run.json`, `index.json`, `metrics.json` and optional
`preview.png`. The index is published only after arrays and the run record exist;
a failed run does not have a successful index. Evaluation/preview failures do not
overwrite numerical arrays. `run.json` distinguishes `completed`,
`budget_exhausted` and `failed`, with stage messages and actual wall time.
Budget exhaustion is not convergence. A killed WUST run may have no usable model.

`usct data prepare --dataset synthetic --shape 32 32 --n-transducers 32 --out
data/synthetic_demo` prepares one matched-model example. The same command accepts
`--dataset openbreastus` or `nbpslice2d` with explicit source data. Pressure imports
use `usct data import-pressure`; `usct inspect`, `demo`, `evaluate` and `bench`
have separate responsibilities. Suite paths are relative to their YAML file.
