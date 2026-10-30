# Final refactor validation

Executed on 2026-10-30 for F01-F22. The changes are on
`codex/final-refactor-f01-f22`, not merged into the original main checkout.

## Environment and complete suites

- Local macOS: Python 3.12, NumPy 2.5.4, SciPy 1.18.1, Pydantic 2.14.2.
- A100 host: Python 3.13.9, NumPy 2.3.5, SciPy 1.17.1, h5py 3.16.0,
  Pydantic 2.12.3, CuPy 13.6.0; MATLAB R2021b; A100-SXM4-80GB, CUDA 11.4.
- A100 complete BenchLab suite: **371 passed in 4.67 s**, no skips.
- A100 complete WUST Python suite: **17 passed, 10 subtests in 1.43 s**, no skips.
- Editable installation with dev/viz, pip check, compileall, Ruff and Black pass.
- CLI help/list JSON and prepare/run/evaluate/read pass. The 32x32 CGLS example
  finishes with max_iterations and a saved model, not an invented convergence flag.

The complete suites ran once after focused unit validation. Tests cover independent
dense constrained/augmented references, Huber refresh, complex data with real
unknowns, anisotropic convolution, phase ambiguity, active Eikonal derivatives,
Born remainder/adjoints, trial policies, accepted-state ownership, data isolation,
budget starts, serialization, and public APIs.

## Hardware and public integration

| Gate | Actual result |
| --- | --- |
| Full-Green CPU/GPU forward relative L2 | 1.08746e-15 |
| Full-Green CPU/GPU Jacobian relative L2 | 1.81315e-13 |
| Full-Green GPU real-inner-product adjoint error | 4.79707e-16 |
| WUST CPU continuum / LDR9 gradient FD | 3.46637e-10 / 1.97022e-10 |
| WUST GPU continuum / LDR9 gradient FD | 1.11200e-3 / 4.30214e-4 |
| RF public prepare vs independent absolute-time DTFT | 2.51010e-8 |
| Native batch vs MATLAB RF / frequency relative L2 | 1.7749e-6 / 8.0549e-7 |
| Same-fixture public CPU reconstruction/save/evaluate/read | passed, 10.734 s |
| Same-fixture public GPU reconstruction/save/evaluate/read | passed, 13.043 s |

The public fixture is 23x27, two TX/four RX, two model updates from a three-entry
plan. Both BenchLab runs reused the one WUST-generated coherent fixture;
each reconstruction launched MATLAB once. Positive-sign pressure deliberately
tests WUST-owned conjugation and sorting. The result has no GT and preserves
missing-GT metric semantics. The native check uses a separate asymmetric
three-TX/four-RX 23x27 fixture and the rebuilt native binary.

Final MATLAB/native checks returned exit code 0. Earlier direct SSH invocations
printed passing numeric checks but returned 255; those were not accepted as
process-success evidence. The final remote-shell checks explicitly captured
MATLAB_SHELL_EXIT=0. MEX rebuild succeeded; nvcc's deprecated target warnings
remain toolchain warnings, not numerical failures.

## Reproduce maintained gates

From BenchLab:

~~~bash
python -m pip install -e ".[dev,viz]"
python -m pip check
python -m compileall -q src tests scripts
ruff check src tests scripts
black --check src tests scripts
pytest -q
usct --help
usct list-algorithms --json
~~~

Create the shared fixture and test WUST from its repository:

~~~bash
python Runtime/tests/integration_runtime.py --matlab /path/to/matlab \
  --backend cpu --out /absolute/new/coherent-fixture
python Runtime/tests/integration_runtime.py --matlab /path/to/matlab \
  --prepare-only --out /absolute/new/rf-check
~~~

Then from BenchLab, for each explicit backend cpu/gpu:

~~~bash
python scripts/validate_wust_integration.py --runtime /path/to/WaveformInversionUST \
  --matlab /path/to/matlab --backend cpu \
  --fixture /absolute/new/coherent-fixture --out /absolute/new/benchlab-cpu
~~~

WUST's Runtime/README.md contains the MEX/native build and numerical test commands.
The Full-Green comparison uses the existing heterogeneous setup_operator fixture
in tests/operators/test_ray_born.py, CPU and green_device=0 RayBornForward
linearizations, seed 53 and the same complex128 model/observations.

## Limits and artifacts

No clinical, large-data, image-quality or throughput campaign was run. WKB's
derivative remains the declared approximation. WUST GPU is complex single
precision and does not promise CPU-double FD errors. WUST schedule completion
is not convergence; optional final residual evaluation is separate work.
No raw data, binary, matrix or result arrays are tracked.

Current docs describe maintained behavior; per-unit development history is in
the sibling research archive. JUnit XML and compact gate reports are retained
outside source under evidence/final_f01_f22. No PR, push or merge was performed.

Final artifact inspection found MATLAB's unrequested final residual encoded as
an empty array. The first Python manifest read now maps that sentinel to null,
without discarding computed residual objects. After the complete suites, the
11-test transport subset (8 subtests) and both public CPU/GPU integration gates
were rerun successfully with an explicit null assertion. No numerical formula
changed in this final serialization correction.
