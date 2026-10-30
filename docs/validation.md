# Current numerical validation

Executed on 2026-10-30 for R01-R20, branch `codex/round3-r01-r20`.
These are new runs of the refactored code, not the earlier F01-F22 results.
No image-quality sweep, full-data benchmark or clinical validation was performed.

## Complete local gates

macOS arm64, Python 3.12.2, NumPy 2.5.4, SciPy 1.18.1, Pydantic 2.14.2:

| Gate | Actual result |
| --- | --- |
| BenchLab final full pytest | 434 passed, 1 skipped, 12.86 s |
| WUST full Python suite | 18 passed and 12 subtests passed, 1.17 s |
| Editable dev/viz installation and pip check | passed |
| compileall, Ruff, Black, diff whitespace checks | passed |
| Public CLI list/describe and 32x32 synthetic prepare/run/evaluate | passed |

The local skip is the CUDA cache test: CuPy is unavailable on this Mac.
An earlier 432-pass integration run preceded the final GN reference-release
change and two lifecycle cases; it is retained as an intermediate record, not
the final result. The complete suite above includes those changes and the
corrected GPU transfer assertion distinguishing CuPy's small GMRES Hessenberg
transfer from a Green-field transfer. The affected GN/Green subset then passed
on A100 (34 passed, no skips). Focused runs overlap the full suite and are not
added to its count. No production code changed after the final complete suite.

## Source-work and state checks

The 4x5 non-water configured background, 4 unique sources, 3 frequencies,
cache=0 and max_iterations=0 test wraps the actual GMRES/FMM functions:

| Initialization | Actual source solves | Other checked facts |
| --- | ---: | --- |
| Full-Green, absolute regularization | 12 GMRES | one background; selected iteration 0 |
| Full-Green, relative normal-diagonal scaling | 12 GMRES | pressure and diagonal share one frequency loop |
| WKB | 8 FMM | water and model fields; one configured model background |

The taskbook reported historical counts 24/36/12 respectively. Those historical
numbers were not rerun here; the new 12/12/8 counts are independently observed.
With cache=0, subsequent J/J* solves are counted at execution rather than inferred
from construction. The already-paid initial Linearization is committed at the
two-forward-call boundary, without a duplicate initial prediction.

The selected-checkpoint quadratic test restores x=0.2 after the working point
reaches x=0.5: its selected gradient is 0.6 and selected convergence is false.
Dense AB/LB references, independent priors/ROI, both physical trial policies,
actual Armijo inequalities and invalid-trial checkpoint consistency remain in
their [single test ownership locations](physics_validation.md).

## A100 execution

Fresh mirrors, separate from the old remote working directories. Host im2:
Python 3.13.9, NumPy 2.3.5, SciPy 1.17.1, h5py 3.16.0, Pydantic 2.12.3,
CuPy 13.6.0; MATLAB R2021b; NVIDIA A100-SXM4-80GB; CUDA 11.4 toolchain.

| Gate | Actual result |
| --- | --- |
| Green operator file on GPU | 15 passed in 2.92 s |
| 9x10 Full-Green CPU/GPU pressure relative L2 | 3.16486e-14 |
| CPU/GPU Jacobian relative L2 | 2.31770e-12 |
| GPU real-inner-product adjoint error | 3.19805e-16 |
| WUST CPU continuum / LDR9 gradient FD | 1.23141e-11 / 3.81486e-10 |
| WUST GPU continuum / LDR9 gradient FD | 1.11200e-3 / 4.30214e-4 |
| WUST CPU loss before / after one checked update | 3.88345e-17 / 8.95907e-18 |
| WUST GPU loss before / after one checked update | 3.88327e-17 / 8.95982e-18 |
| Public RF preparation vs absolute-time DTFT | 2.51010e-8 relative L2 |
| Rebuilt native batch vs MATLAB k-Wave RF / spectrum | 1.77488e-6 / 8.05494e-7 relative L2 |
| Shared-fixture public CPU run/save/evaluate/read | passed, 11.185 s |
| Same fixture public GPU run/save/evaluate/read | passed, 16.273 s |

The Green diagnostic uses the existing heterogeneous fixture, seed 53,
complex128 and GMRES rtol=1e-11. Its one device field cache occupies 11,520 bytes;
8 source solves are reused by subsequent J/J*. No full Green-field transfer to
host occurs until explicit public green_fields. CuPy GMRES does transfer a
small Hessenberg matrix for CPU least squares; no zero-host-transfer claim is made.

MATLAB CPU, rebuilt MEX/GPU and the rebuilt native binary all completed with
process exit 0. The native fixture has 23x27 pixels, 3 unordered TX and 4 RX;
native/serial times were 3.503/1.344 s, not a throughput improvement claim.
MATLAB ingestion and FIR resampling checks passed (passband 0.999878,
alias rejection -72.39 dB). MEX deprecated-target warnings are toolchain warnings.

The shared public fixture has 23x27 pixels, 2 TX/4 RX and 2 updates from a
3-entry schedule. Both BenchLab backends reused it without simulation; each
launched reconstruction once. BenchLab deliberately supplies reversed frequencies
and positive-DTFT pressure to test WUST-owned sorting/conjugation. No GT is used;
saved missing-GT metrics remain unavailable. Final residual disabled is producer
JSON null on its first read. Enabled final-loss tests report zero adjoint RHS.

## Reproduce

From BenchLab, Python >=3.12:

~~~bash
python -m pip install -e ".[dev,viz]"
python -m pip check
python -m compileall -q src tests scripts
ruff check src tests scripts
black --check src tests scripts
pytest -q
usct --help
usct list-algorithms --json
usct describe-algorithm born
~~~

From WUST, create one fixture and exercise its public reconstruction:

~~~bash
python Runtime/tests/integration_runtime.py --matlab /path/to/matlab \
  --backend cpu --out /absolute/new/coherent-fixture
python Runtime/tests/integration_runtime.py --matlab /path/to/matlab \
  --prepare-only --out /absolute/new/rf-check
~~~

Then from BenchLab, substitute cpu or gpu and a fresh output directory:

~~~bash
python scripts/validate_wust_integration.py --runtime /path/to/WaveformInversionUST \
  --matlab /path/to/matlab --backend cpu \
  --fixture /absolute/new/coherent-fixture --out /absolute/new/benchlab-cpu
~~~

WUST's Runtime/README.md gives existing CPU/MEX/native commands. Local JUnit and
compact device/public gate records are outside source in the sibling evidence
directory. Research data, old checkouts and prior validation artifacts are retained.

## Boundaries

WKB uses a declared approximate derivative; Full-Green tests validate the discrete
model, not continuum accuracy at arbitrary tissue contrast. WUST GPU uses complex
single precision, so its FD error is not CPU-double error. WUST supports schedule
and elapsed-time budgets, not online numerical convergence. These representative
device and public-chain checks are not exhaustive resource-leak or scaling tests.
