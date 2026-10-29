# Agent API

BenchLab is the authoritative parameter/model source. Consumers must not infer
physics from names or arbitrary YAML, and must not construct shell commands from
Agent text.

~~~python
from usctbench import describe, run
from usctbench.core.algorithm_specs import make_agent_config, case_capabilities
from usctbench.data.synthetic import make_sound_speed_case

case = make_sound_speed_case(shape=(16, 16), n_transducers=16)
description = describe("straight_cgls")
config = make_agent_config(
    "straight_cgls",
    parameters={"sound_speed_bounds_mps": [1400.0, 1600.0]},
    run_controls={"max_iterations": 10},
    host_limits={"max_iterations": 8, "max_elapsed_s": 60},
    case=case,
)
facts = case_capabilities("straight_cgls", case, config=config)
result = run("straight_cgls", case, config, "runs/agent_example")
~~~

**describe** is static: model family, variant, observation requirements, allowed
parameter schema/defaults, iteration unit, limitations and runtime requirements.
It does not start MATLAB or probe hardware.

**case_capabilities** adds frequencies, calibration presence and resolved policy.
Presence is not validation of calibration units. Runtime availability is unknown
unless supplied by the host.

**make_agent_config** admits only agent-exposed fields and budget requests.
Advanced/internal settings enter only through trusted host configuration, never
by relaying an unvalidated Agent object. Host limits cap requests, and deployment
selects backend/runtime/scratch ownership.

**run** owns preparation, GT exclusion, numerical execution, saving and optional
evaluation/preview. Bare Algorithm.run is the numerical interface and can raise
original numerical exceptions; the public runner records failure stage/message.

**evaluate(run_directory)** summarizes saved records without rerunning physics.
Its passed field means complete available numerical artifacts and successful
requested reporting, not achievement of a scientific image-quality target.

Result consumers inspect status, failure_reason, metrics.stop_reason, stages,
resolved policy, consumed work, runtime and artifact locations. A budget-exhausted
run may have a useful complete checkpoint; an external timeout may have no image.

~~~bash
usct list-algorithms --json
usct describe-algorithm bent_ray_gn --json
usct describe-algorithm rwave_adapter --variant full_green_nonlinear --json
usct evaluate --run runs/agent_example
~~~

JSON stdout is machine-readable; diagnostics belong on stderr. No changes to
consumer repositories are required by this source refactor, but consumers must
migrate removed aliases and v1 runtime messages as listed in [CHANGELOG](../CHANGELOG.md).
