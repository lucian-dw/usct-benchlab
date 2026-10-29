# Parameter contract

The authoritative definitions are the Pydantic classes in
[algorithms/parameters.py](../src/usctbench/algorithms/parameters.py) and
[algorithms/fwi/parameters.py](../src/usctbench/algorithms/fwi/parameters.py).
Fields carry description, units, default, type/range and exposure. YAML and direct
Python calls resolve through **core.config.prepare_config** to the same objects.

## Request structure

- **parameters**: a concrete per-algorithm Parameters instance after preparation.
- **run_controls**: requested budgets and supported numerical stopping rules.
- **evaluation**: training/validation split and seed.
- **image_evaluation**: foreground/full-image reporting policy.
- **deployment**: explicit backend, runtime root, MATLAB executable, device and
  scratch root. These are host settings, not algorithm hyperparameters.

Unknown names fail. There is one regularization name,
**regularization_lambda**; the quadratic objective coefficient is its square.
Scientific ranges reflect admission constraints, not proven optimal search
intervals. No weak/moderate/strong numerical presets exist.

## Exposure

- **agent**: permitted scientific choices such as bounds, a supported
  initialization mode, or SIRT/SART relaxation. Exact allowed fields come from
  describe(), not a list copied into a consumer.
- **advanced**: expert solver controls, including regularization strength,
  derivative tolerances and inner solve settings where declared.
- **internal**: runtime/numerical details or artifact arrays not admitted to the
  autonomous parameter set. Deployment remains a separate host-owned object.

The default Agent schema omits advanced/internal definitions and their defaults.
A trusted host can supply expert parameters independently. A field's availability
can depend on the selected Born variant.

## Cross-field semantics

Speed bounds must be ordered and include the chosen starting model. Differential
ToF cannot silently change its reference speed. Fixed-Born does not accept
unused nonlinear controls. Cache capacity may be zero; Green solve tolerance is
strictly between zero and one. WUST needs explicit PML and speed bounds for
execution and a host-supplied runtime with an elapsed-time limit.

Run controls resolve request and host caps by intersection. No universal
update_rtol is injected. See [evaluation](evaluation.md).

Names removed during this refactor are listed once in [migration notes](../CHANGELOG.md);
old YAML aliases are not permanent runtime APIs.
