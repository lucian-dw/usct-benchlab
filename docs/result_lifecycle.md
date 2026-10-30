# Numerical results and optional reports

The public run function prepares input, calls the numerical algorithm without GT,
saves arrays, publishes the run record and only then evaluates/plots.

- result.h5: numerical arrays and solver-side metrics at save time.
- run.json: resolved config, status, failure stage/message, work, time and artifacts.
- index.json: numerical-success index, published only after arrays and run record.
- metrics.json: latest numerical and optional evaluation records.
- preview.png: optional visualization, with its own stage status.

A reporting failure retains the numerical image. A failed numerical save is not
published as success. A failed solve may preserve a last complete checkpoint;
that does not turn its failure into convergence. A timed-out external process may
return no image. evaluate reports these distinctions without rerunning physics.

Runtime includes preparation and solve. process_peak_rss_mib is the current
process's cumulative peak RSS, not incremental per-case or GPU memory.

See [evaluation](evaluation.md) for image metrics and missing-GT semantics.

Default history is one compact record per accepted iteration, including the
actual objective, update/optimality measures and inner solve summary. Setting
`diagnostics: true` on the run configuration adds trial/direction checks; it
does not change the numerical decisions. The current accepted Linearization is
reused by the next step and is not stored in historical snapshots.

Zero observation energy or zero baseline RMSE yields a null relative measure
with its availability status; absolute errors remain reported.
