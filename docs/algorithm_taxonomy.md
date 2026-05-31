# Algorithm Taxonomy

v0.1 is traditional-first. The library currently separates algorithms by physical model and runtime dependency:

| Family | Registered names | Status | Primary input |
|---|---|---|---|
| Straight-ray sound speed | `straight_sart`, `straight_sirt`, `straight_cgls` | Runnable | `delta_tof_s` features |
| Straight-ray attenuation | `attenuation_sirt` | Runnable | `log_amp` features |
| Refraction-corrected travel time | `bent_ray_gn` | Runnable native v0.1 backend; optional MATLAB path | `delta_tof_s` features |
| Ray-Born / weak scattering | `rwave_adapter` | Runnable native v0.1 backend; optional MATLAB path | `delta_tof_s` features |
| Tiny waveform inversion | `fwi_tiny` | Synthetic proof of life | synthetic sound-speed ground truth |
| k-Wave FWI adapter | `fwi_kwave_adapter` | A100 external-result ingestion and full-pipeline smoke path | external k-Wave/MATLAB result MAT |

## Excluded From v0.1 Acceptance

Diffusion models, GANs, score-based posterior samplers, large neural operators, and heavy supervised neural training are out of scope for v0.1. They may be documented as future work, but they are not acceptance criteria.

## Algorithm Contract

Every runnable method must follow:

```text
USCTCase -> Algorithm.run(case, config) -> ReconstructionResult
```

Every benchmark run writes:

- `result.h5`
- `metrics.json`
- `metadata.yaml`
- `preview.png` for successful image-producing runs
- `failure_report.md` for failed or skipped runs

## Acceptance Notes

The ray-family quality benchmarks compare the standard wrapper image metrics
and forward-model residuals across `straight_*`, `bent_ray_gn`, and
`rwave_adapter`. The k-Wave FWI adapter is different: it also writes wrapper
metrics for plotting, but its pass/fail evidence is the external k-Wave ground
truth and native scalar metrics recorded as `kwave_gt_*` and `kwave_native_*`.
