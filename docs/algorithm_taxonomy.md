# Algorithm Taxonomy

v0.1 is traditional-first. The library currently separates algorithms by physical model and runtime dependency:

| Family | Registered names | Status | Primary input |
|---|---|---|---|
| Straight-ray sound speed | `straight_sart`, `straight_sirt`, `straight_cgls` | Runnable | `delta_tof_s` features |
| Straight-ray attenuation | `attenuation_sirt` | Runnable | `log_amp` features |
| Refraction-corrected travel time | `bent_ray_gn` | Optional adapter, skips by default | external MATLAB package |
| Ray-Born / weak scattering | `rwave_adapter` | Optional adapter, skips by default | external MATLAB package |
| Tiny waveform inversion | `fwi_tiny` | Synthetic proof of life | synthetic sound-speed ground truth |

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

