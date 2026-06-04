# Algorithms

This table summarizes the algorithms registered by v0.1.

| Algorithm | Registered name | Mathematical model | Input | Output | Strength | Limitation | Config |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Straight-ray CGLS | `straight_cgls` | Weighted regularized least squares for `A delta_s = b` | Travel-time delays `delta_tof_s` | Sound speed | Fast Krylov baseline with good regression value | Straight-ray approximation ignores refraction and wave effects | `configs/algorithms/cgls.yaml` |
| Straight-ray SIRT | `straight_sirt` | Simultaneous algebraic reconstruction for the same ray system | Travel-time delays `delta_tof_s` | Sound speed | Stable iterative baseline | Can converge slowly and needs smoothing/relaxation tuning | `configs/algorithms/sirt.yaml` |
| Straight-ray SART | `straight_sart` | Ordered algebraic reconstruction over transmitter groups | Travel-time delays `delta_tof_s` | Sound speed | Often sharper than SIRT for the same iteration count | More sensitive to ordering and relaxation | `configs/algorithms/sart.yaml` |
| Attenuation SIRT | `attenuation_sirt` | Straight-ray log-amplitude tomography | Log-amplitude ratio `log_amp` | Attenuation | Simple attenuation baseline | Depends on amplitude calibration and line-integral assumptions | `configs/algorithms/attenuation.yaml` |
| Bent-ray-style baseline | `bent_ray_gn` | Regularized travel-time baseline inspired by refraction-corrected tomography | Travel-time delays `delta_tof_s` | Sound speed | Provides a refraction-themed comparison using native package I/O | v0.1 backend is not a full external eikonal or ray-tracing solver | `configs/algorithms/bent_ray.yaml` |
| rWave/ray-Born-inspired adapter | `rwave_adapter` | Adapter-style baseline inspired by ray-Born/rWave modeling | Travel-time delays `delta_tof_s` | Sound speed | Keeps an rWave-compatible command slot and reporting contract | Does not claim full complex ray-Born reproduction in v0.1 | `configs/algorithms/rwave.yaml` |
| FWI adapter | `fwi_kwave_adapter` | External full-wave or frequency-domain pressure inversion result ingestion | `USCTCase` plus external FWI result or configured command | Sound speed | Lets benchmark reports include high-fidelity FWI outputs | Requires an external k-Wave/FWI environment or artifact | `configs/algorithms/fwi_kwave.yaml` |
| Tiny FWI sanity | `fwi_tiny` | Small synthetic waveform mismatch model | Tiny synthetic sound-speed case | Sound speed | Local proof-of-life for waveform inversion plumbing | Not a production FWI solver | `configs/algorithms/fwi_tiny.yaml` |

## Notes on Adapter Names

`bent_ray_gn` is a bent-ray-style regularized travel-time baseline in v0.1. It
records `full_external_eikonal_solver = False` and
`v0_1_backend = "regularized_travel_time_baseline"`.

`rwave_adapter` is an rWave/ray-Born-inspired adapter baseline in v0.1. It
records `full_ray_born_solver = False` and
`v0_1_backend = "adapter_style_travel_time_baseline"`.

For the mathematical background, see
[docs/math_formulation.md](math_formulation.md).
