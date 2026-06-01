# Measurement Provenance

Allowed values:

- `oracle_travel_time`
- `speedmap_travel_time_surrogate`
- `self_simulated_kwave_wavefield`
- `openbreastus_precomputed_wavefield`
- `external_measurement`

`oracle_travel_time` and `speedmap_travel_time_surrogate` are debug/sanity tiers. They may use realistic property maps, but the observation is generated from ground truth. They must not be described as formal wavefield inversion results.

`self_simulated_kwave_wavefield` means a property map was converted to raw time/frequency wavefield data, then features were extracted from the same raw case. This is the unified benchmark tier for straight-ray, bent-ray, attenuation, rWave/ray-Born, and FWI adapters.

`openbreastus_precomputed_wavefield` means OpenBreastUS provides channel data directly. Prefer this tier over regenerating data from OpenBreastUS speed maps when available.

NBPslices2D 2025 is currently property-map-only in this repo: `sos`, `den`, `att`, `y`, and `label`. Without built-in wavefields, it should be run as a property-map library through `self_simulated_kwave_wavefield`. The related 2021 measurement dataset requires separate inspection before support is claimed.
