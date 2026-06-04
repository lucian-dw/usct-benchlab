# Measurement Provenance

Allowed values:

- `oracle_travel_time`
- `speedmap_travel_time_surrogate`
- `self_simulated_kwave_wavefield`
- `openbreastus_precomputed_wavefield`
- `external_measurement`

`oracle_travel_time` and `speedmap_travel_time_surrogate` are debug/sanity tiers. They may use realistic property maps, but the observation is generated from ground truth. They must not be described as formal wavefield inversion results.

`self_simulated_kwave_wavefield` means a property map was converted to raw
time/frequency wavefield data. In v0.1 this provenance is reserved for FWI
mainline evidence or simulation/QC diagnostics. k-Wave-derived ray/rWave
feature experiments are archived as observable-mismatch diagnostics and must
not be ranked with the travel-time surrogate baselines.

`openbreastus_precomputed_wavefield` means OpenBreastUS provides channel data directly. Prefer this tier over regenerating data from OpenBreastUS speed maps when available.

NBPslices2D 2025 is currently property-map-only in this repo: `sos`, `den`,
`att`, `y`, and `label`. Without built-in wavefields, release traditional
comparisons use `speedmap_travel_time_surrogate`. A k-Wave self-simulated
wavefield can be generated for FWI or diagnostic studies, but it is not a
ray/rWave release ranking path. The related 2021 measurement dataset requires
separate inspection before support is claimed.
