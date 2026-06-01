# Benchmark Tiers

v0.1 keeps traditional algorithms first, but separates measurement provenance.

## Tier 0: travel-time oracle/debug

`measurement_provenance=oracle_travel_time` is a fast solver sanity benchmark. The feature is generated directly from ground truth with a known projector. Use it for adjoint checks, sign checks, and regression tests. Do not report it as wavefield inversion evidence.

## Tier 1: speed-map travel-time surrogate

`measurement_provenance=speedmap_travel_time_surrogate` covers OpenBreastUS speed-map-only mirrors and NBPslices2D property maps converted into ray features. It is useful for comparing solver stability on realistic anatomy, but the measurement is still generated from GT property maps.

## Tier 2: self-simulated k-Wave wavefield

`measurement_provenance=self_simulated_kwave_wavefield` starts from a property-map case, runs a forward wave simulation, QC checks the raw time/frequency data, then extracts shared features. This is the default unified wavefield benchmark path, but it has inverse-crime risk because the forward model uses the same GT property map library.

## Tier 3: OpenBreastUS precomputed wavefield

`measurement_provenance=openbreastus_precomputed_wavefield` is preferred when a dataset provides independent precomputed RF/frequency-domain wavefields and water/reference pairs. It should be the formal OpenBreastUS benchmark tier once inspection confirms the schema.

## Tier 4: external measurement

`measurement_provenance=external_measurement` is reserved for real measured data or independently provided channel data with documented acquisition metadata.

Every run metadata must include `benchmark_type`, `measurement_provenance`, `forward_model`, `uses_gt_generated_measurement`, `uses_kwave_wavefield`, `uses_openbreastus_precomputed_wavefield`, `uses_complex_wavefield`, `feature_source`, and `inverse_crime_risk`.
