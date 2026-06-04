# FWI k-Wave Adapter

The FWI adapter is the v0.1 high-fidelity k-Wave mainline. It has two release
roles:

- `ingest_existing_result`: load an external k-Wave/WaveformInversionUST result
  and convert it to `ReconstructionResult`.
- `invert_existing_dataset`: launch or ingest the aligned A100 pure-FWI path
  for precomputed k-Wave channel data.

Older `data_sanity_only` and k-Wave unified feature checks are diagnostic-only
and have moved to archived suites. They are useful for data plumbing but are
not claimed FWI inversion results.

Required metadata:

- `uses_kwave_wavefield: true`
- `feature_source: raw_time_and_frequency_wavefield`
- `inverse_crime_risk` inherited from the case provenance
