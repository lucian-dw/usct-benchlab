# FWI k-Wave Adapter

The FWI adapter has two modes:

- `ingest_existing_result`: load an external k-Wave/WaveformInversionUST result and convert it to `ReconstructionResult`.
- `data_sanity_only`: validate that the adapter sees the same unified raw `time_data`/`freq_data` and water/reference data as the ray feature extractors.

The second mode is used by `kwave_unified_smoke` only to verify data routing. It is not a claimed FWI inversion result. A formal FWI benchmark must run the external FWI pipeline on the same wavefield case or generated dataset, then ingest the result with `result_path`.

Required metadata:

- `uses_kwave_wavefield: true`
- `feature_source: raw_time_and_frequency_wavefield` or the extracted feature source when run through a feature case
- `inverse_crime_risk` inherited from the case provenance
