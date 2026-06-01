# rWave / Ray-Born Unified Input

The v0.1 native `rwave_adapter` is a ray-Born-style surrogate over travel-time features. It should be marked as surrogate when `measurement_provenance` is `oracle_travel_time` or `speedmap_travel_time_surrogate`.

For a true ray-Born benchmark, the adapter should consume complex `freq_data`, water/reference data, and a scattered-field convention from a wavefield case. Until that external path is connected, the unified suite uses the same feature case as the ray solvers and reports provenance explicitly.

Formal reports must distinguish:

- surrogate rWave over ToF features
- true ray-Born over complex frequency-domain wavefields
