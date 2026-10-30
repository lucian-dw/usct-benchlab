"""Explicit acquisition facts; unknown provenance is never inferred from names."""

from enum import StrEnum


class MeasurementProvenance(StrEnum):
    UNKNOWN = "unknown"
    ORACLE_TRAVEL_TIME = "oracle_travel_time"
    SPEEDMAP_TRAVEL_TIME_SURROGATE = "speedmap_travel_time_surrogate"
    SELF_SIMULATED_KWAVE_WAVEFIELD = "self_simulated_kwave_wavefield"
    OPENBREASTUS_PRECOMPUTED_WAVEFIELD = "openbreastus_precomputed_wavefield"
    EXTERNAL_MEASUREMENT = "external_measurement"


def case_measurement_metadata(metadata):
    """Copy declared facts to a run; do not derive risk ratings or uses_* flags."""
    return {
        name: metadata.get(name, "unknown")
        for name in (
            "measurement_provenance",
            "measurement_domain",
            "acquisition_type",
            "forward_model",
            "reference_acquisition",
            "preprocessing",
            "feature_source",
            "feature_channel",
            "picker_method",
            "observable_definition",
            "tof_observation_contract",
            "water_reference_used",
            "roi_source",
            "roi_use",
        )
    }


def stamp_measurement_metadata(
    metadata,
    *,
    measurement_provenance,
    benchmark_type,
    forward_model,
    feature_source="unknown",
    extra=None,
):
    """Attach supplied preparation facts without inferring acquisition quality."""
    stamped = dict(metadata or {})
    stamped.update(
        measurement_provenance=MeasurementProvenance(measurement_provenance).value,
        benchmark_type=benchmark_type,
        forward_model=forward_model,
        feature_source=feature_source,
    )
    if extra:
        stamped.update(extra)
    return stamped
