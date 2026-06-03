"""Observable-channel contracts for wavefield-derived features."""

from __future__ import annotations

from typing import Any

from usctbench.schema import USCTCase

APPARENT_TOF = "apparent_tof"
EIKONAL_TOF = "eikonal_tof"
COMPLEX_WAVEFIELD = "complex_wavefield"
RAW_WAVEFIELD = "raw_wavefield"

FEATURE_CHANNELS = {APPARENT_TOF, EIKONAL_TOF, COMPLEX_WAVEFIELD}


def feature_channel(case: USCTCase) -> str:
    """Return the declared feature channel, or an empty string if undeclared."""

    return str(case.metadata.get("feature_channel", "") or "")


def require_feature_channel(
    case: USCTCase,
    required: str,
    *,
    algorithm: str,
    allow_legacy_missing: bool = False,
) -> None:
    """Raise when a case does not satisfy an algorithm observable contract."""

    actual = feature_channel(case)
    if allow_legacy_missing and not actual:
        return
    if actual != str(required):
        raise ValueError(
            f"{algorithm} requires feature_channel={required!r}, got {actual or '<missing>'!r}; "
            "extract the correct observable channel before running this algorithm"
        )


def require_raw_wavefield(case: USCTCase, *, algorithm: str) -> None:
    """Require a raw wavefield case rather than a derived ToF feature case."""

    actual = feature_channel(case)
    if actual in {APPARENT_TOF, EIKONAL_TOF}:
        raise ValueError(f"{algorithm} requires raw time/frequency wavefield, got feature_channel={actual!r}")
    if case.measurement.time_data is None and case.measurement.freq_data is None:
        raise ValueError(f"{algorithm} requires measurement.time_data or measurement.freq_data")


def channel_metadata(
    *,
    feature_channel: str,
    picker_method: str,
    observable_definition: str,
    water_reference_used: bool,
    calibration_terms: dict[str, Any] | None = None,
    uncertainty_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return standard metadata fields for channelized feature cases."""

    return {
        "feature_channel": feature_channel,
        "picker_method": picker_method,
        "observable_definition": observable_definition,
        "water_reference_used": bool(water_reference_used),
        "calibration_terms": calibration_terms or {},
        "uncertainty_diagnostics": uncertainty_diagnostics or {},
    }
