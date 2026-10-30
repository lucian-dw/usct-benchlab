"""Pressure units and Fourier contract, shared by importers and consumers."""

from usctbench.core.schema import USCTCase


def dtft_pressure_contract(data_units="unknown"):
    return {
        "fourier_sign": 1,
        "real_pressure": True,
        "pressure_type": "total_pressure",
        "data_units": data_units,
        "spectrum_normalization": "dtft_dt",
    }


def pressure_units(metadata):
    return metadata.get("pressure_contract", {}).get("data_units", "unknown")


def require_total_pressure_contract(metadata, *, target_fourier_sign=-1):
    """Validate total pressure and whether conversion to the consumer is possible.

    WUST uses -DTFT; Born requests +DTFT. Conjugation switches the Fourier
    kernel only for a declared real time-domain pressure signal.
    """
    declared = dict(metadata.get("pressure_contract", {}))
    required = {
        "fourier_sign",
        "real_pressure",
        "pressure_type",
        "data_units",
        "spectrum_normalization",
    }
    if set(declared) != required or declared["pressure_type"] != "total_pressure":
        raise ValueError(
            "Explicit total-pressure contract required; ToF/scattered/ratio data are not total pressure"
        )
    if type(declared["fourier_sign"]) is not int or declared["fourier_sign"] not in (
        -1,
        1,
    ):
        raise ValueError(
            "Unknown DTFT sign; time-harmonic convention alone is insufficient"
        )
    if type(declared["real_pressure"]) is not bool:
        raise ValueError("real_pressure must explicitly be true or false")
    if (
        not isinstance(declared["data_units"], str)
        or not declared["data_units"].strip()
        or declared["spectrum_normalization"]
        not in ("dtft_dt", "discrete_sum", "unknown")
    ):
        raise ValueError("Pressure units and spectrum_normalization must be declared")
    if (
        declared["data_units"] == "Pa*s"
        and declared["spectrum_normalization"] != "dtft_dt"
    ):
        raise ValueError("Pa*s requires dtft_dt spectrum_normalization")
    if type(target_fourier_sign) is not int or target_fourier_sign not in (-1, 1):
        raise ValueError("target_fourier_sign must be -1 or +1")
    if (
        declared["fourier_sign"] != target_fourier_sign
        and not declared["real_pressure"]
    ):
        raise ValueError("Fourier-sign conversion requires declared real pressure")
    return declared


def prepare_born_pressure(case: USCTCase) -> USCTCase:
    """Return a +DTFT total-pressure view without changing the reusable case.

    Data, source spectrum and independent water spectrum share one Fourier
    contract. Only negative-DTFT spectra are conjugated; RF, masks, geometry,
    units and normalization are retained. Unchanged arrays are shared read-only
    by convention, as with other prepared case views.
    """
    declared = require_total_pressure_contract(case.metadata, target_fourier_sign=1)
    measurement = case.measurement
    if measurement.freq_data is None or measurement.frequencies_hz is None:
        raise ValueError("Born requires freq_data and frequencies_hz")
    spectra = {}
    if declared["fourier_sign"] == -1:
        for name in ("freq_data", "source_spectrum", "water_reference"):
            value = getattr(measurement, name)
            if value is not None:
                spectra[name] = value.conj()
    declared["fourier_sign"] = 1
    return case.model_copy(
        update={
            "measurement": measurement.model_copy(update=spectra),
            "metadata": {**case.metadata, "pressure_contract": declared},
        }
    )
