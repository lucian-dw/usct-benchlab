"""Pressure units and Fourier contract, shared by importers and consumers."""


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


def require_total_pressure_contract(metadata):
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
            "Explicit total-pressure contract required; ToF/scattered/ratio data are not FWI pressure"
        )
    if type(declared["fourier_sign"]) is not int or declared["fourier_sign"] not in (
        -1,
        1,
    ):
        raise ValueError(
            "Unknown DTFT sign; time-harmonic convention alone is insufficient"
        )
    if declared["fourier_sign"] == 1 and declared["real_pressure"] is not True:
        raise ValueError("Positive-sign conversion requires declared real pressure")
    return declared
