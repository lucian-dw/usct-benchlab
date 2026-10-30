import h5py
import numpy as np
import pytest

from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.core.schema import GeometrySpec, GridSpec, MeasurementSpec, USCTCase
from usctbench.data.pressure import (
    dtft_pressure_contract,
    prepare_born_pressure,
    require_total_pressure_contract,
)


@pytest.fixture
def pressure_case():
    frequencies = np.array([100e3, 150e3, 200e3])
    time = 2e-6 + np.arange(16) * 1e-7
    pressure = np.zeros((16, 2, 3))
    pressure[4] = np.arange(6).reshape(2, 3)
    pressure[:, 0, 2] = np.nan
    spectrum = (
        1e-7 * np.exp(2j * np.pi * frequencies[:, None, None] * time[4]) * pressure[4]
    )
    return USCTCase(
        case_id="pressure_contract",
        grid=GridSpec(shape=(2, 3), spacing_m=(0.001, 0.002)),
        geometry=GeometrySpec(
            tx_pos_m=[[-0.01, 0], [0, -0.02]],
            rx_pos_m=[[0.01, 0], [0, 0.02], [0.02, 0.02]],
        ),
        measurement=MeasurementSpec(
            domain="frequency",
            frequencies_hz=frequencies,
            freq_data=spectrum,
            source_spectrum=np.array([1 + 2j, -3 + 4j, 5 - 6j]),
            water_reference=0.5 * spectrum,
            time_data=pressure,
            water_reference_time=0.5 * pressure,
            source_wavelet=np.arange(16, dtype=float),
            time_axis_s=time,
            valid_mask=[[True, True, False], [True, True, True]],
        ),
        metadata={"pressure_contract": dtft_pressure_contract("Pa*s")},
    )


@pytest.mark.parametrize("source_per_tx", [False, True])
def test_born_prepares_all_spectra_without_mutating_shared_case(
    pressure_case, source_per_tx
):
    positive = pressure_case
    if source_per_tx:
        positive.measurement.source_spectrum = positive.measurement.source_spectrum[
            :, None
        ] * np.array([[1, 2j]])
    negative = positive.model_copy(deep=True)
    negative.metadata["pressure_contract"]["fourier_sign"] = -1
    # A stale flat convention is not a second source of physical interpretation.
    negative.metadata["frequency_convention"] = "unknown"
    spectra = ("freq_data", "source_spectrum", "water_reference")
    for name in spectra:
        setattr(negative.measurement, name, getattr(positive.measurement, name).conj())
    before = negative.model_copy(deep=True)
    prepared = prepare_born_pressure(negative)
    canonical = prepare_born_pressure(positive)
    for name in spectra:
        np.testing.assert_array_equal(
            getattr(prepared.measurement, name), getattr(canonical.measurement, name)
        )
        np.testing.assert_array_equal(
            getattr(negative.measurement, name), getattr(before.measurement, name)
        )
        assert not np.shares_memory(
            getattr(prepared.measurement, name), getattr(negative.measurement, name)
        )
    for name in (
        "time_data",
        "water_reference_time",
        "time_axis_s",
        "source_wavelet",
        "valid_mask",
    ):
        assert getattr(prepared.measurement, name) is getattr(
            negative.measurement, name
        )
    assert prepared is not negative
    assert prepared.measurement is not negative.measurement
    assert prepared.metadata["pressure_contract"] == dtft_pressure_contract("Pa*s")
    assert negative.metadata == before.metadata
    again = prepare_born_pressure(prepared)
    np.testing.assert_array_equal(
        again.measurement.freq_data, prepared.measurement.freq_data
    )


def test_born_preparation_preserves_absent_calibration(pressure_case):
    pressure_case.measurement.source_spectrum = None
    pressure_case.measurement.water_reference = None
    pressure_case.metadata["pressure_contract"]["fourier_sign"] = -1
    prepared = prepare_born_pressure(pressure_case)
    assert prepared.measurement.source_spectrum is None
    assert prepared.measurement.water_reference is None


@pytest.mark.parametrize(
    "change,match",
    [
        ({"pressure_type": "scattered_pressure"}, "total-pressure"),
        ({"pressure_type": "pressure_ratio"}, "total-pressure"),
        ({"fourier_sign": 0}, "DTFT sign"),
        ({"fourier_sign": True}, "DTFT sign"),
        ({"real_pressure": "true"}, "real_pressure"),
        ({"fourier_sign": -1, "real_pressure": False}, "declared real pressure"),
        ({"spectrum_normalization": "fft_guess"}, "spectrum_normalization"),
        ({"spectrum_normalization": "discrete_sum"}, "Pa\\*s"),
        ({"data_units": ""}, "units"),
    ],
)
def test_born_rejects_unsupported_pressure_contract(pressure_case, change, match):
    pressure_case.metadata["pressure_contract"].update(change)
    with pytest.raises(ValueError, match=match):
        prepare_born_pressure(pressure_case)


def test_contract_requires_declarations_and_only_real_sign_conversion(pressure_case):
    pressure_case.metadata = {"frequency_convention": "exp(-i omega t)"}
    with pytest.raises(ValueError, match="total-pressure"):
        prepare_born_pressure(pressure_case)
    pressure_case.metadata["pressure_contract"] = {
        **dtft_pressure_contract(),
        "real_pressure": False,
    }
    # Complex time signals are usable only in their declared Fourier convention.
    prepare_born_pressure(pressure_case)
    with pytest.raises(ValueError, match="declared real pressure"):
        require_total_pressure_contract(pressure_case.metadata)
    pressure_case.metadata["pressure_contract"]["fourier_sign"] = -1
    require_total_pressure_contract(pressure_case.metadata)


def test_frequency_units_use_only_pressure_contract_and_roundtrip(
    pressure_case, tmp_path
):
    pressure_case.metadata["frequency_data_units"] = "obsolete"
    pressure_case.metadata["pressure_units"] = "Pa"
    path = write_case_hdf5(pressure_case, tmp_path / "case.h5")
    with h5py.File(path, "r") as handle:
        for name in ("freq_data", "water_reference"):
            assert handle[f"measurement/{name}"].attrs["units"] == "Pa*s"
            assert handle[f"measurement/{name}"].attrs["axes"] == "frequency,tx,rx"
        for name in ("time_data", "water_reference_time"):
            assert handle[f"measurement/{name}"].attrs["units"] == "Pa"
    restored = read_case_hdf5(path)
    assert restored.metadata["pressure_contract"] == dtft_pressure_contract("Pa*s")
    np.testing.assert_array_equal(
        restored.measurement.freq_data, pressure_case.measurement.freq_data
    )
    assert restored.measurement.freq_data[0, 0, 0] == 0
    np.testing.assert_array_equal(
        restored.measurement.valid_mask, pressure_case.measurement.valid_mask
    )
