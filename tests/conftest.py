from __future__ import annotations

from pathlib import Path

import pytest

from usctbench.data.synthetic import make_sound_speed_case


@pytest.fixture
def synthetic_case():
    return make_sound_speed_case(shape=(12, 12), n_transducers=12, inclusion_mps=1450.0)


@pytest.fixture
def written_case(tmp_path: Path, synthetic_case):
    from usctbench.core.io import write_case_hdf5

    path = tmp_path / "case.h5"
    write_case_hdf5(synthetic_case, path)
    return path
