import numpy as np
import pytest
from scipy.special import hankel1

from usctbench.core.schema import GeometrySpec, GridSpec
from tests.helpers import adjoint_error
from usctbench.operators.ray_born import RayBornOperator
from usctbench.operators.ray_born import RayBornForward
from usctbench.operators.volume_integral import VolumeIntegralGreen


def test_volume_solver_checks_deadline_inside_source_solve():
    from usctbench.core.stopping import BudgetExhausted

    op = setup_operator(True)
    calls = []

    def deadline():
        calls.append(1)
        if len(calls) == 3:
            raise BudgetExhausted("time_budget")

    solver = VolumeIntegralGreen(
        op.grid,
        180e3,
        op.background,
        1500,
        rtol=1e-12,
        maxiter=20,
        budget_check=deadline,
    )
    with pytest.raises(BudgetExhausted, match="time_budget"):
        solver.fields(op.green_fields(0))
    assert len(calls) == 3
    assert solver.work_counts == {"green_solves": 1, "green_matvecs": 1}


def test_conservative_transport_converges_for_radial_energy_flux():
    from usctbench.operators._marching import conservative_transport

    errors = []
    for n in (32, 64):
        h = 1 / n
        y, x = np.mgrid[:n, :n] * h
        distance = np.hypot(y + 0.2, x - 0.5)
        initial = -0.5 * np.log(distance.ravel())
        parents = np.zeros((n * n, 2), dtype=int)
        parents[:n] = -1
        actual = conservative_transport(
            np.argsort(distance.ravel()),
            parents,
            distance.ravel(),
            (n, n),
            np.array([h, h]),
            initial,
        )
        intensity = np.exp(2 * actual).reshape(n, n)
        errors.append(np.mean(np.abs(intensity[3:] * distance[3:] - 1)))
    assert errors[1] < 0.6 * errors[0]
    assert errors[1] < 0.03


def test_volume_convolution_matches_anisotropic_dense_nonperiodic_reference():
    from scipy.integrate import quad

    grid = GridSpec(shape=(4, 7), spacing_m=(1e-4, 8e-4), origin_m=(0, 0))
    solver = VolumeIntegralGreen(
        grid, 180e3, np.full(grid.shape, 1500.0), 1500, rtol=1e-10, maxiter=20
    )
    points = np.indices(grid.shape).reshape(2, -1).T * grid.spacing_m
    distance = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    k = 2 * np.pi * 180e3 / 1500
    dense = np.zeros(distance.shape, complex)
    nonzero = distance > 0
    dense[nonzero] = 0.25j * hankel1(0, k * distance[nonzero])
    # Independent quadrature for the singular equal-area disk self cell.
    radius = np.sqrt(np.prod(grid.spacing_m) / np.pi)

    def integral(r):
        return 0.25j * hankel1(0, k * r) * 2 * r / radius**2

    self_term = (
        quad(lambda r: integral(r).real, 0, radius)[0]
        + 1j * quad(lambda r: integral(r).imag, 0, radius)[0]
    )
    np.fill_diagonal(dense, self_term)
    rng = np.random.default_rng(5)
    values = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    np.testing.assert_allclose(
        solver.convolve(values).ravel(), dense @ values.ravel(), atol=2e-14, rtol=2e-14
    )


def test_volume_born_derivative_matches_nonlinear_finite_difference():
    base = setup_operator(True)
    forward = RayBornForward(
        base.grid,
        base.geometry,
        base.frequencies_hz,
        source_spectrum=base.source_spectrum,
        green_backend="volume_integral",
        green_solver_rtol=1e-12,
        max_cache_bytes=10**6,
    )
    model = 1 / base.background**2
    lin = forward.linearize(model)
    rng = np.random.default_rng(53)
    dm = rng.normal(size=model.shape) * 1e-9
    epsilon = 0.01
    numerical = (
        forward.forward(model + epsilon * dm) - forward.forward(model - epsilon * dm)
    ) / (2 * epsilon)
    np.testing.assert_allclose(
        lin.jacobian.forward(dm), numerical, rtol=2e-5, atol=1e-10
    )
    complex_data = rng.normal(size=lin.value.shape) + 1j * rng.normal(
        size=lin.value.shape
    )
    assert adjoint_error(lin.jacobian, dm, complex_data) < 1e-12
    assert lin.derivative_kind == "discrete_volume_integral_born_derivative"
    assert forward.work_counts["green_solves"] > 0
    errors = []
    for epsilon in (0.1, 0.05, 0.025):
        numerical = (
            forward.forward(model + epsilon * dm)
            - forward.forward(model - epsilon * dm)
        ) / (2 * epsilon)
        errors.append(
            np.linalg.norm(numerical - lin.jacobian.forward(dm))
            / np.linalg.norm(lin.jacobian.forward(dm))
        )
    assert errors[-1] < 1e-6
    assert errors[-1] < errors[0]


def test_born_remainder_is_second_order_against_dense_multiple_scattering():
    # Independent dense Lippmann-Schwinger reference on the same cell grid.
    # This checks the Born linearization, not continuum accuracy in tissue.
    op = setup_operator()
    omega = 2 * np.pi * op.frequencies_hz[0]
    points = np.indices(op.grid.shape).reshape(2, -1).T
    points = np.array(op.grid.origin_m) + (points + 0.5) * op.grid.spacing_m
    radius = np.maximum(
        np.linalg.norm(points[:, None] - points[None], axis=-1), op.source_radius_m
    )
    gpp = 0.25j * hankel1(0, omega / 1500 * radius)
    gs = (
        0.25j
        * hankel1(
            0, omega / 1500 * np.linalg.norm(points - op.geometry.tx_pos_m[0], axis=-1)
        )
        * op.source_spectrum[0, 0]
    )
    gr = 0.25j * hankel1(
        0,
        omega
        / 1500
        * np.linalg.norm(op.geometry.rx_pos_m[:, None] - points[None], axis=-1),
    )
    errors = []
    for scale in (1, 0.5, 0.25, 0.125):
        delta = np.full(op.grid.shape, 2e-10 * scale)
        potential = omega**2 * op.area * delta.ravel()
        total = np.linalg.solve(np.eye(op.n_pixels) - gpp * potential, gs)
        errors.append(
            np.linalg.norm(gr @ (potential * total) - op.forward(delta)[0, 0])
        )
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all((orders > 1.9) & (orders < 2.1))


def setup_operator(heterogeneous=False, cache=0):
    grid = GridSpec(shape=(9, 10), spacing_m=(0.001, 0.001), origin_m=(-0.0045, -0.005))
    geom = GeometrySpec(
        tx_pos_m=[[-0.009, -0.008], [0.009, -0.003]],
        rx_pos_m=[[0.005, 0.009], [-0.006, 0.007]],
    )
    background = np.full(grid.shape, 1500.0)
    if heterogeneous:
        yy, xx = np.indices(grid.shape)
        background += 35 * np.exp(-((yy - 4) ** 2 + (xx - 5) ** 2) / 8)
    return RayBornOperator(
        grid,
        geom,
        np.array([180e3, 240e3]),
        background_sound_speed_mps=background,
        source_spectrum=np.array([[1 + 0.2j, 0.7j], [0.8, -0.4j]]),
        max_cache_bytes=cache,
    )


def test_complex_real_adjoint_and_no_dense_jacobian():
    rng = np.random.default_rng(21)
    for heterogeneous in (False, True):
        op = setup_operator(heterogeneous)
        image = rng.normal(size=op.grid.shape) * 1e-9
        data = rng.normal(size=op.data_shape) + 1j * rng.normal(size=op.data_shape)
        assert adjoint_error(op, image, data) < 2e-12
        assert not op._cache
        assert op.green_method == (
            "eikonal_wkb_transport" if heterogeneous else "analytic_hankel"
        )


def test_normal_diagonal_matches_explicit_basis_columns():
    op = setup_operator()
    weights = np.arange(np.prod(op.data_shape)).reshape(op.data_shape) / 10
    diagonal = op.normal_diagonal(weights)
    for index in ((0, 0), (2, 7), (8, 9)):
        basis = np.zeros(op.grid.shape)
        basis[index] = 1
        expected = np.sum(weights * np.abs(op.forward(basis)) ** 2)
        np.testing.assert_allclose(diagonal[index], expected, rtol=1e-12)


def test_homogeneous_born_matches_independent_point_scatterer():
    op = setup_operator()
    point = (2, 7)
    dm = np.zeros(op.grid.shape)
    dm[point] = 1e-9
    coordinate = (
        np.array(op.grid.origin_m) + (np.array(point) + 0.5) * op.grid.spacing_m
    )
    result = op.forward(dm)
    for index, frequency in enumerate(op.frequencies_hz):
        k = 2 * np.pi * frequency / 1500
        gs = 0.25j * hankel1(
            0, k * np.linalg.norm(op.geometry.tx_pos_m - coordinate, axis=1)
        )
        gr = 0.25j * hankel1(
            0, k * np.linalg.norm(op.geometry.rx_pos_m - coordinate, axis=1)
        )
        expected = (
            (2 * np.pi * frequency) ** 2
            * op.area
            * dm[point]
            * gs[:, None]
            * gr[None, :]
            * op.source_spectrum[index, :, None]
        )
        np.testing.assert_allclose(result[index], expected, rtol=1e-13)
    # The scatterer is off the transmitter-receiver line, yet remains observable.
    assert np.all(np.abs(result) > 0)


def test_heterogeneous_background_changes_phase_and_spreading():
    uniform = setup_operator()
    refracted = setup_operator(True)
    assert np.max(np.abs(refracted.delay)) > 1e-8
    assert np.max(np.abs(refracted.log_amplitude_ratio)) > 1e-3
    assert not np.allclose(uniform.background_data(), refracted.background_data())
    np.testing.assert_allclose(
        refracted.predict(refracted.background), refracted.background_data()
    )


def test_green_cache_budget_does_not_change_results():
    uncached = setup_operator(cache=0)
    cached = setup_operator(cache=100000)
    image = np.full(uncached.grid.shape, 1e-10)
    np.testing.assert_allclose(cached.forward(image), uncached.forward(image))
    assert sum(g.nbytes for g in cached._cache.values()) <= cached.max_cache_bytes


@pytest.mark.parametrize("cache", [0, 100000])
def test_combined_background_diagonal_and_later_jacobian_count_actual_work(
    monkeypatch, cache
):
    from usctbench.operators import volume_integral
    from usctbench.operators.ray_born import born_jacobian

    base = setup_operator(True)
    work = {"forward_calls": 7}
    settings = dict(
        background_sound_speed_mps=base.background,
        acquisition=base.acquisition,
        green_backend="volume_integral",
        green_solver_rtol=1e-12,
        max_cache_bytes=cache,
    )
    reference = RayBornOperator(
        base.grid, base.geometry, base.frequencies_hz, **settings
    )
    weights = np.arange(base.n_rays).reshape(base.data_shape) / 10
    expected = reference.background_data(), reference.normal_diagonal(weights)
    actual_work = {"green_solves": 0, "green_matvecs": 0}
    real_gmres, real_matvec = volume_integral.gmres, VolumeIntegralGreen._matvec

    def gmres(*args, **kwargs):
        actual_work["green_solves"] += 1
        return real_gmres(*args, **kwargs)

    def matvec(self, values):
        actual_work["green_matvecs"] += 1
        return real_matvec(self, values)

    monkeypatch.setattr(volume_integral, "gmres", gmres)
    monkeypatch.setattr(VolumeIntegralGreen, "_matvec", matvec)
    op = RayBornOperator(
        base.grid, base.geometry, base.frequencies_hz, work_counts=work, **settings
    )
    prediction, diagonal = op.background_data_and_diagonal(weights)
    np.testing.assert_allclose(prediction, expected[0], rtol=1e-13)
    np.testing.assert_allclose(diagonal, expected[1], rtol=1e-13)
    batch = len(op.positions) * len(op.frequencies_hz)
    assert work["green_solves"] == batch
    jacobian = born_jacobian(op)
    delta = np.full(op.grid.shape, 1e-10)
    for multiplier, execute in enumerate(
        (lambda: jacobian.forward(delta), lambda: jacobian.adjoint(prediction)), start=2
    ):
        execute()
        assert work["green_solves"] == (batch * multiplier if cache == 0 else batch)
        assert all(work[name] == count for name, count in actual_work.items())
    assert work["forward_calls"] == 7
    assert work["background_builds"] == 1
    op.close()
    assert not op._cache


def test_acquisition_calibrated_source_and_water_reference_shared_across_backgrounds():
    from usctbench.operators.ray_born import RayBornAcquisition

    base = setup_operator()
    work = {}
    acquisition = RayBornAcquisition(base.grid, base.geometry, base.frequencies_hz)
    op = RayBornOperator(
        base.grid,
        base.geometry,
        base.frequencies_hz,
        acquisition=acquisition,
        background_sound_speed_mps=1505.0,
        work_counts=work,
    )
    op.source_spectrum = base.source_spectrum
    prediction, diagonal = op.background_data_and_diagonal()
    assert diagonal is None
    assert work["eikonal_solves"] == 2 * len(op.positions)
    forward = RayBornForward(
        base.grid,
        base.geometry,
        base.frequencies_hz,
        acquisition=op.acquisition,
        work_counts=work,
        max_cache_bytes=0,
    )
    lin = forward.linearize(op.background_squared_slowness)
    assert lin.jacobian.acquisition is acquisition
    assert lin.jacobian.reference_cache is op.reference_cache
    assert lin.jacobian.distance is op.distance
    assert lin.jacobian.source_spectrum is op.source_spectrum
    assert work["eikonal_solves"] == 3 * len(op.positions)
    np.testing.assert_allclose(lin.value, prediction, rtol=1e-13)
    op.close()
    np.testing.assert_allclose(lin.jacobian.source_spectrum, base.source_spectrum)


def test_volume_failed_solve_is_counted_and_residual_guard_remains(monkeypatch):
    from usctbench.operators import volume_integral

    base = setup_operator()
    work = {}
    solver = VolumeIntegralGreen(
        base.grid,
        base.frequencies_hz[0],
        np.full(base.grid.shape, 1600),
        1500,
        rtol=1e-12,
        maxiter=1,
        work_counts=work,
    )
    monkeypatch.setattr(volume_integral, "gmres", lambda op, rhs, **kw: (rhs, 0))
    with pytest.raises(FloatingPointError, match="volume-integral Green solve failed"):
        solver.fields(base.green_fields(0))
    assert work == {"green_solves": 1, "green_matvecs": 1}


def test_cuda_background_and_jacobian_share_device_cache(monkeypatch):
    from collections import Counter

    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")
    from usctbench.operators.ray_born import born_jacobian

    base = setup_operator(True)
    settings = dict(
        acquisition=base.acquisition,
        background_sound_speed_mps=base.background,
        green_backend="volume_integral",
        green_solver_rtol=1e-11,
        max_cache_bytes=100000,
    )
    cpu = RayBornOperator(base.grid, base.geometry, base.frequencies_hz, **settings)
    gpu = RayBornOperator(
        base.grid, base.geometry, base.frequencies_hz, green_device=0, **settings
    )
    transfers = []
    real_asnumpy = cp.asnumpy

    def asnumpy(values, *args, **kwargs):
        transfers.append(values.shape)
        return real_asnumpy(values, *args, **kwargs)

    monkeypatch.setattr(cp, "asnumpy", asnumpy)
    prediction, diagonal = gpu.background_data_and_diagonal(np.ones(gpu.data_shape))
    expected_prediction, expected_diagonal = cpu.background_data_and_diagonal(
        np.ones(cpu.data_shape)
    )
    np.testing.assert_allclose(prediction, expected_prediction, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(diagonal, expected_diagonal, rtol=1e-9)
    # CuPy GMRES may transfer its small Hessenberg system for CPU least squares.
    # The field cache must stay on-device; only public results need host arrays.
    field_shape = (len(gpu.positions), gpu.n_pixels)
    transfer_counts = Counter(transfers)
    assert field_shape not in transfer_counts, transfer_counts
    assert transfer_counts[gpu.data_shape] == 1, transfer_counts
    assert transfer_counts[(gpu.n_pixels,)] == 1, transfer_counts
    assert all(isinstance(field, cp.ndarray) for field in gpu._cache.values())
    assert sum(field.nbytes for field in gpu._cache.values()) <= gpu.max_cache_bytes
    counts = dict(gpu.work_counts)
    jacobian = born_jacobian(gpu)
    assert jacobian is gpu
    arrays = gpu.acquisition.device_arrays(0)
    delta = np.full(gpu.grid.shape, 1e-10)
    np.testing.assert_allclose(jacobian.forward(delta), cpu.forward(delta), rtol=1e-9)
    np.testing.assert_allclose(
        jacobian.adjoint(prediction), cpu.adjoint(prediction), rtol=1e-9
    )
    assert gpu.work_counts == counts
    assert gpu.acquisition.device_arrays(0) is arrays
    assert adjoint_error(jacobian, delta, prediction) < 1e-12
    assert field_shape not in transfers, Counter(transfers)
    assert isinstance(gpu.green_fields(0), np.ndarray)
    assert transfers[-1] == field_shape  # Explicit public host-field request.
    gpu.close()
    gpu.acquisition.close()
    assert not gpu._cache and not gpu.acquisition._device_arrays
