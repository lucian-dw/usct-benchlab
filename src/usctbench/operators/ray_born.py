"""Finite-frequency single-scattering operator with refracted Green functions.

For squared slowness m, K dm = omega^2 integral G(r,x) G(x,s) q_s dm(x) dx.
The Fourier convention is exp(-i omega t); the outgoing homogeneous 2-D Green
function is i/4 H0^(1)(k r), satisfying (Laplacian + k^2) G = -delta.
Heterogeneous phase uses fast marching; positive WKB intensity solves the
conservative transport equation div(A^2 grad(T)) = 0 on upwind cell faces.

This is a fixed-background ray-Born linearization, not a full-wave solver or a
complete port of the upstream r-Wave Hessian-free, paraxial shooting algorithm.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext

import numpy as np
from scipy.special import hankel1

from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.operators.eikonal import EikonalForward, PreparedMarching


def homogeneous_green(
    distance: np.ndarray, frequency_hz: float, speed_mps: float, radius: float
):
    """Outgoing 2-D Green function with an explicit cell-scale source cutoff."""
    r = np.maximum(np.asarray(distance, dtype=float), radius)
    return 0.25j * hankel1(0, (2 * np.pi * frequency_hz / speed_mps) * r)


def transport_log_amplitude(tape, slowness: np.ndarray, spacing, source):
    """Conservative first-order transport (no caustic phase corrections).

    Do not differentiate a numerical first-arrival front twice to build its
    curvature: front/stencil changes become spurious attenuation and focusing.
    Use shared face fluxes, positive intensity and independent water calibration.
    """
    shape = slowness.shape
    coordinates = np.indices(shape).reshape(2, -1).T
    distance = np.linalg.norm((coordinates - source) * np.asarray(spacing), axis=1)
    # Frequency-independent amplitude; the common 1/sqrt(8*pi*omega) cancels.
    log_amplitude = -0.5 * np.log(
        np.maximum(distance, min(spacing) / 2) * slowness.ravel()
    )
    from usctbench.operators._marching import conservative_transport

    return conservative_transport(
        tape.order,
        tape.parents,
        tape.times,
        shape,
        np.asarray(spacing),
        log_amplitude,
    )


class RayBornAcquisition:
    """Fixed geometry and calibrated source spectrum shared across backgrounds."""

    def __init__(self, grid, geometry, frequencies_hz, source_spectrum=None):
        self.grid, self.geometry = grid, geometry
        self.frequencies_hz = np.array(frequencies_hz, dtype=float, copy=True)
        if (
            self.frequencies_hz.ndim != 1
            or not self.frequencies_hz.size
            or not np.all(np.isfinite(self.frequencies_hz))
            or np.any(self.frequencies_hz <= 0)
        ):
            raise ValueError("frequencies_hz must be a finite positive vector")
        self.data_shape = (
            len(self.frequencies_hz),
            len(geometry.tx_pos_m),
            len(geometry.rx_pos_m),
        )
        all_positions = np.vstack((geometry.tx_pos_m, geometry.rx_pos_m))
        if not np.all(np.isfinite(all_positions)):
            raise ValueError("transducer positions must be finite")
        self.positions, inverse = np.unique(all_positions, axis=0, return_inverse=True)
        self.tx_ids, self.rx_ids = np.split(inverse, [len(geometry.tx_pos_m)])
        coordinates = np.indices(grid.shape).reshape(2, -1).T
        self.coordinates = (
            np.array(grid.origin_m) + (coordinates + 0.5) * grid.spacing_m
        )
        self.distance = np.linalg.norm(
            self.positions[:, None] - self.coordinates[None], axis=-1
        )
        self.direct_distance = np.linalg.norm(
            geometry.tx_pos_m[:, None] - geometry.rx_pos_m[None], axis=-1
        )
        self.valid_pair_mask = self.direct_distance > 0
        self.reference_cache = {}
        self._device_arrays = {}
        self.set_source_spectrum(
            np.ones(self.data_shape[:2], dtype=complex)
            if source_spectrum is None
            else source_spectrum
        )
        for value in (
            self.frequencies_hz,
            self.positions,
            self.tx_ids,
            self.rx_ids,
            self.coordinates,
            self.distance,
            self.direct_distance,
            self.valid_pair_mask,
        ):
            value.setflags(write=False)

    def set_source_spectrum(self, source_spectrum):
        source = np.asarray(source_spectrum, dtype=complex)
        if source.shape != self.data_shape[:2] or not np.all(np.isfinite(source)):
            raise ValueError(
                "source_spectrum must have canonical shape (frequency, tx)"
            )
        self.source_spectrum = source.copy()
        self.source_spectrum.setflags(write=False)
        self._device_arrays.clear()

    def device_arrays(self, device):
        import cupy as cp

        if device not in self._device_arrays:
            with cp.cuda.Device(device):
                self._device_arrays[device] = (
                    cp.asarray(self.tx_ids),
                    cp.asarray(self.rx_ids),
                    cp.asarray(self.source_spectrum),
                )
        return self._device_arrays[device]

    def close(self):
        """Release run-owned reference fields and fixed device arrays."""
        self.reference_cache.clear()
        self._device_arrays.clear()


class RayBornForward:
    """Nonlinear Eikonal/WKB pressure and its distorted-wave Born update map.

    The background is rebuilt at every call. The Born map approximates the
    derivative of continuum wave physics; it is NOT the exact derivative of
    this discretized WKB pressure. Its separately implemented adjoint is exact
    for the frozen Born map. Nonlinear solvers must check actual WKB predictions
    in their line search instead of trusting a linearized residual decrease.
    The model variable is squared slowness in s^2/m^2, not speed.
    """

    def __init__(
        self,
        grid,
        geometry,
        frequencies_hz,
        *,
        exterior_speed_mps=1500.0,
        source_spectrum=None,
        max_cache_bytes=128 * 1024**2,
        green_backend="eikonal_wkb",
        green_solver_rtol=1e-7,
        green_solver_maxiter=20,
        budget_check=None,
        green_device=None,
        reference_cache=None,
        acquisition=None,
        work_counts=None,
    ):
        self.grid, self.geometry = grid, geometry
        self.acquisition = (
            acquisition
            if acquisition is not None
            else RayBornAcquisition(grid, geometry, frequencies_hz, source_spectrum)
        )
        self.frequencies_hz = self.acquisition.frequencies_hz
        self.work_counts = work_counts if work_counts is not None else {}
        self.initial_linearization = None
        self.settings = dict(
            exterior_speed_mps=exterior_speed_mps,
            acquisition=self.acquisition,
            work_counts=self.work_counts,
            max_cache_bytes=max_cache_bytes,
            green_backend=green_backend,
            green_solver_rtol=green_solver_rtol,
            green_solver_maxiter=green_solver_maxiter,
            budget_check=budget_check,
            green_device=green_device,
            reference_cache=(
                reference_cache
                if reference_cache is not None
                else self.acquisition.reference_cache
            ),
        )

    def take_initial_linearization(self):
        """Transfer the prepared initial state without retaining its field cache."""
        initial, self.initial_linearization = self.initial_linearization, None
        return initial

    def linearize(self, squared_slowness):
        from usctbench.operators.base import Linearization

        model = np.asarray(squared_slowness, dtype=float)
        if (
            model.shape != self.grid.shape
            or not np.all(np.isfinite(model))
            or np.any(model <= 0)
        ):
            raise ValueError("squared slowness must be positive, finite and match grid")
        operator = RayBornOperator(
            self.grid,
            self.geometry,
            self.frequencies_hz,
            background_sound_speed_mps=1 / np.sqrt(model),
            **self.settings,
        )
        value = operator.background_data()
        jacobian = born_jacobian(operator)
        return Linearization(
            value,
            jacobian,
            derivative_kind=(
                "discrete_volume_integral_born_derivative"
                if operator.green_backend == "volume_integral"
                else "continuum_born_approximation_not_discrete_wkb_derivative"
            ),
        )

    def forward(self, squared_slowness):
        return self.linearize(squared_slowness).value


class RayBornOperator:
    """Matrix-free real squared-slowness to complex-pressure perturbation map.

    Data shape is (frequency, transmitter, receiver). Background and source
    spectrum are fixed during this linearization. No ground truth is inspected.
    Green fields, not the dense data-by-pixel Jacobian, are cached within a byte
    budget. Source-receiver self channels must be excluded by acquisition masks.
    """

    data_dtype = np.dtype(np.complex128)

    def __init__(
        self,
        grid: GridSpec,
        geometry: GeometrySpec,
        frequencies_hz: np.ndarray,
        *,
        background_sound_speed_mps: np.ndarray | float = 1500.0,
        exterior_speed_mps: float = 1500.0,
        source_spectrum: np.ndarray | None = None,
        max_cache_bytes: int = 128 * 1024**2,
        reference_cache=None,
        green_backend="eikonal_wkb",
        green_solver_rtol=1e-7,
        green_solver_maxiter=20,
        budget_check=None,
        green_device=None,
        acquisition=None,
        work_counts=None,
    ):
        if green_device is not None and (
            isinstance(green_device, bool)
            or not isinstance(green_device, (int, np.integer))
            or green_device < 0
            or green_backend != "volume_integral"
        ):
            raise ValueError(
                "green_device requires a nonnegative GPU index and volume_integral"
            )
        self.green_device = green_device
        if green_backend not in {"eikonal_wkb", "volume_integral"}:
            raise ValueError("green_backend must be eikonal_wkb or volume_integral")
        if not np.isfinite(green_solver_rtol) or not 0 < green_solver_rtol < 1:
            raise ValueError("green_solver_rtol must be between zero and one")
        if (
            isinstance(green_solver_maxiter, (bool, np.bool_))
            or not isinstance(green_solver_maxiter, (int, np.integer))
            or green_solver_maxiter <= 0
        ):
            raise ValueError("green_solver_maxiter must be a positive integer")
        self.green_backend = green_backend
        self.green_solver_rtol, self.green_solver_maxiter = (
            green_solver_rtol,
            green_solver_maxiter,
        )
        self.work_counts = work_counts if work_counts is not None else {}
        for name in (
            "background_builds",
            "eikonal_solves",
            "green_solves",
            "green_matvecs",
        ):
            self.work_counts.setdefault(name, 0)
        self.budget_check = budget_check
        self.acquisition = (
            acquisition
            if acquisition is not None
            else RayBornAcquisition(grid, geometry, frequencies_hz, source_spectrum)
        )
        self.grid, self.geometry = self.acquisition.grid, self.acquisition.geometry
        self.frequencies_hz = self.acquisition.frequencies_hz
        if not np.isfinite(exterior_speed_mps) or exterior_speed_mps <= 0:
            raise ValueError("exterior_speed_mps must be finite and positive")
        background = np.asarray(background_sound_speed_mps, dtype=float)
        if background.ndim == 0:
            background = np.full(grid.shape, background)
        if (
            background.shape != grid.shape
            or not np.all(np.isfinite(background))
            or np.any(background <= 0)
        ):
            raise ValueError(
                "background sound speed must be finite, positive and match the grid"
            )
        self.background = background.copy()
        self.background_squared_slowness = 1 / self.background**2
        self.exterior_speed_mps = float(exterior_speed_mps)
        self.reference_cache = (
            reference_cache
            if reference_cache is not None
            else self.acquisition.reference_cache
        )
        self.data_shape = self.acquisition.data_shape
        self.n_rays = int(np.prod(self.data_shape))
        self.n_pixels = int(np.prod(grid.shape))
        self.area = float(np.prod(grid.spacing_m))
        self.source_radius_m = float(np.sqrt(self.area / np.pi))
        self.max_cache_bytes = int(max_cache_bytes)
        if self.max_cache_bytes < 0:
            raise ValueError("max_cache_bytes must be nonnegative")
        self.positions = self.acquisition.positions
        self.tx_ids, self.rx_ids = self.acquisition.tx_ids, self.acquisition.rx_ids
        self.distance = self.acquisition.distance
        self.direct_distance = self.acquisition.direct_distance
        self.valid_pair_mask = self.acquisition.valid_pair_mask
        self.xp = np
        if self.green_device is not None:
            import cupy as cp

            self.xp = cp
        if self.green_backend == "eikonal_wkb":
            self.delay = np.zeros_like(self.distance)
            self.log_amplitude_ratio = np.zeros_like(self.distance)
            self.direct_delay = np.zeros(self.data_shape[1:])
            self.direct_log_amplitude_ratio = np.zeros(self.data_shape[1:])
        self.green_method = "analytic_hankel"
        self._cache = OrderedDict()
        self.work_counts["background_builds"] += 1
        if not np.all(self.background == self.exterior_speed_mps):
            if self.green_backend == "eikonal_wkb":
                self._heterogeneous_fields()
            else:
                self.green_method = "full_green_volume_integral"

    @property
    def source_spectrum(self):
        return self.acquisition.source_spectrum

    @source_spectrum.setter
    def source_spectrum(self, value):
        self.acquisition.set_source_spectrum(value)

    def _device_context(self):
        return (
            nullcontext()
            if self.green_device is None
            else self.xp.cuda.Device(self.green_device)
        )

    def _sampling_arrays(self):
        if self.green_device is None:
            return self.tx_ids, self.rx_ids, self.source_spectrum
        return self.acquisition.device_arrays(self.green_device)

    def _host(self, value):
        return value if self.green_device is None else self.xp.asnumpy(value)

    def close(self):
        """Release this background's fields without invalidating shared acquisition."""
        self._cache.clear()

    def _heterogeneous_fields(self):
        geometry_key = (
            "eikonal_geometry",
            id(self.acquisition),
            self.exterior_speed_mps,
        )
        if geometry_key not in self.reference_cache:
            geometry = GeometrySpec(
                tx_pos_m=self.positions, rx_pos_m=self.geometry.rx_pos_m
            )
            self.reference_cache[geometry_key] = EikonalForward(
                self.grid, geometry, background_speed_mps=self.exterior_speed_mps
            )
        model = self.reference_cache[geometry_key]
        slowness = model.extend(
            1 / self.background, background=1 / self.exterior_speed_mps
        )
        prepared = PreparedMarching(
            slowness,
            self.grid.spacing_m,
            budget_check=self.budget_check,
            work_counts=self.work_counts,
        )
        water_marching = None
        for index, source in enumerate(model.source_coordinates):
            if self.budget_check is not None:
                self.budget_check()
            key = (
                geometry_key,
                model.shape,
                tuple(self.grid.spacing_m),
                tuple(source),
                self.exterior_speed_mps,
            )
            if key not in self.reference_cache:
                if water_marching is None:
                    water = np.full(model.shape, 1 / self.exterior_speed_mps)
                    water_marching = PreparedMarching(
                        water,
                        self.grid.spacing_m,
                        budget_check=self.budget_check,
                        work_counts=self.work_counts,
                    )
                reference = water_marching.solve(source)
                reference_log = transport_log_amplitude(
                    reference, water_marching.model, self.grid.spacing_m, source
                )
                self.reference_cache[key] = (reference.times, reference_log)
            reference_times, reference_log = self.reference_cache[key]
            tape = prepared.solve(source)
            delay = tape.times - reference_times
            log_ratio = (
                transport_log_amplitude(tape, slowness, self.grid.spacing_m, source)
                - reference_log
            )
            if not np.all(np.isfinite(log_ratio)) or np.max(np.abs(log_ratio)) > 10:
                raise FloatingPointError(
                    "WKB transport is unstable: refine the grid or use a full-wave backend"
                )
            self.delay[index] = delay.reshape(model.shape)[model.crop].ravel()
            self.log_amplitude_ratio[index] = log_ratio.reshape(model.shape)[
                model.crop
            ].ravel()
            for tx in np.flatnonzero(self.tx_ids == index):
                self.direct_delay[tx] = model.sample_receivers(delay)
                self.direct_log_amplitude_ratio[tx] = model.sample_receivers(log_ratio)
        self.green_method = "eikonal_wkb_transport"

    def green_fields(self, frequency_index: int):
        """Return host fields; internal consumers keep the cache on its device."""
        with self._device_context():
            return self._host(self._green_fields(frequency_index))

    def _green_fields(self, frequency_index):
        if frequency_index in self._cache:
            self._cache.move_to_end(frequency_index)
            return self._cache[frequency_index]
        frequency = self.frequencies_hz[frequency_index]
        green = homogeneous_green(
            self.distance, frequency, self.exterior_speed_mps, self.source_radius_m
        )
        if self.green_backend == "eikonal_wkb":
            green *= np.exp(
                self.log_amplitude_ratio + 2j * np.pi * frequency * self.delay
            )
        if self.green_method == "full_green_volume_integral":
            from usctbench.operators.volume_integral import VolumeIntegralGreen

            solver = VolumeIntegralGreen(
                self.grid,
                frequency,
                self.background,
                self.exterior_speed_mps,
                rtol=self.green_solver_rtol,
                maxiter=self.green_solver_maxiter,
                budget_check=self.budget_check,
                device=self.green_device,
                work_counts=self.work_counts,
            )
            green = solver.fields(green, return_device=True)
        else:
            green = self.xp.asarray(green)
        if green.nbytes <= self.max_cache_bytes:
            while (
                self._cache
                and sum(g.nbytes for g in self._cache.values()) + green.nbytes
                > self.max_cache_bytes
            ):
                self._cache.popitem(last=False)
            self._cache[frequency_index] = green
        return green

    def background_data(self) -> np.ndarray:
        return self.background_data_and_diagonal()[0]

    def background_data_and_diagonal(self, weights=None):
        """Prepare pressure and optional Re(J^H W J) diagonal in one field pass.

        Each frequency's fields serve both outputs even with max_cache_bytes=0.
        Precision must already exclude held-out channels. Both outputs are NumPy.
        """
        weights = None if weights is None else self._precision(weights)
        xp = self.xp
        with self._device_context():
            tx_ids, _, source = self._sampling_arrays()
            result = xp.empty(self.data_shape, dtype=complex)
            diagonal = None if weights is None else xp.zeros(self.n_pixels)
            precision = None if weights is None else xp.asarray(weights)
            for index, frequency in enumerate(self.frequencies_hz):
                if self.budget_check is not None:
                    self.budget_check()
                direct = homogeneous_green(
                    self.direct_distance,
                    frequency,
                    self.exterior_speed_mps,
                    self.source_radius_m,
                )
                if self.green_backend == "eikonal_wkb":
                    direct *= np.exp(
                        self.direct_log_amplitude_ratio
                        + 2j * np.pi * frequency * self.direct_delay
                    )
                result[index] = xp.asarray(direct) * source[index, :, None]
                fields = (
                    self._green_fields(index)
                    if diagonal is not None
                    or self.green_method == "full_green_volume_integral"
                    else None
                )
                if self.green_method == "full_green_volume_integral":
                    receivers = xp.asarray(
                        homogeneous_green(
                            self.distance[self.rx_ids],
                            frequency,
                            self.exterior_speed_mps,
                            self.source_radius_m,
                        )
                    )
                    potential = xp.asarray(
                        (2 * np.pi * frequency) ** 2
                        * self.area
                        * (
                            self.background_squared_slowness
                            - 1 / self.exterior_speed_mps**2
                        )
                    )
                    result[index] += (
                        (fields[tx_ids] * potential.ravel()) @ receivers.T
                    ) * source[index, :, None]
                if diagonal is not None:
                    diagonal += self._frequency_diagonal(
                        index, fields, precision[index]
                    )
                del fields
            return self._host(result), (
                None
                if diagonal is None
                else self._host(diagonal).reshape(self.grid.shape)
            )

    def forward(self, squared_slowness_perturbation: np.ndarray) -> np.ndarray:
        dm = np.asarray(squared_slowness_perturbation)
        if (
            dm.shape != self.grid.shape
            or np.iscomplexobj(dm)
            or not np.all(np.isfinite(dm))
        ):
            raise ValueError("squared-slowness perturbation must be a finite real grid")
        with self._device_context():
            tx_ids, rx_ids, source = self._sampling_arrays()
            perturbation = self.xp.asarray(dm.ravel())
            result = self.xp.empty(self.data_shape, dtype=complex)
            for index, frequency in enumerate(self.frequencies_hz):
                if self.budget_check is not None:
                    self.budget_check()
                green = self._green_fields(index)
                field = (green[tx_ids] * perturbation) @ green[rx_ids].T
                result[index] = (
                    field
                    * (2 * np.pi * frequency) ** 2
                    * self.area
                    * source[index, :, None]
                )
                del green
            return self._host(result)

    def predict(self, sound_speed_mps: np.ndarray) -> np.ndarray:
        speed = np.asarray(sound_speed_mps, dtype=float)
        if (
            speed.shape != self.grid.shape
            or not np.all(np.isfinite(speed))
            or np.any(speed <= 0)
        ):
            raise ValueError("sound speed must be finite, positive and match the grid")
        return self.background_data() + self.forward(
            1 / speed**2 - self.background_squared_slowness
        )

    def adjoint(self, data: np.ndarray) -> np.ndarray:
        values = np.asarray(data, dtype=complex)
        if values.shape != self.data_shape or not np.all(np.isfinite(values)):
            raise ValueError(
                "adjoint data must be finite with shape (frequency, tx, rx)"
            )
        with self._device_context():
            tx_ids, rx_ids, source = self._sampling_arrays()
            values = self.xp.asarray(values)
            result = self.xp.zeros(self.n_pixels)
            for index, frequency in enumerate(self.frequencies_hz):
                if self.budget_check is not None:
                    self.budget_check()
                green = self._green_fields(index)
                weighted = values[index] * source[index, :, None].conj()
                receiver_field = weighted @ green[rx_ids].conj()
                result += (
                    self.xp.sum(green[tx_ids].conj() * receiver_field, axis=0).real
                    * (2 * np.pi * frequency) ** 2
                    * self.area
                )
                del green
            return self._host(result).reshape(self.grid.shape)

    def _precision(self, precision):
        weights = np.broadcast_to(np.asarray(precision, dtype=float), self.data_shape)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("precision must be finite and nonnegative")
        return weights

    def _frequency_diagonal(self, index, green, weights):
        tx_ids, rx_ids, source = self._sampling_arrays()
        tx = self.xp.abs(green[tx_ids]) ** 2
        rx = self.xp.abs(green[rx_ids]) ** 2
        weighted = weights * self.xp.abs(source[index, :, None]) ** 2
        return (
            self.xp.sum(tx * (weighted @ rx), axis=0)
            * (2 * np.pi * self.frequencies_hz[index]) ** 4
            * self.area**2
        )

    def normal_diagonal(self, precision):
        """Exact diagonal of Re(J^H W J), without a dense Jacobian.

        Precision must already exclude validation channels. This makes penalty
        scaling covariant with the pressure/source units, not arbitrary units.
        """
        weights = self._precision(precision)
        with self._device_context():
            weights = self.xp.asarray(weights)
            result = self.xp.zeros(self.n_pixels)
            for index in range(len(self.frequencies_hz)):
                if self.budget_check is not None:
                    self.budget_check()
                green = self._green_fields(index)
                result += self._frequency_diagonal(index, green, weights[index])
                del green
            return self._host(result).reshape(self.grid.shape)


def born_jacobian(operator):
    """Prediction and Jacobian share the background's sole CPU/device field cache."""
    return operator
