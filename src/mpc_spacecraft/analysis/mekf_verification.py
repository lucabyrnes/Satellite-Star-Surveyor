"""Monte Carlo statistical verification for the spacecraft MEKF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import quaternion as qu
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2

from mpc_spacecraft.dynamics.disturbances import gravity
from mpc_spacecraft.estimation.MEKF import MEKF
from mpc_spacecraft.estimation.sensor_models import (
    GNSSConfig,
    GyroConfig,
    IMUConfig,
    MEKF_ERROR_STATE_SIZE,
    StarTrackerConfig,
    make_mekf_initial_covariance,
)
from mpc_spacecraft.simulation.sensor_state_updater import SensorBiasUpdater
from mpc_spacecraft.simulation.state_measurement_gen import StateMeasurementGenerator
from mpc_spacecraft.utilities.utils import FloatArray, FullSimState, SensorRigidBodyState


@dataclass(frozen=True)
class MEKFVerificationConfig:
    """Configuration for an isolated MEKF Monte Carlo experiment."""

    run_count: int = 100
    duration_s: float = 120.0
    dt_s: float = 0.1
    gnss_period_s: float = 1.0
    star_tracker_period_s: float = 0.1
    base_seed: int = 20261023
    confidence_level: float = 0.95

    def validate(self) -> None:
        if self.run_count <= 0:
            raise ValueError("run_count must be positive.")
        if self.duration_s <= 0.0 or self.dt_s <= 0.0:
            raise ValueError("duration_s and dt_s must be positive.")
        if self.gnss_period_s <= 0.0 or self.star_tracker_period_s <= 0.0:
            raise ValueError("Measurement periods must be positive.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be strictly between zero and one.")


@dataclass(frozen=True)
class InnovationConsistencyResult:
    """Average normalized innovation statistics for one sensor."""

    times: FloatArray
    average_nis: FloatArray
    lower_bound: FloatArray
    upper_bound: FloatArray
    sample_count: NDArray[np.int64]
    measurement_dimension: int


@dataclass(frozen=True)
class MEKFVerificationResult:
    """Aggregated output from an MEKF Monte Carlo experiment."""

    config: MEKFVerificationConfig
    times: FloatArray
    mean_error: FloatArray
    component_rmse: FloatArray
    attitude_rmse_rad: FloatArray
    average_nees: FloatArray
    nees_lower_bound: FloatArray
    nees_upper_bound: FloatArray
    nees_sample_count: NDArray[np.int64]
    star_tracker_nis: InnovationConsistencyResult
    gnss_nis: InnovationConsistencyResult
    run_summaries: pd.DataFrame


@dataclass(frozen=True)
class _RunResult:
    errors: FloatArray
    nees: FloatArray
    star_times: FloatArray
    star_nis: FloatArray
    gnss_times: FloatArray
    gnss_nis: FloatArray


def mekf_error_state(
    truth: SensorRigidBodyState,
    estimate: SensorRigidBodyState,
) -> FloatArray:
    """Return ``truth - estimate`` in the MEKF's 15-state tangent convention."""
    error = np.empty(MEKF_ERROR_STATE_SIZE, dtype=np.float64)
    error[0:3] = truth.position - estimate.position
    error[3:6] = truth.velocity - estimate.velocity

    q_truth = qu.quaternion(*truth.quat).normalized()
    q_estimate = qu.quaternion(*estimate.quat).normalized()
    q_error = q_estimate.conjugate() * q_truth
    if q_error.w < 0.0:
        q_error = -q_error
    error[6:9] = qu.as_rotation_vector(q_error)
    error[9:12] = truth.gyro_bias - estimate.gyro_bias
    error[12:15] = truth.accel_bias - estimate.accel_bias
    return error


def normalized_squared_error(vector: FloatArray, covariance: FloatArray) -> float:
    """Calculate ``vector.T @ covariance^-1 @ vector`` without an inverse."""
    value = np.asarray(vector, dtype=np.float64)
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (value.size, value.size):
        raise ValueError("Covariance shape must match the vector dimension.")
    if not np.all(np.isfinite(value)) or not np.all(np.isfinite(matrix)):
        raise ValueError("Vector and covariance must contain only finite values.")
    matrix = 0.5 * (matrix + matrix.T)
    factor = cho_factor(matrix, lower=True, check_finite=False)
    solved = cho_solve(factor, value, check_finite=False)
    return float(value @ solved)


def average_chi_square_bounds(
    dimension: int,
    sample_count: int | NDArray[np.int64],
    confidence_level: float = 0.95,
) -> tuple[FloatArray, FloatArray]:
    """Return confidence bounds for an average of chi-squared variables."""
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one.")
    count = np.asarray(sample_count, dtype=np.int64)
    if np.any(count <= 0):
        raise ValueError("sample_count must be positive.")
    alpha = 1.0 - confidence_level
    degrees_of_freedom = dimension * count
    lower = chi2.ppf(alpha / 2.0, degrees_of_freedom) / count
    upper = chi2.ppf(1.0 - alpha / 2.0, degrees_of_freedom) / count
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def run_mekf_verification(
    config: MEKFVerificationConfig = MEKFVerificationConfig(),
    *,
    show_progress: bool = False,
) -> MEKFVerificationResult:
    """Run independent sensor-noise and bias-walk realizations of the MEKF."""
    config.validate()
    step_count = int(round(config.duration_s / config.dt_s))
    if not np.isclose(
        step_count * config.dt_s, config.duration_s, rtol=0.0, atol=1e-10
    ):
        raise ValueError("duration_s must be an integer multiple of dt_s.")

    times = np.arange(step_count + 1, dtype=np.float64) * config.dt_s
    errors = np.empty(
        (config.run_count, step_count + 1, MEKF_ERROR_STATE_SIZE), dtype=np.float64
    )
    nees = np.empty((config.run_count, step_count + 1), dtype=np.float64)
    run_results: list[_RunResult] = []

    seed_sequences = np.random.SeedSequence(config.base_seed).spawn(config.run_count)
    iterator = enumerate(seed_sequences)
    if show_progress:
        from tqdm import tqdm

        iterator = enumerate(tqdm(seed_sequences, desc="MEKF Monte Carlo runs"))

    summary_rows: list[dict[str, float | int]] = []
    for run_index, seed_sequence in iterator:
        run = _run_single_realization(config, times, seed_sequence)
        errors[run_index] = run.errors
        nees[run_index] = run.nees
        run_results.append(run)
        attitude_norm = np.linalg.norm(run.errors[:, 6:9], axis=1)
        summary_rows.append(
            {
                "run": run_index,
                "spawn_key": int(seed_sequence.spawn_key[-1]),
                "final_attitude_error_deg": float(np.rad2deg(attitude_norm[-1])),
                "attitude_rms_deg": float(np.rad2deg(np.sqrt(np.mean(attitude_norm**2)))),
                "mean_nees": float(np.mean(run.nees)),
                "star_tracker_mean_nis": float(np.mean(run.star_nis)),
                "gnss_mean_nis": float(np.mean(run.gnss_nis)),
            }
        )

    mean_error = np.mean(errors, axis=0)
    component_rmse = np.sqrt(np.mean(errors**2, axis=0))
    attitude_rmse = np.sqrt(np.mean(np.sum(errors[:, :, 6:9] ** 2, axis=2), axis=0))
    average_nees = np.mean(nees, axis=0)
    nees_count = np.full(step_count + 1, config.run_count, dtype=np.int64)
    nees_lower, nees_upper = average_chi_square_bounds(
        MEKF_ERROR_STATE_SIZE, nees_count, config.confidence_level
    )

    star_result = _aggregate_innovations(
        run_results,
        sensor="star_tracker",
        dimension=3,
        confidence_level=config.confidence_level,
    )
    gnss_result = _aggregate_innovations(
        run_results,
        sensor="gnss",
        dimension=6,
        confidence_level=config.confidence_level,
    )
    return MEKFVerificationResult(
        config=config,
        times=times,
        mean_error=mean_error,
        component_rmse=component_rmse,
        attitude_rmse_rad=attitude_rmse,
        average_nees=average_nees,
        nees_lower_bound=nees_lower,
        nees_upper_bound=nees_upper,
        nees_sample_count=nees_count,
        star_tracker_nis=star_result,
        gnss_nis=gnss_result,
        run_summaries=pd.DataFrame(summary_rows),
    )


def _run_single_realization(
    config: MEKFVerificationConfig,
    times: FloatArray,
    seed_sequence: np.random.SeedSequence,
) -> _RunResult:
    initial_seed, measurement_seed, bias_seed = seed_sequence.spawn(3)
    initial_rng = np.random.default_rng(initial_seed)
    measurement_rng_seed = int(measurement_seed.generate_state(1, dtype=np.uint32)[0])
    bias_rng_seed = int(bias_seed.generate_state(1, dtype=np.uint32)[0])

    imu_config = IMUConfig()
    gyro_config = GyroConfig()
    gnss_config = GNSSConfig()
    star_config = StarTrackerConfig()
    initial_covariance = make_mekf_initial_covariance()
    truth = _make_initial_truth_state()
    initial_error = initial_rng.multivariate_normal(
        np.zeros(MEKF_ERROR_STATE_SIZE), initial_covariance
    )
    initial_estimate = _estimate_from_error(truth.sensor_rigid_body, initial_error)
    mekf = MEKF(
        initial_estimate,
        imu_config,
        gyro_config,
        gnss_config,
        star_config,
        initial_covariance,
    )
    measurement_generator = StateMeasurementGenerator(
        imu_config,
        gyro_config,
        gnss_config,
        star_config,
        measurement_rng_seed,
        enable_noise=True,
    )
    bias_updater = SensorBiasUpdater(
        imu_config, gyro_config, bias_rng_seed, enabled=True
    )

    errors = np.empty((len(times), MEKF_ERROR_STATE_SIZE), dtype=np.float64)
    nees = np.empty(len(times), dtype=np.float64)
    errors[0] = mekf_error_state(truth.sensor_rigid_body, initial_estimate)
    nees[0] = normalized_squared_error(errors[0], mekf.P)
    star_times: list[float] = []
    star_nis: list[float] = []
    gnss_times: list[float] = []
    gnss_nis: list[float] = []
    last_star_time: float | None = None
    last_gnss_time: float | None = None

    estimate = initial_estimate
    for index, current_time in enumerate(times[1:], start=1):
        truth = _propagate_truth(truth, config.dt_s)
        truth = bias_updater.tick(truth, config.dt_s)
        measurement = measurement_generator.generate_state_measurement(truth, config.dt_s)
        use_star = _measurement_due(
            float(current_time), last_star_time, config.star_tracker_period_s
        )
        use_gnss = _measurement_due(
            float(current_time), last_gnss_time, config.gnss_period_s
        )
        estimate = mekf.update(
            config.dt_s,
            measurement.inertial_accel,
            measurement.omega,
            gnss_measure=measurement.translation.data if use_gnss else None,
            st_measure=qu.quaternion(*measurement.quat) if use_star else None,
        )
        if use_star:
            last_star_time = float(current_time)
        if use_gnss:
            last_gnss_time = float(current_time)

        for diagnostic in mekf.last_innovations:
            nis = normalized_squared_error(diagnostic.residual, diagnostic.covariance)
            if diagnostic.sensor == "star_tracker":
                star_times.append(float(current_time))
                star_nis.append(nis)
            else:
                gnss_times.append(float(current_time))
                gnss_nis.append(nis)
        errors[index] = mekf_error_state(truth.sensor_rigid_body, estimate)
        nees[index] = normalized_squared_error(errors[index], mekf.P)

    return _RunResult(
        errors=errors,
        nees=nees,
        star_times=np.asarray(star_times, dtype=np.float64),
        star_nis=np.asarray(star_nis, dtype=np.float64),
        gnss_times=np.asarray(gnss_times, dtype=np.float64),
        gnss_nis=np.asarray(gnss_nis, dtype=np.float64),
    )


def _make_initial_truth_state() -> FullSimState:
    state = FullSimState.zeros()
    state.position[:] = [7_078e3, 0.0, 0.0]
    state.velocity[:] = [0.0, 7.504e3, 0.0]
    state.quat[:] = qu.as_float_array(
        qu.from_rotation_vector(np.array([0.15, -0.10, 0.05]))
    )
    state.omega[:] = [0.004, -0.003, 0.002]
    state.inertial_accel[:] = 0.0
    return state


def _estimate_from_error(
    truth: SensorRigidBodyState, error: FloatArray
) -> SensorRigidBodyState:
    estimate = truth.copy()
    estimate.position[:] = truth.position - error[0:3]
    estimate.velocity[:] = truth.velocity - error[3:6]
    q_truth = qu.quaternion(*truth.quat).normalized()
    q_estimate = q_truth * qu.from_rotation_vector(-error[6:9])
    estimate.quat[:] = qu.as_float_array(q_estimate.normalized())
    estimate.gyro_bias[:] = truth.gyro_bias - error[9:12]
    estimate.accel_bias[:] = truth.accel_bias - error[12:15]
    return estimate


def _propagate_truth(state: FullSimState, dt: float) -> FullSimState:
    next_state = state.copy()
    position, velocity = _rk4_orbit_step(state.position, state.velocity, dt)
    next_state.position[:] = position
    next_state.velocity[:] = velocity
    q_current = qu.quaternion(*state.quat).normalized()
    q_next = q_current * qu.from_rotation_vector(state.omega * dt)
    next_state.quat[:] = qu.as_float_array(q_next.normalized())
    next_state.inertial_accel[:] = 0.0
    return next_state


def _rk4_orbit_step(
    position: FloatArray, velocity: FloatArray, dt: float
) -> tuple[FloatArray, FloatArray]:
    def derivative(state: FloatArray) -> FloatArray:
        return np.concatenate([state[3:6], gravity(state[0:3])])

    state = np.concatenate([position, velocity])
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    result = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return result[0:3], result[3:6]


def _measurement_due(
    current_time: float, last_time: float | None, period: float
) -> bool:
    if last_time is None:
        return True
    tolerance = 1e-12 * max(1.0, abs(current_time), abs(last_time))
    return current_time - last_time >= period - tolerance


def _aggregate_innovations(
    runs: list[_RunResult],
    *,
    sensor: str,
    dimension: int,
    confidence_level: float,
) -> InnovationConsistencyResult:
    time_attribute = "star_times" if sensor == "star_tracker" else "gnss_times"
    value_attribute = "star_nis" if sensor == "star_tracker" else "gnss_nis"
    reference_times = getattr(runs[0], time_attribute)
    values = []
    for run in runs:
        run_times = getattr(run, time_attribute)
        if run_times.shape != reference_times.shape or not np.allclose(
            run_times, reference_times, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"Inconsistent {sensor} update times between runs.")
        values.append(getattr(run, value_attribute))
    stacked = np.asarray(values, dtype=np.float64)
    count = np.full(reference_times.shape, len(runs), dtype=np.int64)
    lower, upper = average_chi_square_bounds(
        dimension, count, confidence_level
    )
    return InnovationConsistencyResult(
        times=reference_times.copy(),
        average_nis=np.mean(stacked, axis=0),
        lower_bound=lower,
        upper_bound=upper,
        sample_count=count,
        measurement_dimension=dimension,
    )
