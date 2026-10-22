"""Timestep-convergence experiments for dynamics and closed-loop simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd
import quaternion as qu

from mpc_spacecraft.analysis.mission_results import (
    build_derived_timeseries,
    compute_mission_summary,
)
from mpc_spacecraft.config import ClosedLoopTestConfig
from mpc_spacecraft.dynamics import DisturbanceModel, SpacecraftDynamics
from mpc_spacecraft.simulation.closed_loop_testing import (
    ClosedLoopTestResult,
    run_closed_loop_test,
)
from mpc_spacecraft.utilities.utils import RotationState


@dataclass(frozen=True)
class RotationTrajectory:
    """A sampled rotational trajectory."""

    time: np.ndarray
    quaternion: np.ndarray
    angular_velocity: np.ndarray


@dataclass
class DynamicsConvergenceResult:
    """Dynamics-only sweep outputs."""

    metrics: pd.DataFrame
    trajectories: dict[float, RotationTrajectory]
    reference_dt: float
    fitted_attitude_order: float
    fitted_angular_velocity_order: float
    principal_axis_max_error_rad: float


@dataclass
class ClosedLoopConvergenceResult:
    """Closed-loop sensitivity sweep outputs."""

    metrics: pd.DataFrame
    runs: dict[float, ClosedLoopTestResult]
    reference_dt: float
    reference_mission_summary: dict[str, object]


def quaternion_angular_distance_rad(
    quaternion_a: np.ndarray, quaternion_b: np.ndarray
) -> np.ndarray:
    """Return the sign-invariant shortest angular distance between quaternions."""
    a = np.asarray(quaternion_a, dtype=np.float64)
    b = np.asarray(quaternion_b, dtype=np.float64)
    if a.shape != b.shape or a.shape[-1] != 4:
        raise ValueError("Quaternion arrays must have equal shape (..., 4).")

    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    sign = np.where(np.sum(a * b, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    chord = np.linalg.norm(a - sign * b, axis=-1)
    return 4.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))


def estimate_observed_order(
    timesteps: Iterable[float], errors: Iterable[float]
) -> float:
    """Fit the exponent in ``error = C * dt**p`` on finite positive samples."""
    dt = np.asarray(list(timesteps), dtype=np.float64)
    error = np.asarray(list(errors), dtype=np.float64)
    mask = np.isfinite(dt) & np.isfinite(error) & (dt > 0.0) & (error > 0.0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.polyfit(np.log(dt[mask]), np.log(error[mask]), 1)[0])


def validate_step_grid(
    timesteps: Iterable[float],
    reference_dt: float,
    duration: float,
    event_periods: Iterable[float] = (),
) -> list[float]:
    """Validate that all integration and event times share the reference grid."""
    steps = [float(value) for value in timesteps]
    values = [reference_dt, duration, *steps, *event_periods]
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(
            "Timesteps, duration, and event periods must be finite and positive."
        )
    if len(set(steps)) != len(steps):
        raise ValueError("Timesteps must be unique.")
    if any(step <= reference_dt for step in steps):
        raise ValueError("Every tested timestep must be larger than reference_dt.")

    for value in [duration, *steps, *event_periods]:
        ratio = value / reference_dt
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-10):
            raise ValueError(
                f"{value:g} is not an integer multiple of reference_dt={reference_dt:g}."
            )
    for step in [reference_dt, *steps]:
        ratio = duration / step
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-10):
            raise ValueError(
                f"duration={duration:g} is not an integer multiple of dt={step:g}."
            )
    for period in event_periods:
        for step in [reference_dt, *steps]:
            ratio = period / step
            if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-10):
                raise ValueError(
                    f"event period {period:g} is not an integer multiple of dt={step:g}."
                )
    return sorted(steps, reverse=True)


def simulate_rotation_dynamics(
    dt: float,
    duration: float,
    inertia: np.ndarray,
    initial_quaternion: np.ndarray,
    initial_angular_velocity: np.ndarray,
    torque: np.ndarray,
) -> RotationTrajectory:
    """Propagate rotational dynamics with a constant body torque."""
    step_count = int(round(duration / dt))
    dynamics = SpacecraftDynamics(
        np.asarray(inertia), DisturbanceModel(enable_gravity=False)
    )
    state = RotationState.zeros()
    state.quat[:] = np.asarray(initial_quaternion, dtype=np.float64)
    state.omega[:] = np.asarray(initial_angular_velocity, dtype=np.float64)

    time = np.arange(step_count + 1, dtype=np.float64) * dt
    quaternion = np.empty((step_count + 1, 4), dtype=np.float64)
    angular_velocity = np.empty((step_count + 1, 3), dtype=np.float64)
    quaternion[0] = state.quat
    angular_velocity[0] = state.omega
    for index in range(step_count):
        state = dynamics.discrete_dynamics_rk4_rotation(state, torque, dt)
        quaternion[index + 1] = state.quat
        angular_velocity[index + 1] = state.omega
    return RotationTrajectory(time, quaternion, angular_velocity)


def run_dynamics_convergence(
    timesteps: Iterable[float] = (0.08, 0.04, 0.02, 0.01),
    reference_dt: float = 0.00125,
    duration: float = 10.0,
) -> DynamicsConvergenceResult:
    """Run the isolated nonlinear rotational-dynamics convergence sweep."""
    steps = validate_step_grid(timesteps, reference_dt, duration)
    inertia = np.array(
        [[1200.0, -45.0, 30.0], [-45.0, 1000.0, -20.0], [30.0, -20.0, 800.0]],
        dtype=np.float64,
    )
    initial_quaternion = qu.as_float_array(
        qu.from_rotation_vector(np.array([0.25, -0.18, 0.12], dtype=np.float64))
    )
    # Deliberately dynamic enough that all four RK4 error levels remain above
    # floating-point noise while staying well resolved by the coarsest step.
    initial_angular_velocity = np.array([1.2, -0.8, 0.5], dtype=np.float64)
    torque = np.array([20.0, -15.0, 8.0], dtype=np.float64)

    trajectories = {
        dt: simulate_rotation_dynamics(
            dt, duration, inertia, initial_quaternion, initial_angular_velocity, torque
        )
        for dt in [*steps, reference_dt]
    }
    reference = trajectories[reference_dt]
    rows: list[dict[str, float]] = []
    for dt in steps:
        trajectory = trajectories[dt]
        reference_indices = _matching_reference_indices(
            trajectory.time, reference.time, reference_dt
        )
        attitude_error = quaternion_angular_distance_rad(
            trajectory.quaternion, reference.quaternion[reference_indices]
        )
        angular_velocity_error = np.linalg.norm(
            trajectory.angular_velocity - reference.angular_velocity[reference_indices],
            axis=1,
        )
        rows.append(
            {
                "sim_dt_s": dt,
                "attitude_error_final_rad": float(attitude_error[-1]),
                "attitude_error_rms_rad": _rms(attitude_error),
                "attitude_error_max_rad": float(np.max(attitude_error)),
                "angular_velocity_error_final_radps": float(angular_velocity_error[-1]),
                "angular_velocity_error_rms_radps": _rms(angular_velocity_error),
                "angular_velocity_error_max_radps": float(
                    np.max(angular_velocity_error)
                ),
            }
        )

    metrics = (
        pd.DataFrame(rows)
        .sort_values("sim_dt_s", ascending=False)
        .reset_index(drop=True)
    )
    metrics["attitude_pairwise_order"] = _pairwise_orders(
        metrics["sim_dt_s"].to_numpy(), metrics["attitude_error_max_rad"].to_numpy()
    )
    metrics["angular_velocity_pairwise_order"] = _pairwise_orders(
        metrics["sim_dt_s"].to_numpy(),
        metrics["angular_velocity_error_max_radps"].to_numpy(),
    )
    attitude_order = estimate_observed_order(
        metrics["sim_dt_s"], metrics["attitude_error_max_rad"]
    )
    omega_order = estimate_observed_order(
        metrics["sim_dt_s"], metrics["angular_velocity_error_max_radps"]
    )

    return DynamicsConvergenceResult(
        metrics=metrics,
        trajectories=trajectories,
        reference_dt=reference_dt,
        fitted_attitude_order=attitude_order,
        fitted_angular_velocity_order=omega_order,
        principal_axis_max_error_rad=_principal_axis_sanity_error(
            reference_dt, duration
        ),
    )


def run_closed_loop_convergence(
    base_config: ClosedLoopTestConfig,
    timesteps: Iterable[float] = (0.05, 0.025, 0.0125),
    reference_dt: float = 0.003125,
    duration: float = 120.0,
) -> ClosedLoopConvergenceResult:
    """Run deterministic full-system timestep sensitivity against a fine reference."""
    periods = (
        base_config.timing.mpc_dt,
        base_config.sensors.gnss_measurement_period,
        base_config.sensors.star_tracker_measurement_period,
    )
    steps = validate_step_grid(timesteps, reference_dt, duration, periods)
    deterministic_environment = replace(
        base_config.environment, disturbance_noise_std=None
    )
    deterministic_sensors = replace(
        base_config.sensors,
        enable_measurement_noise=False,
        enable_bias_random_walk=False,
    )

    runs: dict[float, ClosedLoopTestResult] = {}
    for dt in [*steps, reference_dt]:
        timing = replace(
            base_config.timing,
            sim_dt=dt,
            sim_cycles=int(round(duration / dt)),
        )
        config = replace(
            base_config,
            timing=timing,
            environment=deterministic_environment,
            sensors=deterministic_sensors,
            show_progress=False,
        )
        runs[dt] = run_closed_loop_test(config)

    reference = runs[reference_dt]
    reference_arrays = reference.to_arrays()
    rows: list[dict[str, object]] = []
    for dt in steps:
        run = runs[dt]
        arrays = run.to_arrays()
        reference_indices = _matching_reference_indices(
            arrays["time"], reference_arrays["time"], reference_dt
        )
        true_state = arrays["true_state"]
        reference_state = reference_arrays["true_state"][reference_indices]
        attitude_error = quaternion_angular_distance_rad(
            true_state[:, 6:10], reference_state[:, 6:10]
        )
        omega_error = np.linalg.norm(
            true_state[:, 10:13] - reference_state[:, 10:13], axis=1
        )
        position_error = np.linalg.norm(
            true_state[:, 0:3] - reference_state[:, 0:3], axis=1
        )
        velocity_error = np.linalg.norm(
            true_state[:, 3:6] - reference_state[:, 3:6], axis=1
        )
        torque_error = np.linalg.norm(
            arrays["control"] - reference_arrays["control"][reference_indices], axis=1
        )

        final_state = run.final_state
        reference_final_state = reference.final_state
        if final_state is None or reference_final_state is None:
            raise RuntimeError("Closed-loop simulation did not provide a final state.")
        final_attitude_error = float(
            quaternion_angular_distance_rad(
                final_state.quat, reference_final_state.quat
            )
        )
        final_omega_error = float(
            np.linalg.norm(final_state.omega - reference_final_state.omega)
        )
        final_position_error = float(
            np.linalg.norm(final_state.position - reference_final_state.position)
        )
        final_velocity_error = float(
            np.linalg.norm(final_state.velocity - reference_final_state.velocity)
        )

        # Records are captured before propagation and end at T - dt. Include
        # the separately stored state at T so final, RMS, and maximum errors all
        # describe the same physical interval.
        attitude_error = np.append(attitude_error, final_attitude_error)
        omega_error = np.append(omega_error, final_omega_error)
        position_error = np.append(position_error, final_position_error)
        velocity_error = np.append(velocity_error, final_velocity_error)

        logs = run.to_dataframe()
        derived = build_derived_timeseries(logs)
        mission = compute_mission_summary(
            logs, run.config.targets.targets, run.config, derived
        )
        rows.append(
            {
                "sim_dt_s": dt,
                "attitude_error_final_rad": final_attitude_error,
                "attitude_error_rms_rad": _rms(attitude_error),
                "attitude_error_max_rad": float(np.max(attitude_error)),
                "angular_velocity_error_final_radps": final_omega_error,
                "angular_velocity_error_rms_radps": _rms(omega_error),
                "angular_velocity_error_max_radps": float(np.max(omega_error)),
                "position_error_final_m": final_position_error,
                "position_error_max_m": float(np.max(position_error)),
                "velocity_error_final_mps": final_velocity_error,
                "velocity_error_max_mps": float(np.max(velocity_error)),
                "torque_error_rms_nm": _rms(torque_error),
                "torque_error_max_nm": float(np.max(torque_error)),
                "control_impulse_nms": float(
                    np.sum(np.linalg.norm(arrays["control"], axis=1)) * dt
                ),
                "attitude_estimation_error_mean_deg": mission[
                    "attitude_estimation_error_mean_deg"
                ],
                "position_estimation_error_mean_m": mission["position_error_mean_m"],
                "velocity_estimation_error_mean_mps": mission[
                    "velocity_error_mean_mps"
                ],
                "targets_completed": mission["targets_completed"],
                "observing_time_s": mission["observing_time_s"],
                "mpc_update_count": mission["mpc_update_count"],
                "mpc_solve_success_count": mission["mpc_solve_success_count"],
                "mpc_fallback_count": mission["mpc_fallback_count"],
            }
        )

    metrics = (
        pd.DataFrame(rows)
        .sort_values("sim_dt_s", ascending=False)
        .reset_index(drop=True)
    )
    reference_logs = reference.to_dataframe()
    reference_derived = build_derived_timeseries(reference_logs)
    reference_summary = compute_mission_summary(
        reference_logs,
        reference.config.targets.targets,
        reference.config,
        reference_derived,
    )
    return ClosedLoopConvergenceResult(
        metrics=metrics,
        runs=runs,
        reference_dt=reference_dt,
        reference_mission_summary=reference_summary,
    )


def _matching_reference_indices(
    sample_times: np.ndarray, reference_times: np.ndarray, reference_dt: float
) -> np.ndarray:
    indices = np.rint(np.asarray(sample_times) / reference_dt).astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= len(reference_times)):
        raise ValueError("Sample times fall outside the reference trajectory.")
    if not np.allclose(reference_times[indices], sample_times, rtol=0.0, atol=1e-9):
        raise ValueError("Sample and reference times are not aligned.")
    return indices


def _pairwise_orders(dt: np.ndarray, error: np.ndarray) -> np.ndarray:
    orders = np.full(len(dt), np.nan, dtype=np.float64)
    for index in range(len(dt) - 1):
        if error[index] > 0.0 and error[index + 1] > 0.0:
            orders[index] = np.log(error[index] / error[index + 1]) / np.log(
                dt[index] / dt[index + 1]
            )
    return orders


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _principal_axis_sanity_error(dt: float, duration: float) -> float:
    omega_z = 0.2
    trajectory = simulate_rotation_dynamics(
        dt=dt,
        duration=duration,
        inertia=np.diag([1200.0, 1000.0, 800.0]),
        initial_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        initial_angular_velocity=np.array([0.0, 0.0, omega_z]),
        torque=np.zeros(3),
    )
    half_angle = 0.5 * omega_z * trajectory.time
    exact = np.column_stack(
        [
            np.cos(half_angle),
            np.zeros_like(half_angle),
            np.zeros_like(half_angle),
            np.sin(half_angle),
        ]
    )
    return float(np.max(quaternion_angular_distance_rad(trajectory.quaternion, exact)))
