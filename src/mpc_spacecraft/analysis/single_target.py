"""Reusable setup and metrics for single-target closed-loop experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import quaternion as qu

from mpc_spacecraft.analysis.mission_results import quaternion_angle_error_deg
from mpc_spacecraft.config import (
    ClosedLoopEnvironmentConfig,
    ClosedLoopMPCConfig,
    ClosedLoopTestConfig,
    ClosedLoopTimingConfig,
    InitialStateConfig,
    SpacecraftPhysicalConfig,
    TargetConfig,
)
from mpc_spacecraft.simulation import ClosedLoopTestResult, run_closed_loop_test
from mpc_spacecraft.utilities.utils import (
    BODY_FORWARD_VEC3,
    FloatArray,
    quat_rotate_vector_a_to_b,
)

DEFAULT_TARGET_DECLINATION_RAD = 0.174533
DEFAULT_TARGET_RIGHT_ASCENSION_RAD = 0.261799
DEFAULT_OBSERVATION_TIME_S = 30.0


@dataclass
class SingleTargetCaseResult:
    """One named single-target simulation and its derived results."""

    name: str
    label: str
    result: ClosedLoopTestResult
    logs: pd.DataFrame
    metrics: dict[str, Any]


def make_single_target_table(
    observation_time_s: float = DEFAULT_OBSERVATION_TIME_S,
    declination_rad: float = DEFAULT_TARGET_DECLINATION_RAD,
    right_ascension_rad: float = DEFAULT_TARGET_RIGHT_ASCENSION_RAD,
) -> pd.DataFrame:
    """Create the fixed, feasible target used by performance experiments."""
    return pd.DataFrame(
        {
            "req_obs_time": [float(observation_time_s)],
            "Dec": [float(declination_rad)],
            "RA": [float(right_ascension_rad)],
        }
    )


def target_direction_from_table(targets: pd.DataFrame) -> FloatArray:
    """Return the first target's inertial unit direction."""
    target = targets.iloc[0]
    declination = float(target["Dec"])
    right_ascension = float(target["RA"])
    return np.array(
        [
            np.cos(declination) * np.cos(right_ascension),
            np.cos(declination) * np.sin(right_ascension),
            np.sin(declination),
        ],
        dtype=np.float64,
    )


def make_target_relative_initial_state(
    base_state: InitialStateConfig,
    targets: pd.DataFrame,
    attitude_offset_deg: float,
    angular_rate_deg_s: float,
    error_axis_body: FloatArray | None = None,
) -> InitialStateConfig:
    """Create an initial state offset about an axis perpendicular to boresight."""
    axis = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if error_axis_body is None
        else np.asarray(error_axis_body, dtype=np.float64)
    )
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("error_axis_body must be a finite three-vector.")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm == 0.0:
        raise ValueError("error_axis_body must be nonzero.")
    axis = axis / axis_norm

    target_direction = target_direction_from_table(targets)
    goal_quaternion = quat_rotate_vector_a_to_b(BODY_FORWARD_VEC3, target_direction)
    attitude_offset = qu.from_rotation_vector(
        axis * np.deg2rad(float(attitude_offset_deg))
    )
    initial_quaternion = goal_quaternion * attitude_offset
    initial_omega = axis * np.deg2rad(float(angular_rate_deg_s))
    return replace(
        base_state,
        quat=qu.as_float_array(initial_quaternion),
        omega=initial_omega,
    )


def make_single_target_config(
    *,
    attitude_offset_deg: float,
    angular_rate_deg_s: float,
    max_duration_s: float = 300.0,
    sim_dt: float = 0.01,
    mpc_dt: float = 2.0,
    base_config: ClosedLoopTestConfig | None = None,
    spacecraft: SpacecraftPhysicalConfig | None = None,
    mpc: ClosedLoopMPCConfig | None = None,
) -> ClosedLoopTestConfig:
    """Build a complete MEKF-in-the-loop configuration for one fixed target."""
    config = ClosedLoopTestConfig.defaults() if base_config is None else base_config
    if max_duration_s <= 0.0 or sim_dt <= 0.0 or mpc_dt <= 0.0:
        raise ValueError("max_duration_s, sim_dt, and mpc_dt must be positive.")
    cycle_count = max_duration_s / sim_dt
    if not np.isclose(cycle_count, round(cycle_count), rtol=0.0, atol=1e-9):
        raise ValueError("max_duration_s must be an integer multiple of sim_dt.")

    target_table = make_single_target_table()
    initial_state = make_target_relative_initial_state(
        config.initial_state,
        target_table,
        attitude_offset_deg,
        angular_rate_deg_s,
    )
    timing = replace(
        config.timing,
        sim_dt=float(sim_dt),
        mpc_dt=float(mpc_dt),
        sim_cycles=int(round(cycle_count)),
    )
    return replace(
        config,
        timing=timing,
        spacecraft=config.spacecraft if spacecraft is None else spacecraft,
        mpc=config.mpc if mpc is None else mpc,
        initial_state=initial_state,
        targets=TargetConfig(targets=target_table),
        show_progress=False,
        stop_when_complete=True,
    )


def run_single_target_case(
    name: str,
    label: str,
    config: ClosedLoopTestConfig,
) -> SingleTargetCaseResult:
    """Run and analyze one single-target case."""
    result = run_closed_loop_test(config)
    logs = result.to_dataframe()
    return SingleTargetCaseResult(
        name=name,
        label=label,
        result=result,
        logs=logs,
        metrics=compute_single_target_metrics(result, logs),
    )


def compute_single_target_metrics(
    result: ClosedLoopTestResult,
    logs: pd.DataFrame | None = None,
    *,
    settling_threshold_deg: float = 0.5,
    settling_dwell_s: float = 10.0,
    observation_threshold_deg: float = 0.01,
    observation_rate_threshold_deg_s: float = 0.01,
    steady_state_window_s: float = 10.0,
) -> dict[str, Any]:
    """Calculate repeatable pointing, estimation, control, and solver metrics."""
    if logs is None:
        logs = result.to_dataframe()
    if logs.empty:
        raise ValueError("Cannot compute metrics from an empty closed-loop result.")

    time = logs["time"].to_numpy(dtype=np.float64)
    final_time = float(time[-1] if result.final_time is None else result.final_time)
    durations = _sample_durations(time, final_time)
    pointing_error = logs["active_target_boresight_angle_deg"].to_numpy(
        dtype=np.float64
    )
    true_omega = logs[["true_omega_x", "true_omega_y", "true_omega_z"]].to_numpy(
        dtype=np.float64
    )
    angular_rate_deg_s = np.rad2deg(np.linalg.norm(true_omega, axis=1))
    control = logs[["control_tx", "control_ty", "control_tz"]].to_numpy(
        dtype=np.float64
    )

    finite_pointing = np.isfinite(pointing_error)
    settled = finite_pointing & (pointing_error <= settling_threshold_deg)
    observing_truth = (
        finite_pointing
        & (pointing_error <= observation_threshold_deg)
        & (angular_rate_deg_s <= observation_rate_threshold_deg_s)
    )
    planner_observing = logs["guidance_mode"].to_numpy() == "OBSERVING"
    complete = logs["is_complete"].to_numpy(dtype=bool)

    true_quaternion = logs[
        ["true_quat_w", "true_quat_x", "true_quat_y", "true_quat_z"]
    ].to_numpy(dtype=np.float64)
    estimated_quaternion = logs[
        [
            "estimated_quat_w",
            "estimated_quat_x",
            "estimated_quat_y",
            "estimated_quat_z",
        ]
    ].to_numpy(dtype=np.float64)
    estimation_error_deg = quaternion_angle_error_deg(
        true_quaternion, estimated_quaternion
    )

    limits = np.maximum(
        np.abs(result.config.spacecraft.u_min),
        np.abs(result.config.spacecraft.u_max),
    )
    saturated = np.any(np.abs(control) >= limits * (1.0 - 1e-9), axis=1)
    steady_mask = time >= max(0.0, final_time - steady_state_window_s)
    mpc_updates = logs["mpc_updated"].to_numpy(dtype=bool)
    mpc_success = logs["mpc_solve_success"].to_numpy(dtype=bool)
    mpc_fallback = logs["mpc_fallback_used"].to_numpy(dtype=bool)

    return {
        "final_time_s": final_time,
        "completed_observation": bool(np.any(complete)),
        "settling_time_s": sustained_entry_time(time, settled, settling_dwell_s),
        "settling_threshold_deg": settling_threshold_deg,
        "settling_dwell_s": settling_dwell_s,
        "truth_observation_criteria_time_s": sustained_entry_time(
            time, observing_truth, 0.0
        ),
        "planner_observing_entry_time_s": _first_true_time(time, planner_observing),
        "completion_time_s": _first_true_time(time, complete),
        "steady_state_pointing_rms_deg": _weighted_rms(
            pointing_error[steady_mask], durations[steady_mask]
        ),
        "peak_angular_rate_deg_s": float(np.nanmax(angular_rate_deg_s)),
        "control_effort_nm2_s": float(
            np.sum(np.sum(np.square(control), axis=1) * durations)
        ),
        "actuator_saturation_fraction": _weighted_fraction(saturated, durations),
        "attitude_estimation_rms_deg": _weighted_rms(estimation_error_deg, durations),
        "observation_criteria_fraction": _weighted_fraction(observing_truth, durations),
        "target_observation_time_s": float(logs["target_0_obs_time"].iloc[-1]),
        "mpc_update_count": int(np.count_nonzero(mpc_updates)),
        "mpc_solve_success_rate": (
            float(np.mean(mpc_success[mpc_updates]))
            if np.any(mpc_updates)
            else float("nan")
        ),
        "mpc_fallback_count": int(np.count_nonzero(mpc_fallback & mpc_updates)),
    }


def sustained_entry_time(
    time: FloatArray, condition: np.ndarray, dwell_time_s: float
) -> float:
    """Return the first true interval start lasting at least ``dwell_time_s``."""
    sample_time = np.asarray(time, dtype=np.float64)
    mask = np.asarray(condition, dtype=bool)
    if sample_time.ndim != 1 or mask.shape != sample_time.shape:
        raise ValueError("time and condition must be one-dimensional and equal length.")
    if dwell_time_s < 0.0:
        raise ValueError("dwell_time_s must be nonnegative.")
    if len(sample_time) == 0:
        return float("nan")

    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        end = index
        while end + 1 < len(mask) and mask[end + 1]:
            end += 1
        if sample_time[end] - sample_time[index] >= dwell_time_s - 1e-12:
            return float(sample_time[index])
        index = end + 1
    return float("nan")


def perturb_inertia(inertia: FloatArray, axis_factors: FloatArray) -> FloatArray:
    """Scale principal coordinate directions while preserving positive-definiteness."""
    matrix = np.asarray(inertia, dtype=np.float64)
    factors = np.asarray(axis_factors, dtype=np.float64)
    if matrix.shape != (3, 3) or factors.shape != (3,):
        raise ValueError("Expected inertia shape (3, 3) and axis_factors shape (3,).")
    if np.any(factors <= 0.0):
        raise ValueError("axis_factors must be positive.")
    scaling = np.diag(np.sqrt(factors))
    perturbed = scaling @ matrix @ scaling
    if np.any(np.linalg.eigvalsh(perturbed) <= 0.0):
        raise ValueError("Perturbed inertia must remain positive-definite.")
    return perturbed


def summarize_disturbance_torque(
    environment: ClosedLoopEnvironmentConfig,
    *,
    duration_s: float,
    sim_dt: float,
    rng_seed: int,
    actuator_min: FloatArray,
    actuator_max: FloatArray,
) -> dict[str, object]:
    """Summarize a configured disturbance profile against actuator capability."""
    if duration_s <= 0.0 or sim_dt <= 0.0:
        raise ValueError("duration_s and sim_dt must be positive.")
    sample_count = int(np.ceil(duration_s / sim_dt))
    time = np.arange(sample_count, dtype=np.float64) * sim_dt
    bias = (
        np.zeros(3, dtype=np.float64)
        if environment.disturbance_bias is None
        else np.asarray(environment.disturbance_bias, dtype=np.float64)
    )
    torque = np.tile(bias, (sample_count, 1))

    noise_std = environment.disturbance_noise_std
    if noise_std is not None and noise_std != 0.0:
        # DisturbanceModel uses NumPy's legacy global RNG. A local RandomState
        # reproduces that sequence without mutating process-global state.
        rng = np.random.RandomState(rng_seed)
        torque += rng.normal(0.0, float(noise_std), size=(sample_count, 3))

    amplitude = environment.disturbance_sinusoidal_amplitude
    frequency = environment.disturbance_sinusoidal_frequency
    if (
        amplitude is not None
        and frequency is not None
        and amplitude != 0.0
        and frequency != 0.0
    ):
        phase = 2.0 * np.pi * float(frequency) * time
        torque += float(amplitude) * np.column_stack(
            [
                np.sin(phase),
                np.sin(phase + 2.0 * np.pi / 3.0),
                np.sin(phase + 4.0 * np.pi / 3.0),
            ]
        )

    limits = np.maximum(
        np.abs(np.asarray(actuator_min, dtype=np.float64)),
        np.abs(np.asarray(actuator_max, dtype=np.float64)),
    )
    if limits.shape != (3,) or np.any(limits <= 0.0):
        raise ValueError("Actuator limits must define three positive axis magnitudes.")
    torque_norm = np.linalg.norm(torque, axis=1)
    mean_vector = np.mean(torque, axis=0)
    peak_abs_axis = np.max(np.abs(torque), axis=0)
    vector_limit = float(np.linalg.norm(limits))
    peak_axis_utilization = float(np.max(np.abs(torque) / limits))
    return {
        "disturbance_mean_x_nm": float(mean_vector[0]),
        "disturbance_mean_y_nm": float(mean_vector[1]),
        "disturbance_mean_z_nm": float(mean_vector[2]),
        "disturbance_mean_norm_nm": float(np.mean(torque_norm)),
        "disturbance_rms_norm_nm": float(np.sqrt(np.mean(np.square(torque_norm)))),
        "disturbance_peak_norm_nm": float(np.max(torque_norm)),
        "disturbance_peak_abs_x_nm": float(peak_abs_axis[0]),
        "disturbance_peak_abs_y_nm": float(peak_abs_axis[1]),
        "disturbance_peak_abs_z_nm": float(peak_abs_axis[2]),
        "actuator_axis_limit_x_nm": float(limits[0]),
        "actuator_axis_limit_y_nm": float(limits[1]),
        "actuator_axis_limit_z_nm": float(limits[2]),
        "actuator_vector_norm_limit_nm": vector_limit,
        "disturbance_mean_norm_fraction_of_vector_limit": float(
            np.mean(torque_norm) / vector_limit
        ),
        "disturbance_peak_norm_fraction_of_vector_limit": float(
            np.max(torque_norm) / vector_limit
        ),
        "disturbance_peak_axis_utilization": peak_axis_utilization,
        "disturbance_exceeds_axis_limit": peak_axis_utilization > 1.0,
    }


def _sample_durations(time: FloatArray, final_time: float) -> FloatArray:
    durations = np.diff(np.append(time, final_time))
    if np.any(durations < -1e-12):
        raise ValueError("Simulation times must be nondecreasing.")
    return np.clip(durations, 0.0, np.inf)


def _weighted_rms(values: FloatArray, weights: FloatArray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return float("nan")
    return float(
        np.sqrt(
            np.sum(np.square(values[finite]) * weights[finite])
            / np.sum(weights[finite])
        )
    )


def _weighted_fraction(condition: np.ndarray, weights: FloatArray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    return float(np.sum(weights[np.asarray(condition, dtype=bool)]) / total)


def _first_true_time(time: FloatArray, condition: np.ndarray) -> float:
    indices = np.flatnonzero(condition)
    return float(time[indices[0]]) if len(indices) else float("nan")
