"""Reusable closed-loop spacecraft simulation test API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import cast

import numpy as np
import pandas as pd
import quaternion as qu

from mpc_spacecraft.config import ClosedLoopTestConfig
from mpc_spacecraft.controllers import NominalMPC
from mpc_spacecraft.controllers.error_dynamics_adapters import (
    SpacecraftErrorDynamicsProvider,
)
from mpc_spacecraft.dynamics import DisturbanceModel, SpacecraftDynamics
from mpc_spacecraft.estimation.sensor_models import GyroConfig, IMUConfig
from mpc_spacecraft.estimation.torque_disturbance_observer import (
    ConstantTorqueObserver,
)
from mpc_spacecraft.guidance import AstropySunDirectionModel, Guidance
from mpc_spacecraft.planner import StarPlanner
from mpc_spacecraft.simulation.clock import SimulationClock
from mpc_spacecraft.simulation.sensor_state_updater import SensorBiasUpdater
from mpc_spacecraft.simulation.state_estimation_sim_handler import (
    StateEstimationSimHandler,
)
from mpc_spacecraft.utilities.utils import (
    BODY_FORWARD_VEC3,
    FloatArray,
    FullSimState,
    R_EARTH_M,
    RigidBodyState,
    RigidBodyControl,
    RotationState,
    SensorRigidBodyState,
)


@dataclass(frozen=True)
class ClosedLoopLogRecord:
    """A single plotting-friendly closed-loop simulation log record."""

    cycle: int
    time: float
    epoch_utc: datetime
    true_state: FloatArray
    estimated_state: FloatArray
    goal_rotation_state: FloatArray
    control: FloatArray
    estimated_disturbance_torque: FloatArray
    guidance_mode_value: int
    guidance_mode_name: str
    is_complete: bool
    mpc_updated: bool
    mpc_solve_success: bool
    mpc_fallback_used: bool
    estimator_used_gnss: bool
    estimator_used_star_tracker: bool
    mekf_covariance_diag: FloatArray | None
    current_target_idx: int | None
    target_obs_times: FloatArray | None
    active_target_boresight_angle_deg: float | None = None
    active_target_sun_angle_deg: float | None = None
    active_target_earth_angle_deg: float | None = None
    active_target_sun_keepout_margin_deg: float | None = None
    active_target_earth_keepout_margin_deg: float | None = None


@dataclass
class ClosedLoopTestResult:
    """Closed-loop test result and log extraction helper."""

    config: ClosedLoopTestConfig
    records: list[ClosedLoopLogRecord] = field(default_factory=list)
    final_state: FullSimState | None = None
    final_time: float | None = None

    def append(self, record: ClosedLoopLogRecord) -> None:
        """Append one log record."""
        self.records.append(record)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Return logged data as numpy arrays for plotting and analysis."""
        if not self.records:
            return {}

        return {
            "cycle": np.array([r.cycle for r in self.records], dtype=np.int64),
            "time": np.array([r.time for r in self.records], dtype=np.float64),
            "true_state": np.array([r.true_state for r in self.records], dtype=np.float64),
            "estimated_state": np.array(
                [r.estimated_state for r in self.records], dtype=np.float64
            ),
            "goal_rotation_state": np.array(
                [r.goal_rotation_state for r in self.records], dtype=np.float64
            ),
            "control": np.array([r.control for r in self.records], dtype=np.float64),
            "estimated_disturbance_torque": np.array(
                [r.estimated_disturbance_torque for r in self.records],
                dtype=np.float64,
            ),
            "guidance_mode_value": np.array(
                [r.guidance_mode_value for r in self.records], dtype=np.int64
            ),
            "is_complete": np.array([r.is_complete for r in self.records], dtype=bool),
            "mpc_updated": np.array([r.mpc_updated for r in self.records], dtype=bool),
            "mpc_solve_success": np.array(
                [r.mpc_solve_success for r in self.records], dtype=bool
            ),
            "mpc_fallback_used": np.array(
                [r.mpc_fallback_used for r in self.records], dtype=bool
            ),
            "estimator_used_gnss": np.array(
                [r.estimator_used_gnss for r in self.records], dtype=bool
            ),
            "estimator_used_star_tracker": np.array(
                [r.estimator_used_star_tracker for r in self.records], dtype=bool
            ),
            "mekf_covariance_diag": np.array(
                [
                    np.full(15, np.nan) if r.mekf_covariance_diag is None else r.mekf_covariance_diag
                    for r in self.records
                ],
                dtype=np.float64,
            ),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return flattened logged data as a pandas DataFrame."""
        rows = []
        for record in self.records:
            true_state = SensorRigidBodyState.from_array(record.true_state)
            estimated_state = SensorRigidBodyState.from_array(record.estimated_state)
            goal_state = RotationState.from_array(record.goal_rotation_state)
            row = {
                "cycle": record.cycle,
                "time": record.time,
                "epoch_utc": record.epoch_utc,
                "guidance_mode_value": record.guidance_mode_value,
                "guidance_mode": record.guidance_mode_name,
                "is_complete": record.is_complete,
                "mpc_updated": record.mpc_updated,
                "mpc_solve_success": record.mpc_solve_success,
                "mpc_fallback_used": record.mpc_fallback_used,
                "estimator_used_gnss": record.estimator_used_gnss,
                "estimator_used_star_tracker": record.estimator_used_star_tracker,
                "current_target_idx": record.current_target_idx,
                "active_target_boresight_angle_deg": record.active_target_boresight_angle_deg,
                "active_target_sun_angle_deg": record.active_target_sun_angle_deg,
                "active_target_earth_angle_deg": record.active_target_earth_angle_deg,
                "active_target_sun_keepout_margin_deg": record.active_target_sun_keepout_margin_deg,
                "active_target_earth_keepout_margin_deg": record.active_target_earth_keepout_margin_deg,
                "control_tx": record.control[0],
                "control_ty": record.control[1],
                "control_tz": record.control[2],
                "estimated_disturbance_tx": record.estimated_disturbance_torque[0],
                "estimated_disturbance_ty": record.estimated_disturbance_torque[1],
                "estimated_disturbance_tz": record.estimated_disturbance_torque[2],
            }
            if record.mekf_covariance_diag is not None:
                for idx, value in enumerate(record.mekf_covariance_diag):
                    row[f"mekf_cov_diag_{idx:02d}"] = value
            if record.target_obs_times is not None:
                for idx, value in enumerate(record.target_obs_times):
                    row[f"target_{idx}_obs_time"] = value
            _add_vec_columns(row, "true_position", true_state.position)
            _add_vec_columns(row, "true_velocity", true_state.velocity)
            _add_quat_columns(row, "true_quat", true_state.quat)
            _add_vec_columns(row, "true_omega", true_state.omega)
            _add_vec_columns(row, "estimated_position", estimated_state.position)
            _add_vec_columns(row, "estimated_velocity", estimated_state.velocity)
            _add_quat_columns(row, "estimated_quat", estimated_state.quat)
            _add_vec_columns(row, "estimated_omega", estimated_state.omega)
            _add_vec_columns(row, "gyro_bias", true_state.gyro_bias)
            _add_vec_columns(row, "accel_bias", true_state.accel_bias)
            _add_quat_columns(row, "goal_quat", goal_state.quat)
            _add_vec_columns(row, "goal_omega", goal_state.omega)
            rows.append(row)

        return pd.DataFrame(rows)


def run_closed_loop_test(config: ClosedLoopTestConfig | None = None) -> ClosedLoopTestResult:
    """Run a reusable closed-loop spacecraft test and return structured logs."""
    if config is None:
        config = ClosedLoopTestConfig.defaults()

    timing = config.timing
    environment = config.environment
    spacecraft = config.spacecraft
    mpc_config = config.mpc
    sensors = config.sensors

    sim_state = config.initial_state.to_full_sim_state()
    clock = SimulationClock(timing.start_epoch_utc, timing.sim_dt)
    sun_model = AstropySunDirectionModel()
    planner = StarPlanner(
        config.targets.targets.copy(),
        environment.sun_keep_out_deg,
        environment.earth_margin_deg,
    )
    guidance = Guidance(planner, sun_model)

    disturbance_model = DisturbanceModel(
        environment.disturbance_bias,
        environment.disturbance_noise_std,
        environment.disturbance_sinusoidal_amplitude,
        environment.disturbance_sinusoidal_frequency,
        config.rng_seed,
        enable_gravity=environment.enable_gravity,
    )
    truth_dynamics = SpacecraftDynamics(
        spacecraft.inertia, disturbance_model, spacecraft.mass
    )
    prediction_inertia = (
        spacecraft.inertia
        if mpc_config.model_inertia is None
        else mpc_config.model_inertia
    )
    prediction_dynamics = SpacecraftDynamics(
        prediction_inertia,
        DisturbanceModel(enable_gravity=False),
        spacecraft.mass,
    )
    error_dynamics_provider = SpacecraftErrorDynamicsProvider(
        prediction_dynamics,
        sun_model,
        np.deg2rad(environment.sun_keep_out_deg),
        timing.mpc_dt,
    )
    mpc = NominalMPC(
        mpc_config.step_horizon,
        mpc_config.q_weight,
        mpc_config.r_weight,
        error_dynamics_provider,
        mpc_config.q_terminal if mpc_config.q_terminal is not None else mpc_config.q_weight,
        spacecraft.u_min,
        spacecraft.u_max,
    )
    disturbance_observer = (
        ConstantTorqueObserver(
            inertia=prediction_inertia,
            time_constant_s=mpc_config.disturbance_observer_time_constant_s,
            max_abs_torque=mpc_config.disturbance_observer_max_abs_torque,
            initial_torque=mpc_config.disturbance_observer_initial_torque,
        )
        if mpc_config.enable_disturbance_rejection
        else None
    )
    estimator = StateEstimationSimHandler(
        sim_state.sensor_rigid_body,
        sensors.gnss_measurement_period,
        sensors.star_tracker_measurement_period,
        config.rng_seed,
        enable_measurement_noise=sensors.enable_measurement_noise,
    )
    sensor_bias_updater = SensorBiasUpdater(
        IMUConfig(),
        GyroConfig(),
        config.rng_seed,
        enabled=sensors.enable_bias_random_walk,
    )

    result = ClosedLoopTestResult(config=config)
    total_control = RigidBodyControl.zeros()
    estimated_disturbance_torque = np.zeros(3, dtype=np.float64)
    previous_mpc_guidance_update: float | None = None
    goal_rotation_state = sim_state.rotation.copy()
    mode: IntEnum | None = None
    is_complete = False
    mpc_solve_success = False
    mpc_fallback_used = False
    estimator_used_star_tracker = False
    estimator_used_gnss = False

    iterator = range(timing.sim_cycles)
    if config.show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator)

    for sim_cycle in iterator:
        if clock.last_dt != 0:
            estimated_state, estimator_used_star_tracker, estimator_used_gnss = estimator.tick(
                clock.current_time, clock.last_dt, sim_state
            )
        else:
            estimated_state = sim_state.sensor_rigid_body
            estimator_used_star_tracker = False
            estimator_used_gnss = False

        mpc_elapsed = (
            0.0
            if previous_mpc_guidance_update is None
            else clock.current_time - previous_mpc_guidance_update
        )
        timing_tolerance = 1e-12 * max(1.0, abs(clock.current_time))
        mpc_update = previous_mpc_guidance_update is None or (
            mpc_elapsed >= timing.mpc_dt - timing_tolerance
        )
        if mpc_update:
            guidance_delta_time = (
                0.0
                if previous_mpc_guidance_update is None
                else clock.current_time - previous_mpc_guidance_update
            )
            goal_rotation_state, mode, is_complete = guidance.tick(
                clock.now(), guidance_delta_time, cast(RigidBodyState, estimated_state.rigid_body)
            )
            if disturbance_observer is not None:
                estimated_disturbance_torque = disturbance_observer.update(
                    estimated_state.rotation,
                    total_control.torque,
                    guidance_delta_time,
                )
            rotation_control = mpc.get_first_control(
                x0=estimated_state.rotation,
                x_goal=goal_rotation_state,
                disturbance_torque=estimated_disturbance_torque,
                current_epoch_utc=clock.now(),
            )
            mpc_solve_success = bool(mpc.last_solve_success)
            mpc_fallback_used = bool(mpc.last_fallback_used)
            total_control = RigidBodyControl.zeros()
            total_control.rotation[:] = rotation_control
            previous_mpc_guidance_update = clock.current_time

        active_geometry = _compute_active_target_geometry(
            true_state=sim_state.rigid_body,
            target_table=planner._targets,
            current_target_idx=getattr(planner, "_current_target_idx", None),
            sun_dir_I=sun_model.sun_dir_eci(clock.now()),
            sun_keepout_deg=environment.sun_keep_out_deg,
            earth_margin_deg=environment.earth_margin_deg,
        )

        result.append(
            ClosedLoopLogRecord(
                cycle=sim_cycle,
                time=clock.current_time,
                epoch_utc=clock.now(),
                true_state=sim_state.sensor_rigid_body.data.copy(),
                estimated_state=estimated_state.data.copy(),
                goal_rotation_state=goal_rotation_state.data.copy(),
                control=total_control.torque.copy(),
                estimated_disturbance_torque=estimated_disturbance_torque.copy(),
                guidance_mode_value=int(mode) if mode is not None else -1,
                guidance_mode_name=mode.name if mode is not None else "UNKNOWN",
                is_complete=bool(is_complete),
                mpc_updated=bool(mpc_update),
                mpc_solve_success=bool(mpc_solve_success),
                mpc_fallback_used=bool(mpc_fallback_used),
                estimator_used_gnss=bool(estimator_used_gnss),
                estimator_used_star_tracker=bool(estimator_used_star_tracker),
                mekf_covariance_diag=np.diag(estimator.mekf.P).copy(),
                current_target_idx=getattr(planner, "_current_target_idx", None),
                target_obs_times=planner._targets["obs_time"].to_numpy().copy(),
                active_target_boresight_angle_deg=active_geometry["boresight_angle_deg"],
                active_target_sun_angle_deg=active_geometry["sun_angle_deg"],
                active_target_earth_angle_deg=active_geometry["earth_angle_deg"],
                active_target_sun_keepout_margin_deg=active_geometry["sun_keepout_margin_deg"],
                active_target_earth_keepout_margin_deg=active_geometry["earth_keepout_margin_deg"],
            )
        )

        if config.stop_when_complete and is_complete:
            break

        time_step = clock.advance(1)
        sim_state.rigid_body[:] = truth_dynamics.discretize_dynamics_rk4_full_state(
            sim_state.rigid_body,
            total_control,
            time_step.t0,
            time_step.dt,
        ).data
        sim_state = sensor_bias_updater.tick(sim_state, time_step.dt)

    result.final_state = sim_state.copy()
    result.final_time = clock.current_time
    return result


def _add_vec_columns(row: dict[str, object], prefix: str, values: FloatArray) -> None:
    row[f"{prefix}_x"] = values[0]
    row[f"{prefix}_y"] = values[1]
    row[f"{prefix}_z"] = values[2]


def _add_quat_columns(row: dict[str, object], prefix: str, values: FloatArray) -> None:
    row[f"{prefix}_w"] = values[0]
    row[f"{prefix}_x"] = values[1]
    row[f"{prefix}_y"] = values[2]
    row[f"{prefix}_z"] = values[3]


def _compute_active_target_geometry(
    true_state: RigidBodyState,
    target_table: pd.DataFrame,
    current_target_idx: int | None,
    sun_dir_I: FloatArray,
    sun_keepout_deg: float,
    earth_margin_deg: float,
) -> dict[str, float | None]:
    """Compute active target pointing and keepout geometry for logging."""
    empty = {
        "boresight_angle_deg": None,
        "sun_angle_deg": None,
        "earth_angle_deg": None,
        "sun_keepout_margin_deg": None,
        "earth_keepout_margin_deg": None,
    }
    if current_target_idx is None or current_target_idx not in target_table.index:
        return empty

    target_vec = target_table.loc[current_target_idx][["x", "y", "z"]].to_numpy(dtype=np.float64)
    target_vec = target_vec / np.linalg.norm(target_vec)
    earth_dir_I = -true_state.position / np.linalg.norm(true_state.position)
    body_dir_I = qu.rotate_vectors(qu.quaternion(*true_state.quat), BODY_FORWARD_VEC3)
    body_dir_I = body_dir_I / np.linalg.norm(body_dir_I)
    sun_unit_I = sun_dir_I / np.linalg.norm(sun_dir_I)

    boresight_angle = np.rad2deg(np.arccos(np.clip(target_vec @ body_dir_I, -1.0, 1.0)))
    sun_angle = np.rad2deg(np.arccos(np.clip(target_vec @ sun_unit_I, -1.0, 1.0)))
    earth_angle = np.rad2deg(np.arccos(np.clip(target_vec @ earth_dir_I, -1.0, 1.0)))
    r = np.linalg.norm(true_state.position)
    earth_occlusion_half_angle_deg = np.rad2deg(np.arcsin(np.clip(R_EARTH_M / r, -1.0, 1.0)))

    return {
        "boresight_angle_deg": float(boresight_angle),
        "sun_angle_deg": float(sun_angle),
        "earth_angle_deg": float(earth_angle),
        "sun_keepout_margin_deg": float(sun_angle - sun_keepout_deg),
        "earth_keepout_margin_deg": float(earth_angle - earth_occlusion_half_angle_deg - earth_margin_deg),
    }
