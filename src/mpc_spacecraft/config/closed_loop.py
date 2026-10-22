"""Closed-loop simulation configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from mpc_spacecraft.utilities.utils import FloatArray, FullSimState


@dataclass(frozen=True)
class ClosedLoopTimingConfig:
    """Timing parameters for a closed-loop test."""

    sim_dt: float = 0.01
    mpc_dt: float = 5
    sim_cycles: int = 120_000
    start_epoch_utc: datetime = field(default_factory=lambda: datetime(2025, 1, 9))


@dataclass(frozen=True)
class ClosedLoopEnvironmentConfig:
    """Environment and guidance constraint parameters."""

    sun_keep_out_deg: float = 35.0
    earth_margin_deg: float = 10.0

    # Unmodeled constant disturbance torque, N·m.
    disturbance_bias: FloatArray | None = field(
        default_factory=lambda: np.array([5.0e-6, -3.0e-6, 2.0e-6], dtype=np.float64)
    )

    # Low-frequency random torque, N·m RMS.
    disturbance_noise_std: float | None = 1.0e-7

    # Slow deterministic disturbance, mostly SRP / thermal / orbit geometry.
    disturbance_sinusoidal_amplitude: float | None = 5.0e-6

    # Environmental disturbance frequency, Hz.
    # 3e-4 Hz ≈ 55.6 min period.
    disturbance_sinusoidal_frequency: float | None = 3.0e-4

    enable_gravity: bool = True

@dataclass(frozen=True)
class SpacecraftPhysicalConfig:
    """Spacecraft physical parameters for closed-loop testing."""

    inertia: FloatArray = field(
        default_factory=lambda: np.array(
            [
                [2600.0, -120.0,   80.0],
                [-120.0, 2400.0,  -60.0],
                [  80.0,  -60.0, 1500.0],
            ],
            dtype=np.float64,
        )
    )
    mass: float = 1_200.0
    u_min: FloatArray = field(
        default_factory=lambda: -0.25 * np.ones(3, dtype=np.float64)
    )
    u_max: FloatArray = field(
        default_factory=lambda: 0.25 * np.ones(3, dtype=np.float64)
    )


@dataclass(frozen=True)
class ClosedLoopMPCConfig:
    """MPC tuning parameters."""

    step_horizon: int = 30
    q_weight: FloatArray = field(
        default_factory=lambda: np.diag(
            [20.0, 20.0, 20.0, 5.0, 5.0, 5.0]
        ).astype(np.float64)
    )
    r_weight: FloatArray = field(
        default_factory=lambda: 1.0 * np.eye(3, dtype=np.float64)
    )
    q_terminal: FloatArray | None = None
    model_inertia: FloatArray | None = None


@dataclass(frozen=True)
class SensorScheduleConfig:
    """Sensor sampling schedule used by the estimator simulation."""

    gnss_measurement_period: float = 1.0
    star_tracker_measurement_period: float = 0.1
    enable_measurement_noise: bool = True
    enable_bias_random_walk: bool = True


@dataclass(frozen=True)
class InitialStateConfig:
    """Initial full simulation state parameters."""

    position: FloatArray = field(
        default_factory=lambda: np.array([7_078e3, 0.0, 0.0], dtype=np.float64)
    )
    velocity: FloatArray = field(
        default_factory=lambda: np.array([0.0, 7.504e3, 0.0], dtype=np.float64)
    )
    quat: FloatArray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    omega: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    gyro_bias: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    accel_bias: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def to_full_sim_state(self) -> FullSimState:
        """Create a mutable full simulation state from this config."""
        state = FullSimState.zeros()
        state.position[:] = self.position
        state.velocity[:] = self.velocity
        state.quat[:] = self.quat
        state.omega[:] = self.omega
        state.gyro_bias[:] = self.gyro_bias
        state.accel_bias[:] = self.accel_bias
        return state


@dataclass(frozen=True)
class TargetConfig:
    """Target table parameters for star-observation closed-loop tests."""

    targets: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            {
                "req_obs_time": [20.0, 20.0, 20.0],
                "Dec": [0.174533, -0.349066, 0.610865],
                "RA": [0.261799, 1.396263, 2.530727],
            }
        )
    )


@dataclass(frozen=True)
class ClosedLoopTestConfig:
    """Top-level configuration for ``run_closed_loop_test``."""

    timing: ClosedLoopTimingConfig = field(default_factory=ClosedLoopTimingConfig)
    environment: ClosedLoopEnvironmentConfig = field(default_factory=ClosedLoopEnvironmentConfig)
    spacecraft: SpacecraftPhysicalConfig = field(default_factory=SpacecraftPhysicalConfig)
    mpc: ClosedLoopMPCConfig = field(default_factory=ClosedLoopMPCConfig)
    sensors: SensorScheduleConfig = field(default_factory=SensorScheduleConfig)
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    rng_seed: int = 2
    show_progress: bool = True
    stop_when_complete: bool = False

    @classmethod
    def defaults(cls, **overrides: Any) -> "ClosedLoopTestConfig":
        """Return the default config, optionally overriding top-level fields."""
        return cls(**overrides)
