import numpy as np
import pytest

from mpc_spacecraft.analysis.timestep_convergence import (
    estimate_observed_order,
    quaternion_angular_distance_rad,
    run_closed_loop_convergence,
    run_dynamics_convergence,
    validate_step_grid,
)
from mpc_spacecraft.config import ClosedLoopTestConfig
from mpc_spacecraft.estimation.sensor_models import (
    GNSSConfig,
    GyroConfig,
    IMUConfig,
    StarTrackerConfig,
)
from mpc_spacecraft.simulation.sensor_state_updater import SensorBiasUpdater
from mpc_spacecraft.simulation.state_estimation_sim_handler import (
    StateEstimationSimHandler,
)
from mpc_spacecraft.simulation.state_measurement_gen import StateMeasurementGenerator
from mpc_spacecraft.utilities.utils import FullSimState


@pytest.mark.unit
def test_quaternion_distance_is_sign_invariant() -> None:
    quaternion = np.array([0.5, -0.5, 0.5, -0.5])
    assert quaternion_angular_distance_rad(quaternion, -quaternion) == pytest.approx(
        0.0
    )


@pytest.mark.unit
def test_observed_order_recovers_fourth_order_sequence() -> None:
    timesteps = np.array([0.08, 0.04, 0.02, 0.01])
    errors = 3.2 * timesteps**4
    assert estimate_observed_order(timesteps, errors) == pytest.approx(4.0)


@pytest.mark.unit
def test_step_grid_rejects_event_period_not_aligned_with_every_timestep() -> None:
    with pytest.raises(ValueError, match="event period"):
        validate_step_grid(
            timesteps=[0.08, 0.04],
            reference_dt=0.01,
            duration=1.6,
            event_periods=[0.1],
        )


@pytest.mark.integration
def test_rotation_dynamics_exhibits_fourth_order_convergence() -> None:
    result = run_dynamics_convergence(
        timesteps=[0.08, 0.04, 0.02, 0.01],
        reference_dt=0.00125,
        duration=1.6,
    )
    assert result.fitted_attitude_order > 3.5
    assert result.fitted_angular_velocity_order > 3.5
    assert result.principal_axis_max_error_rad < 1e-10


@pytest.mark.unit
def test_disabled_measurement_noise_returns_exact_measurement() -> None:
    state = FullSimState.zeros()
    state.position[:] = [1.0, 2.0, 3.0]
    state.velocity[:] = [4.0, 5.0, 6.0]
    state.quat[:] = [1.0, 0.0, 0.0, 0.0]
    state.omega[:] = [0.1, 0.2, 0.3]
    state.inertial_accel[:] = [0.4, 0.5, 0.6]
    state.gyro_bias[:] = [0.01, 0.02, 0.03]
    state.accel_bias[:] = [0.04, 0.05, 0.06]
    generator = StateMeasurementGenerator(
        IMUConfig(),
        GyroConfig(),
        GNSSConfig(),
        StarTrackerConfig(),
        rng_seed=7,
        enable_noise=False,
    )

    measurement = generator.generate_state_measurement(state, prev_dt=0.1)

    np.testing.assert_allclose(measurement.translation.data, state.translation.data)
    np.testing.assert_allclose(measurement.quat, state.quat)
    np.testing.assert_allclose(measurement.omega, state.omega + state.gyro_bias)
    np.testing.assert_allclose(
        measurement.inertial_accel, state.inertial_accel + state.accel_bias
    )


@pytest.mark.unit
def test_disabled_bias_random_walk_leaves_state_unchanged() -> None:
    state = FullSimState.zeros()
    state.gyro_bias[:] = [1.0, 2.0, 3.0]
    state.accel_bias[:] = [4.0, 5.0, 6.0]
    updater = SensorBiasUpdater(IMUConfig(), GyroConfig(), rng_seed=3, enabled=False)

    updated = updater.tick(state, dt=1.0)

    np.testing.assert_allclose(updated.data, state.data)
    assert updated is not state


@pytest.mark.unit
def test_measurement_schedule_tolerates_roundoff_at_period_boundary() -> None:
    assert StateEstimationSimHandler._is_measurement_due(
        current_time=5.099999999999999,
        past_measurement_time=5.0,
        measurement_period=0.1,
    )


@pytest.mark.integration
def test_closed_loop_error_statistics_include_terminal_state() -> None:
    result = run_closed_loop_convergence(
        ClosedLoopTestConfig.defaults(show_progress=False),
        timesteps=[0.05],
        reference_dt=0.003125,
        duration=0.1,
    )
    metrics = result.metrics.iloc[0]

    assert metrics["position_error_final_m"] <= metrics["position_error_max_m"]
    assert metrics["velocity_error_final_mps"] <= metrics["velocity_error_max_mps"]
    assert metrics["attitude_error_final_rad"] <= metrics["attitude_error_max_rad"]
    assert (
        metrics["angular_velocity_error_final_radps"]
        <= metrics["angular_velocity_error_max_radps"]
    )
