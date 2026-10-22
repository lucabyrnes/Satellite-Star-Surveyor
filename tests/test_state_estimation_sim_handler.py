from typing import Any, cast

import numpy as np
import quaternion as qu
import pytest

from mpc_spacecraft.simulation.state_estimation_sim_handler import StateEstimationSimHandler
from mpc_spacecraft.utilities.utils import FullSimState, MeasuredState, SensorRigidBodyState


class DummyMeasurementGenerator:
    def __init__(self):
        self.calls = []

    def generate_state_measurement(self, state: FullSimState, prev_dt: float) -> MeasuredState:
        self.calls.append(state)

        measurement = MeasuredState.zeros()
        measurement.inertial_accel[:] = [1.0, 2.0, 3.0]
        measurement.omega[:] = [4.0, 5.0, 6.0]
        measurement.position[:] = [7.0, 8.0, 9.0]
        measurement.velocity[:] = [10.0, 11.0, 12.0]
        measurement.quat[:] = [1.0, 0.0, 0.0, 0.0]

        return measurement


class DummyMEKF:
    def __init__(self):
        self.calls = []

    def update(self, dt, imu, gyro, gnss_measure=None, st_measure=None):
        self.calls.append(
            {
                "dt": dt,
                "imu": np.array(imu, copy=True),
                "gyro": np.array(gyro, copy=True),
                "gnss_measure": None
                if gnss_measure is None
                else np.array(gnss_measure, copy=True),
                "st_measure": st_measure,
            }
        )

        return SensorRigidBodyState.zeros()


def make_handler() -> tuple[StateEstimationSimHandler, DummyMeasurementGenerator, DummyMEKF]:
    initial_state = SensorRigidBodyState.zeros()
    handler = StateEstimationSimHandler(
        initial_state,
        gnss_measurement_period=10.0,
        star_tracker_measurement_period=5.0,
        rng_seed=0,
    )

    measurement_gen = DummyMeasurementGenerator()
    mekf = DummyMEKF()
    test_handler = cast(Any, handler)
    test_handler.measurementGen = measurement_gen
    test_handler.mekf = mekf

    return handler, measurement_gen, mekf


def make_full_state() -> FullSimState:
    state = FullSimState.zeros()
    state.quat[:] = [1.0, 0.0, 0.0, 0.0]
    return state


def test_tick_uses_all_measurements_on_first_call_and_updates_timestamps() -> None:
    handler, measurement_gen, mekf = make_handler()
    state = make_full_state()

    estimated_state, used_star_tracker, used_gnss = handler.tick(
        current_time=1.0,
        past_dt=0.25,
        state=state,
    )

    assert isinstance(estimated_state, SensorRigidBodyState)
    assert used_star_tracker is True
    assert used_gnss is True
    assert measurement_gen.calls == [state]

    call = mekf.calls[-1]
    assert call["dt"] == pytest.approx(0.25)
    np.testing.assert_allclose(call["imu"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(call["gyro"], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(call["gnss_measure"], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    assert call["st_measure"] == qu.quaternion(1.0, 0.0, 0.0, 0.0)
    assert handler._past_gnss_measurement == pytest.approx(1.0)
    assert handler._past_star_tracker_measurement == pytest.approx(1.0)


def test_tick_omits_optional_measurements_until_each_period_elapsed() -> None:
    handler, _, mekf = make_handler()
    state = make_full_state()

    handler.tick(current_time=0.0, past_dt=0.1, state=state)
    handler.tick(current_time=4.0, past_dt=0.1, state=state)
    handler.tick(current_time=5.0, past_dt=0.1, state=state)
    handler.tick(current_time=10.0, past_dt=0.1, state=state)

    below_period_call = mekf.calls[1]
    assert below_period_call["gnss_measure"] is None
    assert below_period_call["st_measure"] is None
    assert handler._past_gnss_measurement == pytest.approx(10.0)
    assert handler._past_star_tracker_measurement == pytest.approx(10.0)

    star_only_call = mekf.calls[2]
    assert star_only_call["gnss_measure"] is None
    assert star_only_call["st_measure"] == qu.quaternion(1.0, 0.0, 0.0, 0.0)

    both_elapsed_call = mekf.calls[3]
    np.testing.assert_allclose(both_elapsed_call["gnss_measure"], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    assert both_elapsed_call["st_measure"] == qu.quaternion(1.0, 0.0, 0.0, 0.0)


def test_tick_treats_backwards_time_as_reset_for_all_periodic_measurements() -> None:
    handler, _, mekf = make_handler()
    state = make_full_state()

    handler.tick(current_time=20.0, past_dt=0.1, state=state)
    handler.tick(current_time=2.0, past_dt=0.1, state=state)

    reset_call = mekf.calls[-1]
    np.testing.assert_allclose(reset_call["gnss_measure"], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    assert reset_call["st_measure"] == qu.quaternion(1.0, 0.0, 0.0, 0.0)
    assert handler._past_gnss_measurement == pytest.approx(2.0)
    assert handler._past_star_tracker_measurement == pytest.approx(2.0)
