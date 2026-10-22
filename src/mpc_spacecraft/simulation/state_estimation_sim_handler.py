from mpc_spacecraft.estimation.MEKF import MEKF
from mpc_spacecraft.simulation.state_measurement_gen import StateMeasurementGenerator
from mpc_spacecraft.estimation.sensor_models import (
    GNSSConfig,
    GyroConfig,
    IMUConfig,
    StarTrackerConfig,
    make_mekf_initial_covariance,
)
from mpc_spacecraft.utilities.utils import FullSimState, SensorRigidBodyState

import quaternion as qu


class StateEstimationSimHandler:

    def __init__(
        self,
        intial_sensor_state: SensorRigidBodyState,
        gnss_measurement_period: float,
        star_tracker_measurement_period: float,
        rng_seed: int,
        enable_measurement_noise: bool = True,
    ):

        self.mekf = MEKF(
            intial_sensor_state,
            IMUConfig(),
            GyroConfig(),
            GNSSConfig(),
            StarTrackerConfig(),
            start_P=make_mekf_initial_covariance(),
        )
        self.measurementGen = StateMeasurementGenerator(
            IMUConfig(),
            GyroConfig(),
            GNSSConfig(),
            StarTrackerConfig(),
            rng_seed,
            enable_noise=enable_measurement_noise,
        )

        self._gnss_period = gnss_measurement_period
        self._past_gnss_measurement = None

        self._star_tracker_period = star_tracker_measurement_period
        self._past_star_tracker_measurement = None

    def tick(
        self, current_time: float, past_dt: float, state: FullSimState
    ) -> tuple[SensorRigidBodyState, bool, bool]:
        measurement = self.measurementGen.generate_state_measurement(state, past_dt)

        is_time_reset = self._has_time_moved_backward(current_time)

        use_gnss = is_time_reset or self._is_measurement_due(
            current_time,
            self._past_gnss_measurement,
            self._gnss_period,
        )
        use_star_tracker = is_time_reset or self._is_measurement_due(
            current_time,
            self._past_star_tracker_measurement,
            self._star_tracker_period,
        )

        gnss_measure = measurement.translation.data if use_gnss else None
        st_measure = qu.quaternion(*measurement.quat) if use_star_tracker else None

        estimated_state = self.mekf.update(
            past_dt,
            measurement.inertial_accel,
            measurement.omega,
            gnss_measure=gnss_measure,
            st_measure=st_measure,
        )

        if use_gnss:
            self._past_gnss_measurement = current_time

        if use_star_tracker:
            self._past_star_tracker_measurement = current_time

        return estimated_state, use_star_tracker, use_gnss

    def _has_time_moved_backward(self, current_time: float) -> bool:
        past_measurement_times = (
            self._past_gnss_measurement,
            self._past_star_tracker_measurement,
        )

        return any(
            past_time is not None and current_time < past_time
            for past_time in past_measurement_times
        )

    @staticmethod
    def _is_measurement_due(
        current_time: float,
        past_measurement_time: float | None,
        measurement_period: float,
    ) -> bool:
        if past_measurement_time is None:
            return True

        elapsed = current_time - past_measurement_time
        tolerance = 1e-12 * max(1.0, abs(current_time), abs(past_measurement_time))
        return elapsed >= measurement_period - tolerance
