from mpc_spacecraft.estimation.sensor_models import (
    GNSSConfig,
    GyroConfig,
    IMUConfig,
    StarTrackerConfig,
)
from mpc_spacecraft.utilities.utils import FullSimState, MeasuredState

import numpy as np
import quaternion as qu


class StateMeasurementGenerator:

    def __init__(
        self,
        imu_config: IMUConfig,
        gyro_config: GyroConfig,
        gnss_config: GNSSConfig,
        star_config: StarTrackerConfig,
        rng_seed: int,
        enable_noise: bool = True,
    ):

        self.cov_accel = imu_config.sigma_a2
        self.cov_omega = gyro_config.sigma_g2
        self.cov_gnss = gnss_config.R
        self.cov_star_tracker = star_config.R
        self.enable_noise = enable_noise

        self.rng = np.random.default_rng(seed=rng_seed)

    def generate_state_measurement(
        self, state: FullSimState, prev_dt: float
    ) -> MeasuredState:

        if not self.enable_noise:
            x = MeasuredState.zeros()
            x.inertial_accel[:] = state.inertial_accel + state.accel_bias
            x.translation[:] = state.translation.data
            x.quat[:] = state.quat
            x.omega[:] = state.omega + state.gyro_bias
            return x

        # Continuous to discrete divide by dt
        sample_accel = (
            self.rng.multivariate_normal(
                mean=state.inertial_accel, cov=self.cov_accel / prev_dt
            )
            + state.accel_bias
        )

        # Continuous to discrete divide by dt
        sample_omega = (
            self.rng.multivariate_normal(mean=state.omega, cov=self.cov_omega / prev_dt)
            + state.gyro_bias
        )

        sample_gnss = self.rng.multivariate_normal(
            mean=state.translation.data, cov=self.cov_gnss
        )

        sample_star_tracker_theta = self.rng.multivariate_normal(
            mean=np.zeros(self.cov_star_tracker.shape[0]), cov=self.cov_star_tracker
        )

        sample_rotation = qu.from_rotation_vector(
            sample_star_tracker_theta
        ) * qu.quaternion(*state.quat)
        sample_quat = qu.as_float_array(sample_rotation)

        x = MeasuredState.zeros()
        x.inertial_accel[:] = sample_accel
        x.translation[:] = sample_gnss
        x.quat[:] = sample_quat
        x.omega[:] = sample_omega

        return x
