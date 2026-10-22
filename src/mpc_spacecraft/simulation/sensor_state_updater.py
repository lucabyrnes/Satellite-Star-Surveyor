from mpc_spacecraft.estimation.sensor_models import GyroConfig, IMUConfig
from mpc_spacecraft.utilities.utils import FullSimState

import numpy as np


class SensorBiasUpdater:

    def __init__(
        self,
        imu_config: IMUConfig,
        gyro_config: GyroConfig,
        rng_seed: int | None = None,
        enabled: bool = True,
    ):
        self.accel_bias_cov = imu_config.sigma_ba2
        self.gyro_bias_cov = gyro_config.sigma_bg2
        self.rng = np.random.default_rng(seed=rng_seed)
        self.enabled = enabled

    def tick(self, state: FullSimState, dt: float) -> FullSimState:
        next_state = FullSimState.from_array(state.data.copy())

        if not self.enabled:
            return next_state

        next_state.sensor_rigid_body.accel_bias[:] += self.rng.multivariate_normal(
            mean=np.zeros(3),
            cov=self.accel_bias_cov * dt,
        )
        next_state.sensor_rigid_body.gyro_bias[:] += self.rng.multivariate_normal(
            mean=np.zeros(3),
            cov=self.gyro_bias_cov * dt,
        )

        return next_state
