"""Observer for slowly varying external spacecraft torque."""

from __future__ import annotations

import numpy as np

from mpc_spacecraft.utilities.utils import FloatArray, RotationState, as_vec3


class ConstantTorqueObserver:
    """Estimate a body-frame torque bias from angular-rate model residuals.

    The observer applies Euler's rigid-body equation to consecutive
    angular-rate estimates, subtracts the torque that was applied over that
    interval, and low-pass filters the resulting disturbance estimate.
    """

    def __init__(
        self,
        inertia: FloatArray,
        time_constant_s: float,
        max_abs_torque: FloatArray,
        initial_torque: FloatArray | None = None,
    ) -> None:
        inertia_array = np.asarray(inertia, dtype=np.float64)
        if inertia_array.shape != (3, 3):
            raise ValueError(
                f"Expected inertia shape (3, 3), got {inertia_array.shape}."
            )
        if not np.all(np.isfinite(inertia_array)):
            raise ValueError("Inertia contains NaN or infinite values.")
        if time_constant_s <= 0.0 or not np.isfinite(time_constant_s):
            raise ValueError("time_constant_s must be finite and positive.")

        limits = as_vec3(max_abs_torque).copy()
        if np.any(limits <= 0.0):
            raise ValueError("max_abs_torque must be positive on every axis.")

        self._inertia = inertia_array.copy()
        self._time_constant_s = float(time_constant_s)
        self._max_abs_torque = limits
        self._estimate = np.zeros(3, dtype=np.float64)
        self._previous_omega: FloatArray | None = None
        self.reset(initial_torque)

    @property
    def estimate(self) -> FloatArray:
        """Return a copy of the current body-frame disturbance estimate."""
        return self._estimate.copy()

    def reset(self, estimate: FloatArray | None = None) -> None:
        """Reset history and optionally set the initial estimate."""
        value = (
            np.zeros(3, dtype=np.float64)
            if estimate is None
            else as_vec3(estimate)
        )
        self._estimate = np.clip(
            value, -self._max_abs_torque, self._max_abs_torque
        ).copy()
        self._previous_omega = None

    def update(
        self,
        state: RotationState,
        applied_torque: FloatArray,
        dt: float,
    ) -> FloatArray:
        """Update from the torque applied since the previous observation."""
        omega = as_vec3(state.omega)
        control = as_vec3(applied_torque)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and nonnegative.")

        if self._previous_omega is None:
            self._previous_omega = omega.copy()
            return self.estimate
        if dt == 0.0:
            raise ValueError(
                "dt must be positive after the observer is initialized."
            )

        omega_mid = 0.5 * (self._previous_omega + omega)
        omega_dot = (omega - self._previous_omega) / dt
        raw_estimate = (
            self._inertia @ omega_dot
            + np.cross(omega_mid, self._inertia @ omega_mid)
            - control
        )
        raw_estimate = np.clip(
            raw_estimate, -self._max_abs_torque, self._max_abs_torque
        )

        alpha = -np.expm1(-dt / self._time_constant_s)
        self._estimate += alpha * (raw_estimate - self._estimate)
        self._estimate = np.clip(
            self._estimate, -self._max_abs_torque, self._max_abs_torque
        )
        self._previous_omega = omega.copy()
        return self.estimate
