"""Rigid-body spacecraft dynamics with rotational and translational branches."""
from mpc_spacecraft.dynamics.disturbances import DisturbanceModel
from typing import cast

import numpy as np
import quaternion as qu
from scipy.linalg import expm

from mpc_spacecraft.utilities.utils import (
    FloatArray,
    RigidBodyControl,
    RigidBodyState,
    RotationState,
    TranslationState,
    Vec3,
)

# @TODO: Fix quaternion Integrator. RK4 does not natively work well with RK4, Use Lie algebra methods.

class SpacecraftDynamics:
    """
    Spacecraft dynamics with separate rotation and translation integrations.

    Rotation state vector: [q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z]
    Translation state vector: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]

    Rotation input: [tau_x, tau_y, tau_z]
    Translation input: [acc_x, acc_y, acc_z]
    """

    def __init__(
        self,
        inertia: FloatArray,
        disturbance: DisturbanceModel,
        mass: float = 1.0,
    ):
        """
        Initialize spacecraft dynamics.

        Args:
            inertia: 3x3 inertia matrix (kg*m^2).
            disturbance: External disturbance model.
            mass: Spacecraft mass (kg) for translational branch.
        """
        self._inertia = inertia
        self._inertia_inv = cast(FloatArray, np.linalg.inv(inertia))
        self._mass = mass
        self._disturbance_model = disturbance

    @property
    def inertia(self) -> FloatArray:
        return self._inertia

    @property
    def inertia_inv(self) -> FloatArray:
        return self._inertia_inv

    @property
    def mass(self) -> float:
        return self._mass

    def set_inertia(self, inertia: FloatArray) -> None:
        """Set inertia and keep inverse inertia synchronized."""
        self._inertia = inertia
        self._inertia_inv = cast(FloatArray, np.linalg.inv(inertia))

    def _continuous_dynamics_rotation(
        self,
        state: RotationState,
        torque: Vec3
    ) -> RotationState:
        """
        Compute continuous-time dynamics: dx/dt = f(x, u).

        Args:
            state: State vector [q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z]
            torque: torque input [tau_x, tau_y, tau_z]

        Returns:
            State derivative [dq/dt, d_omega/dt]
        """
        # Extract state components
        q = qu.quaternion(*state.quat)
        omega = state.omega

        # Normalize quaternion
        q = q.normalized()

        # Quaternion kinematics: dq/dt = 0.5 * q * omega_body
        q_dot = 0.5 * q * qu.from_vector_part(omega)

        # Euler's equation: d_omega/dt = I^-1 @ (tau - omega x (I @ omega))
        omega_dot = self._inertia_inv @ (
            torque - np.cross(omega, self._inertia @ omega)
        )

        x_dot = RotationState.zeros()
        x_dot.quat[:] = qu.as_float_array(q_dot)
        x_dot.omega[:] = omega_dot
        return x_dot

    def _continuous_dynamics_translation(
        self,
        state: TranslationState,
        acceleration: Vec3,
    ) -> TranslationState:
        state_dt = TranslationState.zeros()
        state_dt.position[:] = state.velocity.copy()
        state_dt.velocity[:] = (
            acceleration + self._disturbance_model.get_acceleration(state.position)
        )
        return state_dt

    # def rotational_dynamics_error_jacobian(
    #     self,
    #     state: RotState,
    #     input: FloatArray,
    # ) -> tuple[FloatArray, FloatArray]:

    #     omega = state[IDX_STATE_OMEGA]
    #     skew_w = skew(omega)
    #     skew_jw = skew(self._inertia_inv @ omega)

    #     dw_block = self._inertia_inv @ (skew_jw - skew_w @ self._inertia)

    #     A = np.block([[-skew_w, I3], [z3, dw_block]])

    #     B = np.block([[z3], [self._inertia_inv]])

    #     return A, B

    def discrete_dynamics_rk4_rotation(
        self,
        state: RotationState,
        torque: Vec3,
        delta_time: float,
    ) -> RotationState:
        """
        Compute discrete-time dynamics using RK4 integration.

        Args:
            state: Current state
            control: Control input
            disturbance: Optional disturbance torque

        Returns:
            Next state
        """
        dt = float(delta_time)
        # RK4 integration
        k1 = self._continuous_dynamics_rotation(state, torque)
        k2 = self._continuous_dynamics_rotation(
            RotationState.from_array(state.data + 0.5 * dt * k1.data), torque
        )
        k3 = self._continuous_dynamics_rotation(
            RotationState.from_array(state.data + 0.5 * dt * k2.data), torque
        )
        k4 = self._continuous_dynamics_rotation(
            RotationState.from_array(state.data + dt * k3.data), torque
        )

        next_state = RotationState.from_array(
            state.data
            + (dt / 6.0)
            * (k1.data + 2 * k2.data + 2 * k3.data + k4.data)
        )

        # Normalize quaternion
        q = qu.quaternion(*next_state.quat).normalized()
        next_state.quat[:] = qu.as_float_array(q)

        return next_state

    def discrete_dynamics_rk4_translation(
        self,
        state: TranslationState,
        acceleration: Vec3,
        delta_time: float,
    ) -> TranslationState:
        """Compute translational discrete-time dynamics using RK4 integration."""
        dt = float(delta_time)
        k1 = self._continuous_dynamics_translation(state, acceleration)
        k2 = self._continuous_dynamics_translation(
            TranslationState.from_array(state.data + 0.5 * dt * k1.data), acceleration
        )
        k3 = self._continuous_dynamics_translation(
            TranslationState.from_array(state.data + 0.5 * dt * k2.data), acceleration
        )
        k4 = self._continuous_dynamics_translation(
            TranslationState.from_array(state.data + dt * k3.data), acceleration
        )

        return TranslationState.from_array(
            state.data
            + (dt / 6.0)
            * (k1.data + 2 * k2.data + 2 * k3.data + k4.data)
        )

    def discretize_dynamics_rk4_full_state(
        self,
        state: RigidBodyState,
        inputs: RigidBodyControl,
        interval_start_time: float,
        delta_time: float,
    ) -> RigidBodyState:
        trans_state = state.translation
        trans_control = inputs.acceleration

        rot_state = state.rotation
        rot_control = inputs.torque

        disturbance_torque = self._disturbance_model.get_torque(interval_start_time)
        rot_torque = rot_control + disturbance_torque

        next_trans_state = self.discrete_dynamics_rk4_translation(
            trans_state, trans_control, delta_time
        )
        next_rot_state = self.discrete_dynamics_rk4_rotation(
            rot_state, rot_torque, delta_time
        )

        next_state = RigidBodyState.zeros()
        next_state.translation.data[:] = next_trans_state.data
        next_state.rotation.data[:] = next_rot_state.data
        return next_state

    def discretize_linear_system(
        self, A: FloatArray, B: FloatArray, delta_time: float
    ) -> tuple[FloatArray, FloatArray]:
        """
        Discretize linear system using matrix exponential.

        Args:
            A: Continuous-time state matrix
            B: Continuous-time input matrix

        Returns:
            Ad: Discrete-time state matrix
            Bd: Discrete-time input matrix
        """

        n = A.shape[0]
        m = B.shape[1]

        # Build augmented matrix
        dt = float(delta_time)
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A * dt
        M[:n, n:] = B * dt

        # Matrix exponential
        exp_M = expm(M)

        Ad = cast(FloatArray, exp_M[:n, :n])
        Bd = cast(FloatArray, exp_M[:n, n:])

        # Supposedly More Robust Solution
        # I = np.eye(6)
        # Ad = np.linalg.solve(I - 0.5 * dt * A, I + 0.5 * dt * A)
        # Bd = np.linalg.solve(I - 0.5 * dt * A, dt * B)

        return Ad, Bd
