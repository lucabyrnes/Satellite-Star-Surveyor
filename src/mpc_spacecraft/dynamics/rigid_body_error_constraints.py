"""Rigid-body error-dynamics linearization utilities for MPC.

This module owns the Lie-error and affine error-step assembly used by MPC,
separating controller-facing constraint construction from plant integration.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
import quaternion as qu

from mpc_spacecraft.utilities.utils import (
    BODY_FORWARD_VEC3,
    FloatArray,
    RotationState,
    expm_so3,
    jacob_r_lie,
    jacob_r_lie_inv,
    logm_so3,
    skew,
    unit,
    z3,
)

StateStepFn = Callable[[RotationState, FloatArray], RotationState]
DiscretizeFn = Callable[[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]


class RigidBodyErrorConstraintBuilder:
    """Build affine error-dynamics constraints for rigid-body MPC."""

    def __init__(self, inertia: FloatArray, dt: float):
        self.inertia = inertia
        self.inertia_inv = cast(FloatArray, np.linalg.inv(inertia))
        self.dt = dt

    def discrete_phi_lie_shifting_constraint(
        self,
        x_nom_k: RotationState,
        x_nom_kp1: RotationState,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Linearized Lie-angle shifting constraint for one step."""
        q_nom_k = qu.quaternion(*x_nom_k.quat)
        q_nom_kp1 = qu.quaternion(*x_nom_kp1.quat)
        omega_nom_k = x_nom_k.omega
        omega_nom_k_angle = omega_nom_k * self.dt

        R_nom_k = qu.as_rotation_matrix(q_nom_k)
        R_nom_kp1 = qu.as_rotation_matrix(q_nom_kp1)
        d_R_centers = R_nom_kp1.T @ R_nom_k

        R_w_nom_k = expm_so3(omega_nom_k_angle)
        s_k = logm_so3(d_R_centers @ R_w_nom_k)

        j_w_k = jacob_r_lie(omega_nom_k_angle)
        j_m1_s = jacob_r_lie_inv(s_k)

        A_phi = j_m1_s @ R_w_nom_k.T
        B_omega = j_m1_s @ j_w_k * self.dt

        return A_phi, B_omega, s_k

    def discrete_mpc_constraint(
        self,
        x_nom_k: RotationState,
        x_nom_kp1: RotationState,
        u_nom_k: FloatArray,
        *,
        discrete_dynamics_step: StateStepFn,
        discretize_linear_system: DiscretizeFn,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Build affine error dynamics step: dx(k+1) = A dx(k) + B du(k) + c."""
        omega = x_nom_k.omega
        skew_w = skew(omega)
        skew_jw = skew(self.inertia @ omega)

        B_omega_c = self.inertia_inv @ (skew_jw - skew_w @ self.inertia)
        D_omega_c = np.block([self.inertia_inv])

        B_omega, D_omega = discretize_linear_system(B_omega_c, D_omega_c)

        x_pred_kp1 = discrete_dynamics_step(x_nom_k, u_nom_k)
        c_omega = x_pred_kp1.omega - x_nom_kp1.omega

        A_phi, B_phi, c_phi = self.discrete_phi_lie_shifting_constraint(
            x_nom_k, x_nom_kp1
        )

        A_k = np.block([[A_phi, B_phi], [z3, B_omega]])
        B_k = np.block([[z3], [D_omega]])
        c_k = np.hstack((c_phi, c_omega))

        return A_k, B_k, c_k

    def affine_error_theta_bc(
        self,
        x_nom_k: RotationState,
        boundary_direction_I: FloatArray,
        margin_rad: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Linearize an inertial-direction keep-out cone about an attitude.

        The outputs should be used in a linear inequality of the form

        ``prog.AddLinearConstraint(2.0 * alpha_k.dot(theta_k) - s[k] <= b_k)``

        where ``s[k]`` is an optional slack variable.

        Args:
            x_nom_k: Nominal rotational state at time step ``k``.
            boundary_direction_I: Inertial unit vector at the center of the
                excluded cone.
            margin_rad: Excluded cone half-angle in radians.

        Returns:
            Tuple ``(alpha_k, b_k)`` where ``alpha_k`` multiplies
            ``theta_k`` and ``b_k`` is the right-hand-side bound.
        """
        q_k = qu.quaternion(*x_nom_k.quat).normalized()
        R_k = qu.as_rotation_matrix(q_k)
        boundary_direction_B = R_k.T @ unit(boundary_direction_I)

        # ErrorStateMappingService uses the right perturbation
        # R(theta) = R_k Exp(theta). Linearizing the keep-out condition
        # boundary_direction_I.T @ R(theta) @ BODY_FORWARD_VEC3 <= cos(margin)
        # gives (BODY_FORWARD_VEC3 x boundary_direction_B).T theta <= b_k.
        alpha_k = 0.5 * np.cross(BODY_FORWARD_VEC3, boundary_direction_B)
        b_k = np.asarray(
            np.cos(margin_rad)
            - np.dot(boundary_direction_B, BODY_FORWARD_VEC3)
        )
        return alpha_k, b_k
