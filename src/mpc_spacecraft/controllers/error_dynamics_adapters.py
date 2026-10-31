from typing import cast

import numpy as np

from mpc_spacecraft.utilities.utils import (
    FloatArray,
    RotationState,
    RotationErrorState,
)
from mpc_spacecraft.utilities.array_view_generic import BatchArrayView
from mpc_spacecraft.guidance.sun_tracker import AstropySunDirectionModel, TimeLike
from ..dynamics.rigid_body import SpacecraftDynamics
from ..dynamics.rigid_body_error_constraints import RigidBodyErrorConstraintBuilder
from .error_state_mapping import ErrorStateMappingService
from .error_dynamics_providers import AffineErrorDynamicsStep, ErrorDynamicsProvider


class SpacecraftErrorDynamicsProvider(ErrorDynamicsProvider):
    """Adapter that exposes `SpacecraftDynamics` through `MPCPredictionModel`.

    This intentionally delegates to existing methods in
    `SpacecraftDynamics` so behavior remains unchanged while we decouple
    controller code from concrete plant implementations.
    """

    def __init__(
        self,
        dynamics: SpacecraftDynamics,
        sun_model: AstropySunDirectionModel,
        sun_bc_margin_rad: float,
        mpc_dt: float,
    ):
        self.dynamics = dynamics
        self._sun_model = sun_model
        self._sun_margin = sun_bc_margin_rad
        self._mpc_dt = float(mpc_dt)
        self._error_mapping = ErrorStateMappingService()
        self._constraint_builder = RigidBodyErrorConstraintBuilder(
            inertia=dynamics.inertia,
            dt=self._mpc_dt,
        )

    def state_error(self, state: RotationState, state_ref: RotationState) -> RotationErrorState:
        return self._error_mapping.state_error(state, state_ref)

    def state_error_batch(
        self,
        states: BatchArrayView[RotationState],
        states_ref: BatchArrayView[RotationState],
    ) -> BatchArrayView[RotationErrorState]:
        return self._error_mapping.state_error_batch(states, states_ref)

    def state_from_error(self, delta_x: RotationErrorState, state_ref: RotationState) -> RotationState:
        return self._error_mapping.state_from_error(delta_x, state_ref)

    def state_from_error_batch(
        self,
        delta_xs: BatchArrayView[RotationErrorState],
        states_ref: BatchArrayView[RotationState],
    ) -> BatchArrayView[RotationState]:
        return self._error_mapping.state_from_error_batch(delta_xs, states_ref)

    def affine_error_dynamics_step(
        self,
        x_nom_k: RotationState,
        x_nom_kp1: RotationState,
        u_nom_k: FloatArray,
    ) -> AffineErrorDynamicsStep:
        A_k, B_k, c_k = self._constraint_builder.discrete_mpc_constraint(
            x_nom_k=x_nom_k,
            x_nom_kp1=x_nom_kp1,
            u_nom_k=u_nom_k,
            discrete_dynamics_step=lambda state, control: self.dynamics.discrete_dynamics_rk4_rotation(
                state, control, self._mpc_dt
            ),
            discretize_linear_system=lambda A, B: self.dynamics.discretize_linear_system(
                A, B, self._mpc_dt
            ),
        )
        return AffineErrorDynamicsStep(
            A=cast(FloatArray, A_k),
            B=cast(FloatArray, B_k),
            c=cast(FloatArray, c_k),
        )

    def affine_error_theta_bc(
        self, x_nom_k: RotationState, current_epoch_utc: TimeLike
    ) -> tuple[FloatArray, FloatArray]:
        sun_vec = self._sun_model.sun_dir_eci(current_epoch_utc)
        return self._constraint_builder.affine_error_theta_bc(
            x_nom_k, sun_vec, self._sun_margin
        )
