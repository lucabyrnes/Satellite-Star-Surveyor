"""Controller-facing prediction model contracts for MPC.

This module introduces a small, stable interface layer so controllers can
depend on *what they need* (error mapping + affine error dynamics) instead of a
concrete dynamics implementation.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from mpc_spacecraft.guidance.sun_tracker import TimeLike
from mpc_spacecraft.utilities.utils import FloatArray, RotationState, RotationErrorState
from mpc_spacecraft.utilities.array_view_generic import BatchArrayView


@dataclass(frozen=True)
class AffineErrorDynamicsStep:
    """Affine error dynamics for one horizon step.

    Represents:
        delta_x_{k+1} = A @ delta_x_k + B @ delta_u_k + c
    """

    A: FloatArray
    B: FloatArray
    c: FloatArray


@runtime_checkable
class ErrorDynamicsProvider(Protocol):
    """Port interface required by MPC formulations in this package."""

    def state_error(self, state: RotationState, state_ref: RotationState) -> RotationErrorState:
        """Map full states to 6D error coordinates."""
        ...

    def state_error_batch(
        self,
        states: BatchArrayView[RotationState],
        states_ref: BatchArrayView[RotationState],
    ) -> BatchArrayView[RotationErrorState]:
        """Vectorized variant of :meth:`state_error`."""
        ...

    def state_from_error(self, delta_x: RotationErrorState, state_ref: RotationState) -> RotationState:
        """Reconstruct full state from 6D error coordinates and reference."""
        ...

    def state_from_error_batch(
        self,
        delta_xs: BatchArrayView[RotationErrorState],
        states_ref: BatchArrayView[RotationState],
    ) -> BatchArrayView[RotationState]:
        """Vectorized variant of :meth:`state_from_error`."""
        ...

    def affine_error_dynamics_step(
        self,
        x_nom_k: RotationState,
        x_nom_kp1: RotationState,
        u_nom_k: FloatArray,
    ) -> AffineErrorDynamicsStep:
        """Return affine dynamics about the nominal total modeled torque."""
        ...

    def affine_error_theta_bc(
        self, x_nom_k: RotationState, current_epoch_utc: TimeLike
    ) -> tuple[FloatArray, FloatArray]:
        """
        The outputs should be used in a linear inequality of the form

        ``prog.AddLinearConstraint(2.0 * alpha_k.dot(theta_k) - s[k] <= b_k)``

        where ``s[k]`` is an optional slack variable.
        """
        ...
