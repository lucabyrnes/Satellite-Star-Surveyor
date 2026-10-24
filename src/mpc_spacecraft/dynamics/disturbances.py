"""Disturbance models for spacecraft dynamics."""

import numpy as np
from typing import cast

from numpy.linalg import norm

from mpc_spacecraft.utilities.utils import (
    FloatArray,
    G_CONST,
    M_EARTH,
    RigidBodyControl,
    RigidBodyState,
    Vec3,
)


class DisturbanceModel:
    """
    Model for external disturbances acting on the spacecraft.

    Supports various disturbance types:
    - Gravity acceleration
    - Constant bias torque
    - Random noise torque
    - Sinusoidal torque disturbances
    - Atmospheric drag (simplified)
    """

    def __init__(
        self,
        bias: np.ndarray | None = None,
        noise_std: float | None = None,
        sinusoidal_amplitude: float | None = None,
        sinusoidal_frequency: float | None = None,
        seed: int | None = None,
        enable_gravity: bool = True,
    ):
        """
        Initialize disturbance model.

        Args:
            bias: Constant bias torque [tau_x, tau_y, tau_z] (N*m). Enables
                bias torque when provided and nonzero.
            noise_std: Standard deviation of random noise torque (N*m). Enables
                noise torque when provided and nonzero.
            sinusoidal_amplitude: Amplitude of sinusoidal torque disturbance
                (N*m). Enables sinusoidal torque only when both sinusoidal
                parameters are provided and nonzero.
            sinusoidal_frequency: Frequency of sinusoidal torque disturbance
                (Hz). Enables sinusoidal torque only when both sinusoidal
                parameters are provided and nonzero.
            seed: Random seed for reproducibility.
            enable_gravity: Whether to include central-gravity acceleration.
                Enabled by default.
        """
        self.bias = np.asarray(bias, dtype=np.float64) if bias is not None else np.zeros(3)
        if self.bias.shape != (3,):
            raise ValueError(f"Expected bias shape (3,), got {self.bias.shape}.")
        self.noise_std = float(noise_std) if noise_std is not None else 0.0
        self.sinusoidal_amplitude = (
            float(sinusoidal_amplitude) if sinusoidal_amplitude is not None else 0.0
        )
        self.sinusoidal_frequency = (
            float(sinusoidal_frequency) if sinusoidal_frequency is not None else 0.0
        )
        self.enable_gravity = enable_gravity
        self.enable_bias = bias is not None and not np.allclose(self.bias, 0.0)
        self.enable_noise = noise_std is not None and self.noise_std != 0.0
        self.enable_sinusoidal = (
            sinusoidal_amplitude is not None
            and sinusoidal_frequency is not None
            and self.sinusoidal_amplitude != 0.0
            and self.sinusoidal_frequency != 0.0
        )

        if seed is not None:
            np.random.seed(seed)

    def get_disturbance(self, time: float, state: RigidBodyState) -> RigidBodyControl:
        """
        Compute total full-body disturbance at given time and state.

        Args:
            time: Current simulation time (s).
            state: Full body state with inertial position in the first 3 entries.

        Returns:
            Full-body disturbance vector with translational acceleration in the
            first 3 entries (m/s^2) and rotational torque in the last 3 entries
            (N*m).
        """
        disturbance = RigidBodyControl.zeros()

        if self.enable_gravity:
            disturbance.acceleration[:] = gravity(state.position)

        torque_disturbance = np.zeros(3, dtype=np.float64)

        if self.enable_bias:
            torque_disturbance += self.bias

        # Add random noise
        if self.enable_noise:
            torque_disturbance += np.random.normal(0, self.noise_std, 3)

        # Add sinusoidal component
        if self.enable_sinusoidal:
            phase = 2.0 * np.pi * self.sinusoidal_frequency * time
            sinusoidal = self.sinusoidal_amplitude * np.array(
                [
                    np.sin(phase),
                    np.sin(phase + 2 * np.pi / 3),
                    np.sin(phase + 4 * np.pi / 3),
                ],
                dtype=np.float64,
            )
            torque_disturbance += sinusoidal

        disturbance.torque[:] = torque_disturbance

        return disturbance


def get_atmospheric_drag(
    velocity: FloatArray,
    altitude: float,
    drag_coefficient: float = 2.2,
    reference_area: float = 1.0,
) -> FloatArray:
    """
    Compute simplified atmospheric drag torque.

    Args:
        velocity: Velocity vector in body frame (m/s)
        altitude: Altitude above surface (m)
        drag_coefficient: Drag coefficient (dimensionless)
        reference_area: Reference area (m^2)

    Returns:
        Drag torque [tau_d_x, tau_d_y, tau_d_z] (N*m)
    """
    # Simplified atmospheric density model (exponential)
    rho_0 = 1.225  # kg/m^3 at sea level
    H = 8500  # Scale height (m)
    rho = rho_0 * np.exp(-altitude / H)

    # Drag force
    v_mag = np.linalg.norm(velocity)
    if v_mag < 1e-6:
        return np.zeros(3)

    drag_force = 0.5 * rho * drag_coefficient * reference_area * v_mag**2

    # Simplified torque (assuming center of pressure offset)
    # In reality, this would depend on spacecraft geometry
    lever_arm = 0.1  # m
    drag_torque = lever_arm * drag_force * (velocity / v_mag)

    return drag_torque


def jacobian_gravity(pos: Vec3) -> FloatArray:
    """
    Compute the Jacobian of the central-gravity acceleration w.r.t. position.

    Args:
        pos: Inertial position vector (m).

    Returns:
        3x3 Jacobian matrix of gravitational acceleration (1/s^2).
    """
    r_vec = np.asarray(pos, dtype=np.float64) - np.zeros(3, dtype=np.float64)
    return cast(FloatArray, -(G_CONST * M_EARTH) * (
        (1 / norm(r_vec) ** 3) * np.eye(3)
        - (3 / norm(r_vec) ** 5) * np.outer(r_vec, r_vec)
    ))


def gravity(pos: Vec3) -> Vec3:
    """
    Compute central-gravity acceleration at a position.

    Args:
        pos: Inertial position vector (m).

    Returns:
        Gravitational acceleration vector (m/s^2).
    """
    r_vec = np.asarray(pos, dtype=np.float64) - np.zeros(3, dtype=np.float64)
    return cast(Vec3, -((G_CONST * M_EARTH) / (norm(r_vec) ** 3)) * r_vec)
