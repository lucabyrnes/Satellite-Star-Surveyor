"""Tests for constant external-torque estimation."""

import numpy as np
import pytest

from mpc_spacecraft.estimation import ConstantTorqueObserver
from mpc_spacecraft.utilities.utils import RotationState


def _state(omega: np.ndarray) -> RotationState:
    state = RotationState.zeros()
    state.omega[:] = omega
    return state


def test_observer_recovers_noiseless_single_axis_torque() -> None:
    observer = ConstantTorqueObserver(
        inertia=np.diag([2.0, 3.0, 4.0]),
        time_constant_s=0.01,
        max_abs_torque=np.ones(3),
    )

    np.testing.assert_allclose(
        observer.update(_state(np.zeros(3)), np.zeros(3), 0.0),
        np.zeros(3),
    )
    estimate = observer.update(
        _state(np.array([0.1, 0.0, 0.0])),
        np.zeros(3),
        1.0,
    )

    np.testing.assert_allclose(estimate, [0.2, 0.0, 0.0], atol=1e-12)


def test_observer_accounts_for_control_and_gyroscopic_torque() -> None:
    inertia = np.diag([2.0, 3.0, 4.0])
    observer = ConstantTorqueObserver(
        inertia=inertia,
        time_constant_s=0.01,
        max_abs_torque=10.0 * np.ones(3),
    )
    omega = np.array([0.2, -0.1, 0.3])
    applied = np.array([0.4, -0.2, 0.1])
    disturbance = np.array([0.05, 0.03, -0.02])
    omega_dot = np.linalg.solve(
        inertia,
        applied + disturbance - np.cross(omega, inertia @ omega),
    )

    observer.update(_state(omega - 0.5 * omega_dot), applied, 0.0)
    estimate = observer.update(_state(omega + 0.5 * omega_dot), applied, 1.0)

    np.testing.assert_allclose(estimate, disturbance, atol=1e-12)


def test_observer_filters_and_clamps_raw_estimate() -> None:
    observer = ConstantTorqueObserver(
        inertia=np.eye(3),
        time_constant_s=1.0,
        max_abs_torque=np.array([0.1, 0.2, 0.3]),
    )
    observer.update(_state(np.zeros(3)), np.zeros(3), 0.0)

    estimate = observer.update(_state(10.0 * np.ones(3)), np.zeros(3), 1.0)
    alpha = 1.0 - np.exp(-1.0)
    np.testing.assert_allclose(
        estimate,
        alpha * np.array([0.1, 0.2, 0.3]),
    )


def test_observer_reset_and_estimate_copy() -> None:
    observer = ConstantTorqueObserver(
        inertia=np.eye(3),
        time_constant_s=1.0,
        max_abs_torque=np.ones(3),
        initial_torque=np.array([0.1, 0.2, 0.3]),
    )
    returned = observer.estimate
    returned[:] = 99.0
    np.testing.assert_allclose(observer.estimate, [0.1, 0.2, 0.3])

    observer.update(_state(np.ones(3)), np.zeros(3), 0.0)
    observer.reset(np.array([-0.1, -0.2, -0.3]))
    np.testing.assert_allclose(
        observer.update(_state(np.zeros(3)), np.zeros(3), 0.0),
        [-0.1, -0.2, -0.3],
    )


def test_observer_rejects_invalid_elapsed_time_after_initialization() -> None:
    observer = ConstantTorqueObserver(
        inertia=np.eye(3),
        time_constant_s=1.0,
        max_abs_torque=np.ones(3),
    )
    observer.update(_state(np.zeros(3)), np.zeros(3), 0.0)

    with pytest.raises(ValueError, match="positive"):
        observer.update(_state(np.zeros(3)), np.zeros(3), 0.0)
