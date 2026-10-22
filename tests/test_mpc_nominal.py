"""Tests for the current NominalMPC API."""

from datetime import datetime, timezone

import numpy as np
import pytest

from mpc_spacecraft.controllers.error_dynamics_adapters import (
    SpacecraftErrorDynamicsProvider,
)
from mpc_spacecraft.controllers.error_dynamics_providers import ErrorDynamicsProvider
from mpc_spacecraft.controllers.mpc_nominal_drake import NominalMPC
from mpc_spacecraft.dynamics.rigid_body import SpacecraftDynamics
from mpc_spacecraft.dynamics.disturbances import DisturbanceModel
from mpc_spacecraft.guidance.sun_tracker import AstropySunDirectionModel
from mpc_spacecraft.utilities.utils import RotationState


@pytest.fixture
def inertia() -> np.ndarray:
    return np.array(
        [[90.0, -5.0, 3.0], [-5.0, 110.0, -4.0], [3.0, -4.0, 100.0]], dtype=float
    )


@pytest.fixture
def disturbance() -> DisturbanceModel:
    return DisturbanceModel()


@pytest.fixture
def dynamics(inertia: np.ndarray, disturbance: DisturbanceModel) -> SpacecraftDynamics:
    return SpacecraftDynamics(inertia, disturbance)


@pytest.fixture
def sun_model() -> AstropySunDirectionModel:
    return AstropySunDirectionModel()


@pytest.fixture
def error_dynamics_provider(
    dynamics: SpacecraftDynamics, sun_model: AstropySunDirectionModel
) -> ErrorDynamicsProvider:
    return SpacecraftErrorDynamicsProvider(dynamics, sun_model, 0.1, 0.1)


@pytest.fixture
def x0() -> RotationState:
    state = RotationState.zeros()
    state.quat[:] = np.array([1.0, 0.0, 0.0, 0.0])
    state.omega[:] = np.array([0.01, -0.02, 0.015])
    return state


@pytest.fixture
def x_goal() -> RotationState:
    state = RotationState.zeros()
    state.quat[:] = np.array([0.70710678, 0.0, 0.70710678, 0.0])
    state.omega[:] = np.zeros(3)
    return state


@pytest.fixture
def Q() -> np.ndarray:
    return np.diag([10.0, 10.0, 10.0, 3.0, 3.0, 3.0])


@pytest.fixture
def R() -> np.ndarray:
    return np.diag([0.1, 0.1, 0.1])


def test_nominal_mpc_solve_uses_current_api(
    error_dynamics_provider: ErrorDynamicsProvider,
    x0: RotationState,
    x_goal: RotationState,
    Q: np.ndarray,
    R: np.ndarray,
):
    mpc = NominalMPC(
        horizon=3,
        error_dynamics_provider=error_dynamics_provider,
        Q=Q,
        R=R,
        u_min=-5.0 * np.ones(3),
        u_max=5.0 * np.ones(3),
        max_sqp_iters=1,
    )

    current_epoch_utc = datetime(2026, 1, 1, tzinfo=timezone.utc)
    u_opt, x_opt, success = mpc._solve(
        x0,
        x_goal=x_goal,
        current_epoch_utc=current_epoch_utc,
    )

    assert success is True
    assert u_opt.shape == (3, 3)
    assert x_opt.data.shape == (4, 7)
    np.testing.assert_allclose(x_opt[0].quat, x0.quat, atol=1e-12)
    np.testing.assert_allclose(x_opt[0].omega, x0.omega, atol=1e-12)


@pytest.mark.filterwarnings("error:invalid value encountered:RuntimeWarning")
def test_nominal_mpc_get_first_control_accepts_current_epoch(
    error_dynamics_provider: ErrorDynamicsProvider,
    x0: RotationState,
    x_goal: RotationState,
    Q: np.ndarray,
    R: np.ndarray,
):
    mpc = NominalMPC(
        horizon=2,
        error_dynamics_provider=error_dynamics_provider,
        Q=Q,
        R=R,
        u_min=-5.0 * np.ones(3),
        u_max=5.0 * np.ones(3),
        max_sqp_iters=1,
    )

    current_epoch_utc = datetime(2026, 1, 1, tzinfo=timezone.utc)
    u0 = mpc.get_first_control(x0, x_goal=x_goal, current_epoch_utc=current_epoch_utc)

    assert u0.shape == (3,)
    assert np.isfinite(u0).all()


def test_nominal_mpc_rejects_missing_reference(
    error_dynamics_provider: ErrorDynamicsProvider,
    x0: RotationState,
    Q: np.ndarray,
    R: np.ndarray,
):
    mpc = NominalMPC(
        horizon=2,
        error_dynamics_provider=error_dynamics_provider,
        Q=Q,
        R=R,
    )

    with pytest.raises(ValueError, match="Need to provide either trajectory or terminal goal"):
        mpc._solve(x0, current_epoch_utc=datetime(2026, 1, 1, tzinfo=timezone.utc))


def test_nominal_mpc_accepts_reference_trajectory(
    error_dynamics_provider: ErrorDynamicsProvider,
    x0: RotationState,
    Q: np.ndarray,
    R: np.ndarray,
):
    mpc = NominalMPC(
        horizon=2,
        error_dynamics_provider=error_dynamics_provider,
        Q=Q,
        R=R,
    )

    x_ref = RotationState.batch_zeros(3)
    x_ref.data[:] = np.vstack([x0.data, x0.data, x0.data])
    u_ref = np.zeros((2, 3))

    u_opt, x_opt, success = mpc._solve(
        x0,
        x_ref=x_ref,
        u_ref=u_ref,
        current_epoch_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert success is True
    assert u_opt.shape == (2, 3)
    assert x_opt.data.shape == (3, 7)


def test_nominal_mpc_compensates_control_reference_for_disturbance(
    error_dynamics_provider: ErrorDynamicsProvider,
    x0: RotationState,
    x_goal: RotationState,
    Q: np.ndarray,
    R: np.ndarray,
):
    mpc = NominalMPC(
        horizon=2,
        error_dynamics_provider=error_dynamics_provider,
        Q=Q,
        R=R,
        u_min=-0.25 * np.ones(3),
        u_max=0.25 * np.ones(3),
    )
    disturbance = np.array([0.03, -0.02, 0.01])

    _, u_cost = mpc._build_ref_trajectory(
        x0,
        x_goal=x_goal,
        disturbance_torque=disturbance,
    )

    np.testing.assert_allclose(u_cost, np.tile(-disturbance, (2, 1)))


def test_nominal_mpc_cancels_feasible_disturbance_at_equilibrium(
    error_dynamics_provider: ErrorDynamicsProvider,
    Q: np.ndarray,
    R: np.ndarray,
):
    equilibrium = RotationState.zeros()
    disturbance = np.array([0.03, -0.02, 0.01])
    mpc = NominalMPC(
        horizon=3,
        error_dynamics_provider=error_dynamics_provider,
        Q=Q,
        R=R,
        u_min=-0.25 * np.ones(3),
        u_max=0.25 * np.ones(3),
    )

    control = mpc.get_first_control(
        equilibrium,
        x_goal=equilibrium,
        disturbance_torque=disturbance,
        current_epoch_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    np.testing.assert_allclose(control, -disturbance, atol=1e-9)


def test_nominal_mpc_compensates_feedforward_reference_and_clips_to_bounds(
    error_dynamics_provider: ErrorDynamicsProvider,
    x0: RotationState,
    Q: np.ndarray,
    R: np.ndarray,
):
    mpc = NominalMPC(
        horizon=2,
        error_dynamics_provider=error_dynamics_provider,
        Q=Q,
        R=R,
        u_min=-0.1 * np.ones(3),
        u_max=0.1 * np.ones(3),
    )
    x_ref = RotationState.batch_zeros(3)
    x_ref.data[:] = np.tile(x0.data, (3, 1))
    u_ref = np.tile(np.array([0.08, 0.0, -0.08]), (2, 1))

    _, u_cost = mpc._build_ref_trajectory(
        x0,
        x_ref=x_ref,
        u_ref=u_ref,
        disturbance_torque=np.array([-0.05, 0.02, 0.05]),
    )

    np.testing.assert_allclose(u_cost, np.tile([0.1, -0.02, -0.1], (2, 1)))
