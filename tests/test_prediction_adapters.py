from datetime import datetime, timezone

import numpy as np
import pytest
import quaternion as qu

from mpc_spacecraft.controllers.error_dynamics_adapters import SpacecraftErrorDynamicsProvider
from mpc_spacecraft.controllers.error_dynamics_providers import ErrorDynamicsProvider
from mpc_spacecraft.controllers.mpc_nominal_drake import NominalMPC
from mpc_spacecraft.dynamics.disturbances import DisturbanceModel
from mpc_spacecraft.dynamics.rigid_body import SpacecraftDynamics
from mpc_spacecraft.dynamics.rigid_body_error_constraints import RigidBodyErrorConstraintBuilder
from mpc_spacecraft.guidance.sun_tracker import AstropySunDirectionModel
from mpc_spacecraft.utilities.utils import RotationState


@pytest.fixture
def inertia() -> np.ndarray:
    return np.array([[90.0, -5.0, 3.0], [-5.0, 110.0, -4.0], [3.0, -4.0, 100.0]])


@pytest.fixture
def dt() -> float:
    return 0.1


@pytest.fixture
def dynamics(inertia: np.ndarray, dt: float) -> SpacecraftDynamics:
    return SpacecraftDynamics(inertia, DisturbanceModel())


@pytest.fixture
def sun_model() -> AstropySunDirectionModel:
    return AstropySunDirectionModel()


@pytest.fixture
def sun_bc_margin_rad() -> float:
    return 0.1


@pytest.fixture
def ref_state() -> RotationState:
    return RotationState.zeros()


@pytest.fixture
def goal_state() -> RotationState:
    q_goal = qu.quaternion(0.707, 0.0, 0.707, 0.0).normalized()
    state = RotationState.zeros()
    state.quat[:] = qu.as_float_array(q_goal)
    return state


@pytest.fixture
def Q() -> np.ndarray:
    return np.diag([10.0, 10.0, 10.0, 3.0, 3.0, 3.0])


@pytest.fixture
def R() -> np.ndarray:
    return np.diag([0.1, 0.1, 0.1])


@pytest.mark.unit
def test_adapter_satisfies_prediction_protocol(dynamics, sun_model, sun_bc_margin_rad):
    adapter = SpacecraftErrorDynamicsProvider(dynamics, sun_model, sun_bc_margin_rad, 0.1)
    assert isinstance(adapter, ErrorDynamicsProvider)


@pytest.mark.unit
def test_adapter_affine_step_matches_legacy_constraint(dynamics, sun_model, sun_bc_margin_rad):
    adapter = SpacecraftErrorDynamicsProvider(dynamics, sun_model, sun_bc_margin_rad, 0.1)
    builder = RigidBodyErrorConstraintBuilder(inertia=dynamics.inertia, dt=0.1)

    x_nom_k = RotationState.zeros()
    x_nom_k.omega[:] = np.array([0.02, -0.01, 0.015])
    u_nom_k = np.array([0.01, -0.02, 0.005])
    x_nom_kp1 = dynamics.discrete_dynamics_rk4_rotation(x_nom_k, u_nom_k, 0.1)

    A_ref, B_ref, c_ref = builder.discrete_mpc_constraint(
        x_nom_k=x_nom_k,
        x_nom_kp1=x_nom_kp1,
        u_nom_k=u_nom_k,
        discrete_dynamics_step=lambda s, u: dynamics.discrete_dynamics_rk4_rotation(s, u, 0.1),
        discretize_linear_system=lambda A, B: dynamics.discretize_linear_system(A, B, 0.1),
    )
    step = adapter.affine_error_dynamics_step(x_nom_k, x_nom_kp1, u_nom_k)

    np.testing.assert_allclose(step.A, A_ref)
    np.testing.assert_allclose(step.B, B_ref)
    np.testing.assert_allclose(step.c, c_ref)


@pytest.mark.unit
def test_error_constraint_builder_matches_legacy_constraint(dynamics, sun_model, sun_bc_margin_rad):
    builder = RigidBodyErrorConstraintBuilder(inertia=dynamics.inertia, dt=0.1)
    adapter = SpacecraftErrorDynamicsProvider(dynamics, sun_model, sun_bc_margin_rad, 0.1)

    x_nom_k = RotationState.zeros()
    x_nom_k.omega[:] = np.array([0.02, -0.01, 0.015])
    u_nom_k = np.array([0.01, -0.02, 0.005])
    x_nom_kp1 = dynamics.discrete_dynamics_rk4_rotation(x_nom_k, u_nom_k, 0.1)

    step = adapter.affine_error_dynamics_step(x_nom_k, x_nom_kp1, u_nom_k)
    A_new, B_new, c_new = builder.discrete_mpc_constraint(
        x_nom_k=x_nom_k,
        x_nom_kp1=x_nom_kp1,
        u_nom_k=u_nom_k,
        discrete_dynamics_step=lambda s, u: dynamics.discrete_dynamics_rk4_rotation(s, u, 0.1),
        discretize_linear_system=lambda A, B: dynamics.discretize_linear_system(A, B, 0.1),
    )

    np.testing.assert_allclose(A_new, step.A)
    np.testing.assert_allclose(B_new, step.B)
    np.testing.assert_allclose(c_new, step.c)


class _CountingPredictionAdapter(SpacecraftErrorDynamicsProvider):
    def __init__(self, dynamics, sun_model, sun_bc_margin_rad):
        super().__init__(dynamics, sun_model, sun_bc_margin_rad, 0.1)
        self.state_error_calls = 0
        self.state_error_batch_calls = 0
        self.state_from_error_calls = 0
        self.state_from_error_batch_calls = 0
        self.affine_step_calls = 0
        self.affine_step_torques = []

    def state_error(self, state, state_ref):
        self.state_error_calls += 1
        return super().state_error(state, state_ref)

    def state_error_batch(self, states, states_ref):
        self.state_error_batch_calls += 1
        return super().state_error_batch(states, states_ref)

    def state_from_error(self, delta_x, state_ref):
        self.state_from_error_calls += 1
        return super().state_from_error(delta_x, state_ref)

    def state_from_error_batch(self, delta_xs, states_ref):
        self.state_from_error_batch_calls += 1
        return super().state_from_error_batch(delta_xs, states_ref)

    def affine_error_dynamics_step(self, x_nom_k, x_nom_kp1, u_nom_k):
        self.affine_step_calls += 1
        self.affine_step_torques.append(u_nom_k.copy())
        return super().affine_error_dynamics_step(x_nom_k, x_nom_kp1, u_nom_k)


@pytest.mark.unit
def test_nominal_mpc_uses_injected_prediction_model(dynamics, sun_model, sun_bc_margin_rad, ref_state, goal_state, Q, R):
    adapter = _CountingPredictionAdapter(dynamics, sun_model, sun_bc_margin_rad)
    mpc = NominalMPC(
        horizon=4,
        error_dynamics_provider=adapter,
        Q=Q,
        R=R,
        u_min=-10.0 * np.ones(3),
        u_max=10.0 * np.ones(3),
        max_sqp_iters=1,
    )

    _, _, success = mpc._solve(ref_state, x_goal=goal_state, current_epoch_utc=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert success

    assert adapter.state_error_calls > 0
    assert adapter.state_error_batch_calls > 0
    assert adapter.state_from_error_batch_calls > 0
    assert adapter.state_from_error_calls == 0
    assert adapter.affine_step_calls == mpc.horizon


@pytest.mark.unit
def test_nominal_mpc_passes_compensated_total_torque_to_prediction_model(
    dynamics, sun_model, sun_bc_margin_rad, ref_state, Q, R
):
    adapter = _CountingPredictionAdapter(dynamics, sun_model, sun_bc_margin_rad)
    mpc = NominalMPC(
        horizon=2,
        error_dynamics_provider=adapter,
        Q=Q,
        R=R,
        u_min=-10.0 * np.ones(3),
        u_max=10.0 * np.ones(3),
    )
    disturbance = np.array([0.2, -0.1, 0.05])

    _, _, success = mpc._solve(
        ref_state,
        x_goal=ref_state,
        disturbance_torque=disturbance,
        current_epoch_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert success
    assert len(adapter.affine_step_torques) == mpc.horizon
    for total_torque in adapter.affine_step_torques:
        np.testing.assert_allclose(total_torque, np.zeros(3), atol=1e-12)
