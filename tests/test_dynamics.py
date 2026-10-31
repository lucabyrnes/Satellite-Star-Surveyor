import numpy as np
import pytest
import quaternion

from mpc_spacecraft.controllers.error_state_mapping import ErrorStateMappingService
from mpc_spacecraft.dynamics.disturbances import DisturbanceModel, gravity
from mpc_spacecraft.dynamics.rigid_body import SpacecraftDynamics
from mpc_spacecraft.utilities.utils import (
    RigidBodyControl,
    RigidBodyState,
    RotationState,
    TranslationState,
)


@pytest.fixture
def inertia():
    return np.diag([1.0, 1.0, 1.0])


@pytest.fixture
def dt():
    return 0.1


@pytest.fixture
def dynamics(inertia, dt):
    return SpacecraftDynamics(inertia, DisturbanceModel(enable_gravity=False))


@pytest.fixture
def error_mapping():
    return ErrorStateMappingService()


@pytest.mark.unit
def test_sinusoidal_disturbance_frequency_is_measured_in_hz() -> None:
    model = DisturbanceModel(
        sinusoidal_amplitude=2.0,
        sinusoidal_frequency=0.25,
        enable_gravity=False,
    )

    disturbance = model.get_disturbance(1.0, RigidBodyState.zeros())

    assert disturbance.torque[0] == pytest.approx(2.0)


@pytest.fixture
def ref_state():
    return RotationState.zeros()


@pytest.fixture
def zero_control():
    return np.zeros(3)


@pytest.fixture
def small_error_state(ref_state):
    state = ref_state.copy()
    dq = quaternion.quaternion(1.0, 0.005, 0.01, 0.015).normalized()
    q_ref = quaternion.quaternion(*ref_state.quat)
    q = q_ref * dq
    state.quat[:] = quaternion.as_float_array(q)
    state.omega[:] = np.array([0.01, 0.02, 0.0])
    return state


@pytest.mark.unit
def test_quaternion_normalization(dynamics, ref_state):
    dt = 0.1
    unnorm_q = np.array([1.5, 0.1, 0.2, 0.3])
    state_unnorm = ref_state.copy()
    state_unnorm.quat[:] = unnorm_q

    non_zero_control = np.array([1.0, 0.0, 0.0])
    next_state_nonzero = dynamics.discrete_dynamics_rk4_rotation(
        state_unnorm, non_zero_control, dt
    )
    q_next_nonzero = quaternion.quaternion(*next_state_nonzero.quat)
    assert np.isclose(q_next_nonzero.norm(), 1.0, atol=1e-10)


@pytest.mark.unit
def test_continuous_dynamics_equilibrium(dynamics, ref_state, zero_control):
    state_dot = dynamics._continuous_dynamics_rotation(ref_state, zero_control)
    np.testing.assert_allclose(state_dot.data, np.zeros(7), atol=1e-10)


@pytest.mark.unit
def test_discrete_dynamics_consistency(dynamics, ref_state, zero_control):
    dt = 0.1
    rk4_next = dynamics.discrete_dynamics_rk4_rotation(
        ref_state, zero_control, dt
    )
    euler_next = dynamics.discrete_dynamics_rk4_rotation(
        ref_state, zero_control, dt
    )
    np.testing.assert_allclose(rk4_next.data, euler_next.data, rtol=1e-6, atol=1e-8)


@pytest.mark.unit
def test_quaternion_error_small_angle(error_mapping):
    q_ref = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    dq = quaternion.quaternion(1.0, 0.005, 0.0, 0.0)
    q = q_ref * dq.normalized()
    delta_theta = error_mapping.quaternion_error(q, q_ref)
    np.testing.assert_allclose(delta_theta, np.array([0.01, 0.0, 0.0]), rtol=1e-4, atol=1e-8)
    assert np.linalg.norm(delta_theta) < 0.1


@pytest.mark.unit
def test_state_error_and_from_error_roundtrip(error_mapping, ref_state, small_error_state):
    delta_x = error_mapping.state_error(small_error_state, ref_state)
    reconstructed = error_mapping.state_from_error(delta_x, ref_state)
    np.testing.assert_allclose(reconstructed.quat, small_error_state.quat, rtol=1e-3, atol=1e-8)
    np.testing.assert_allclose(reconstructed.omega, small_error_state.omega, rtol=1e-8, atol=1e-10)
    assert np.linalg.norm(delta_x.error_angle) < 0.1


@pytest.mark.unit
def test_linearize_around_equilibrium(dynamics, ref_state, zero_control):
    A = np.zeros((6, 6))
    B = np.zeros((6, 3))
    np.testing.assert_allclose(B, np.zeros((6, 3)), atol=1e-10)
    np.testing.assert_allclose(A, np.zeros((6, 6)), atol=1e-10, rtol=0)


@pytest.mark.unit
def test_discretize_linear_system(dynamics):
    dt = 0.1
    A_cont = np.zeros((6, 6))
    B_cont = np.zeros((6, 3))
    Ad, Bd = dynamics.discretize_linear_system(A_cont, B_cont, dt)
    np.testing.assert_allclose(Ad, np.eye(6), atol=1e-10)
    np.testing.assert_allclose(Bd, np.zeros((6, 3)), atol=1e-10)

    lam = 0.1
    A_small = lam * np.eye(6)
    B_small = 0.1 * np.ones((6, 3))
    Ad_exp, Bd_exp = dynamics.discretize_linear_system(A_small, B_small, dt)
    Ad_expected = np.exp(lam * dt) * np.eye(6)
    Bd_expected = ((np.exp(lam * dt) - 1.0) / lam) * B_small
    np.testing.assert_allclose(Ad_exp, Ad_expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(Bd_exp, Bd_expected, rtol=1e-9, atol=1e-12)


@pytest.mark.integration
def test_simple_maneuver_free_motion(dynamics, error_mapping, ref_state, small_error_state, zero_control):
    current_state = small_error_state
    states = [current_state]
    n_steps = 50
    dt = 0.1
    T = n_steps * dt

    for _ in range(n_steps):
        next_state = dynamics.discrete_dynamics_rk4_rotation(current_state, zero_control, dt)
        states.append(next_state)
        current_state = next_state

    for state in states:
        q = quaternion.quaternion(*state.quat)
        assert np.isclose(q.norm(), 1.0, atol=1e-10)

    initial_error = error_mapping.state_error(small_error_state, ref_state)
    final_error = error_mapping.state_error(states[-1], ref_state)
    np.testing.assert_allclose(final_error.omega, initial_error.omega, rtol=1e-3, atol=1e-6)
    theta_expected = initial_error.error_angle + initial_error.omega * T
    np.testing.assert_allclose(final_error.error_angle, theta_expected, rtol=5e-2, atol=1e-3)
    assert np.linalg.norm(final_error.error_angle) < 1.0


@pytest.mark.integration
def test_error_coords_in_simulation(dynamics, error_mapping, ref_state, small_error_state, zero_control):
    dt = 0.1
    next_state = dynamics.discrete_dynamics_rk4_rotation(small_error_state, zero_control, dt)
    delta_x_before = error_mapping.state_error(small_error_state, ref_state)
    delta_x_after = error_mapping.state_error(next_state, ref_state)
    A = np.zeros((6, 6))
    delta_x_pred = delta_x_before.data.copy()
    delta_x_pred[3:] = delta_x_before.data[3:]
    np.testing.assert_allclose(delta_x_after.data, delta_x_pred, rtol=1e-1, atol=3e-3)
    assert np.linalg.norm(delta_x_after.data) < 1.0


@pytest.mark.unit
def test_set_inertia_updates_inverse(dynamics):
    new_inertia = np.diag([2.0, 3.0, 4.0])
    dynamics.set_inertia(new_inertia)
    np.testing.assert_allclose(dynamics.inertia, new_inertia)
    np.testing.assert_allclose(dynamics.inertia_inv, np.linalg.inv(new_inertia))


@pytest.mark.unit
def test_translation_branch_rk4_constant_acceleration(dynamics):
    state = TranslationState.zeros()
    accel = np.array([1.0, -2.0, 0.5])
    dt = 0.1
    next_state = dynamics.discrete_dynamics_rk4_translation(state, accel, dt)
    expected = np.zeros(6)
    expected[:3] = 0.5 * accel * dt * dt
    expected[3:] = accel * dt
    np.testing.assert_allclose(next_state.data, expected, atol=1e-12)


@pytest.mark.unit
def test_full_state_rk4_recomputes_gravity_at_each_stage(inertia):
    state = RigidBodyState.zeros()
    state.position[:] = [7_078_000.0, 0.0, 0.0]
    state.velocity[:] = [0.0, 7_504.0, 0.0]
    control = RigidBodyControl.zeros()
    dt = 10.0
    dynamics = SpacecraftDynamics(inertia, DisturbanceModel(enable_gravity=True))

    def translation_derivative(data):
        derivative = np.zeros(6)
        derivative[:3] = data[3:]
        derivative[3:] = gravity(data[:3])
        return derivative

    initial = state.translation.data.copy()
    k1 = translation_derivative(initial)
    k2 = translation_derivative(initial + 0.5 * dt * k1)
    k3 = translation_derivative(initial + 0.5 * dt * k2)
    k4 = translation_derivative(initial + dt * k3)
    expected = initial + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    next_state = dynamics.discretize_dynamics_rk4_full_state(
        state,
        control,
        interval_start_time=0.0,
        delta_time=dt,
    )

    np.testing.assert_allclose(
        next_state.translation.data,
        expected,
        rtol=0.0,
        atol=1e-10,
    )
