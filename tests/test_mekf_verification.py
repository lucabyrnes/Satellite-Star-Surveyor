import numpy as np
import pytest
import quaternion as qu
from scipy.stats import chi2

from experiments.mekf_statistical_verification import parse_args
from mpc_spacecraft.analysis.mekf_verification import (
    MEKFVerificationConfig,
    average_chi_square_bounds,
    mekf_error_state,
    normalized_squared_error,
    run_mekf_verification,
)
from mpc_spacecraft.dynamics.disturbances import gravity
from mpc_spacecraft.estimation.MEKF import MEKF
from mpc_spacecraft.estimation.sensor_models import (
    GNSSConfig,
    GyroConfig,
    IMUConfig,
    StarTrackerConfig,
    make_mekf_initial_covariance,
)
from mpc_spacecraft.utilities.utils import SensorRigidBodyState


def _state() -> SensorRigidBodyState:
    state = SensorRigidBodyState.zeros()
    state.position[:] = [7_078e3, 0.0, 0.0]
    state.velocity[:] = [0.0, 7.504e3, 0.0]
    state.quat[:] = [1.0, 0.0, 0.0, 0.0]
    return state


def _mekf() -> MEKF:
    return MEKF(
        _state(),
        IMUConfig(),
        GyroConfig(),
        GNSSConfig(),
        StarTrackerConfig(),
        make_mekf_initial_covariance(),
    )


@pytest.mark.unit
def test_mekf_prediction_uses_second_order_translation_update() -> None:
    initial = _state()
    dt = 0.1
    initial_acceleration = gravity(initial.position)
    expected_position = (
        initial.position
        + initial.velocity * dt
        + 0.5 * initial_acceleration * dt**2
    )
    final_acceleration = gravity(expected_position)
    mekf = _mekf()

    estimate = mekf.update(dt, np.zeros(3), np.zeros(3))

    np.testing.assert_allclose(
        estimate.position,
        expected_position,
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        estimate.velocity,
        initial.velocity + 0.5 * (initial_acceleration + final_acceleration) * dt,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.unit
def test_mekf_error_state_uses_documented_order_and_attitude_convention() -> None:
    estimate = _state()
    truth = estimate.copy()
    truth.position[:] += [1.0, 2.0, 3.0]
    truth.velocity[:] += [4.0, 5.0, 6.0]
    attitude_error = np.array([0.01, -0.02, 0.03])
    truth.quat[:] = qu.as_float_array(
        qu.quaternion(*estimate.quat) * qu.from_rotation_vector(attitude_error)
    )
    truth.gyro_bias[:] = [7.0, 8.0, 9.0]
    truth.accel_bias[:] = [10.0, 11.0, 12.0]

    error = mekf_error_state(truth, estimate)

    np.testing.assert_allclose(error[0:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(error[3:6], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(error[6:9], attitude_error, atol=1e-14)
    np.testing.assert_allclose(error[9:12], [7.0, 8.0, 9.0])
    np.testing.assert_allclose(error[12:15], [10.0, 11.0, 12.0])


@pytest.mark.unit
def test_mekf_error_state_is_invariant_to_quaternion_sign() -> None:
    estimate = _state()
    truth = estimate.copy()
    truth.quat[:] *= -1.0

    np.testing.assert_allclose(mekf_error_state(truth, estimate), np.zeros(15))


@pytest.mark.unit
def test_normalized_squared_error_uses_covariance_weighting() -> None:
    vector = np.array([2.0, 3.0])
    covariance = np.diag([4.0, 9.0])

    assert normalized_squared_error(vector, covariance) == pytest.approx(2.0)


@pytest.mark.unit
def test_average_chi_square_bounds_match_scipy_definition() -> None:
    lower, upper = average_chi_square_bounds(15, 100, 0.95)

    assert lower == pytest.approx(chi2.ppf(0.025, 1500) / 100)
    assert upper == pytest.approx(chi2.ppf(0.975, 1500) / 100)


@pytest.mark.unit
def test_mekf_exposes_both_innovations_and_clears_them_next_update() -> None:
    mekf = _mekf()
    gnss = np.concatenate([_state().position, _state().velocity])

    mekf.update(
        0.1,
        np.zeros(3),
        np.zeros(3),
        gnss_measure=gnss,
        st_measure=qu.quaternion(1.0, 0.0, 0.0, 0.0),
    )

    assert [item.sensor for item in mekf.last_innovations] == [
        "star_tracker",
        "gnss",
    ]
    assert mekf.last_innovations[0].residual.shape == (3,)
    assert mekf.last_innovations[0].covariance.shape == (3, 3)
    assert mekf.last_innovations[1].residual.shape == (6,)
    assert mekf.last_innovations[1].covariance.shape == (6, 6)

    mekf.update(0.1, np.zeros(3), np.zeros(3))
    assert mekf.last_innovations == ()


@pytest.mark.unit
def test_verification_cli_defaults_and_overrides() -> None:
    defaults = parse_args([])
    overridden = parse_args(["--runs", "12", "--duration", "5", "--dt", "0.2"])

    assert defaults.runs == 100
    assert defaults.confidence == pytest.approx(0.95)
    assert overridden.runs == 12
    assert overridden.duration == pytest.approx(5.0)
    assert overridden.dt == pytest.approx(0.2)


@pytest.mark.integration
def test_small_mekf_verification_is_finite_and_reproducible() -> None:
    config = MEKFVerificationConfig(
        run_count=2,
        duration_s=0.2,
        dt_s=0.1,
        gnss_period_s=0.2,
        star_tracker_period_s=0.1,
        base_seed=42,
    )

    first = run_mekf_verification(config)
    second = run_mekf_verification(config)

    assert first.mean_error.shape == (3, 15)
    assert first.star_tracker_nis.measurement_dimension == 3
    assert first.gnss_nis.measurement_dimension == 6
    assert np.all(np.isfinite(first.average_nees))
    np.testing.assert_array_equal(first.mean_error, second.mean_error)
    np.testing.assert_array_equal(first.average_nees, second.average_nees)
