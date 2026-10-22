from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import quaternion as qu

from mpc_spacecraft.analysis.single_target import (
    make_single_target_config,
    make_single_target_table,
    make_target_relative_initial_state,
    perturb_inertia,
    summarize_disturbance_torque,
    sustained_entry_time,
    target_direction_from_table,
)
from mpc_spacecraft.config import (
    ClosedLoopEnvironmentConfig,
    ClosedLoopTestConfig,
    InitialStateConfig,
    TargetConfig,
)
from mpc_spacecraft.simulation import run_closed_loop_test
from mpc_spacecraft.utilities.utils import BODY_FORWARD_VEC3


@pytest.mark.unit
def test_sustained_entry_time_requires_full_dwell() -> None:
    time = np.arange(0.0, 8.0, 1.0)
    condition = np.array([False, True, True, False, True, True, True, True])

    assert sustained_entry_time(time, condition, dwell_time_s=3.0) == pytest.approx(4.0)
    assert np.isnan(sustained_entry_time(time, condition, dwell_time_s=4.0))


@pytest.mark.unit
def test_target_relative_state_has_requested_boresight_offset() -> None:
    targets = make_single_target_table()
    state = make_target_relative_initial_state(
        InitialStateConfig(),
        targets,
        attitude_offset_deg=30.0,
        angular_rate_deg_s=0.25,
    )
    body_direction = qu.rotate_vectors(qu.quaternion(*state.quat), BODY_FORWARD_VEC3)
    target_direction = target_direction_from_table(targets)
    angle_deg = np.rad2deg(
        np.arccos(np.clip(body_direction @ target_direction, -1.0, 1.0))
    )

    assert angle_deg == pytest.approx(30.0)
    assert np.rad2deg(np.linalg.norm(state.omega)) == pytest.approx(0.25)


@pytest.mark.unit
def test_perturbed_inertia_remains_symmetric_positive_definite() -> None:
    inertia = np.array(
        [[2600.0, -120.0, 80.0], [-120.0, 2400.0, -60.0], [80.0, -60.0, 1500.0]]
    )
    perturbed = perturb_inertia(inertia, np.array([1.35, 0.70, 1.25]))

    np.testing.assert_allclose(perturbed, perturbed.T)
    assert np.all(np.linalg.eigvalsh(perturbed) > 0.0)


@pytest.mark.unit
def test_disturbance_summary_compares_torque_with_axis_limits() -> None:
    environment = ClosedLoopEnvironmentConfig(
        disturbance_bias=np.array([0.03, -0.04, 0.0]),
        disturbance_noise_std=None,
        disturbance_sinusoidal_amplitude=None,
        disturbance_sinusoidal_frequency=None,
    )
    summary = summarize_disturbance_torque(
        environment,
        duration_s=1.0,
        sim_dt=0.1,
        rng_seed=2,
        actuator_min=-0.25 * np.ones(3),
        actuator_max=0.25 * np.ones(3),
    )

    assert summary["disturbance_mean_norm_nm"] == pytest.approx(0.05)
    assert summary["disturbance_peak_norm_nm"] == pytest.approx(0.05)
    assert summary["disturbance_peak_axis_utilization"] == pytest.approx(0.16)
    assert summary["disturbance_exceeds_axis_limit"] is False


@pytest.mark.unit
def test_single_target_config_uses_requested_timing_and_mekf_noise() -> None:
    config = make_single_target_config(
        attitude_offset_deg=5.0,
        angular_rate_deg_s=0.0,
        max_duration_s=300.0,
        sim_dt=0.01,
    )

    assert config.timing.sim_dt == pytest.approx(0.01)
    assert config.timing.mpc_dt == pytest.approx(2.0)
    assert config.timing.sim_cycles == 30_000
    assert config.stop_when_complete is True
    assert config.sensors.enable_measurement_noise is True
    assert config.sensors.enable_bias_random_walk is True
    assert config.targets.targets.loc[0, "req_obs_time"] == pytest.approx(30.0)


@pytest.mark.integration
def test_closed_loop_can_stop_immediately_when_mission_is_complete() -> None:
    completed_target = pd.DataFrame(
        {"req_obs_time": [0.0], "Dec": [0.174533], "RA": [0.261799]}
    )
    base = ClosedLoopTestConfig.defaults(show_progress=False)
    config = replace(
        base,
        timing=replace(base.timing, sim_dt=0.01, sim_cycles=10),
        targets=TargetConfig(targets=completed_target),
        stop_when_complete=True,
    )

    result = run_closed_loop_test(config)

    assert len(result.records) == 1
    assert result.records[0].is_complete is True
    assert result.final_time == pytest.approx(0.0)


@pytest.mark.integration
def test_mpc_prediction_uses_configured_model_inertia(monkeypatch) -> None:
    import mpc_spacecraft.simulation.closed_loop_testing as closed_loop

    captured: dict[str, np.ndarray] = {}

    class FakeProvider:
        def __init__(self, dynamics, sun_model, sun_margin, mpc_dt):
            captured["inertia"] = dynamics.inertia.copy()

    class FakeMPC:
        last_solve_success = True
        last_fallback_used = False

        def __init__(self, *args, **kwargs):
            pass

        def get_first_control(self, **kwargs):
            return np.zeros(3)

    monkeypatch.setattr(closed_loop, "SpacecraftErrorDynamicsProvider", FakeProvider)
    monkeypatch.setattr(closed_loop, "NominalMPC", FakeMPC)

    base = ClosedLoopTestConfig.defaults(show_progress=False)
    truth_inertia = perturb_inertia(base.spacecraft.inertia, np.array([1.2, 0.8, 1.1]))
    model_inertia = base.spacecraft.inertia.copy()
    completed_target = pd.DataFrame(
        {"req_obs_time": [0.0], "Dec": [0.174533], "RA": [0.261799]}
    )
    config = replace(
        base,
        timing=replace(base.timing, sim_cycles=1),
        spacecraft=replace(base.spacecraft, inertia=truth_inertia),
        mpc=replace(base.mpc, model_inertia=model_inertia),
        targets=TargetConfig(targets=completed_target),
        stop_when_complete=True,
    )

    result = run_closed_loop_test(config)

    np.testing.assert_allclose(captured["inertia"], model_inertia)
    np.testing.assert_allclose(result.config.spacecraft.inertia, truth_inertia)
