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
from experiments.single_target_convergence import make_base_config, parse_args
from experiments.single_target_robustness import (
    _build_scenarios,
    parse_args as parse_robustness_args,
)


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
def test_disturbance_summary_interprets_sinusoidal_frequency_as_hz() -> None:
    environment = ClosedLoopEnvironmentConfig(
        disturbance_bias=None,
        disturbance_noise_std=None,
        disturbance_sinusoidal_amplitude=2.0,
        disturbance_sinusoidal_frequency=0.25,
        enable_gravity=False,
    )

    summary = summarize_disturbance_torque(
        environment,
        duration_s=1.25,
        sim_dt=0.25,
        rng_seed=2,
        actuator_min=-3.0 * np.ones(3),
        actuator_max=3.0 * np.ones(3),
    )

    assert summary["disturbance_peak_abs_x_nm"] == pytest.approx(2.0)


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


@pytest.mark.unit
def test_convergence_cli_toggles_disturbance_rejection() -> None:
    assert parse_args([]).disturbance_rejection is False
    assert parse_args(["--disturbance-rejection"]).disturbance_rejection is True
    assert parse_args(["--no-disturbance-rejection"]).disturbance_rejection is False

    enabled = make_base_config(disturbance_rejection=True)
    disabled = make_base_config(disturbance_rejection=False)
    assert enabled.mpc.enable_disturbance_rejection is True
    assert disabled.mpc.enable_disturbance_rejection is False


@pytest.mark.unit
def test_convergence_cli_configures_disturbance_torque() -> None:
    args = parse_args(
        [
            "--angular-rates",
            "0.0",
            "--disturbance-torque-nm",
            "8e-4",
            "-5e-4",
            "3e-4",
        ]
    )
    config = make_base_config(
        disturbance_rejection=False,
        disturbance_torque_nm=args.disturbance_torque_nm,
    )

    np.testing.assert_allclose(
        config.environment.disturbance_bias,
        np.array([8.0e-4, -5.0e-4, 3.0e-4]),
    )
    assert args.angular_rates_deg_s == [0.0]


@pytest.mark.unit
def test_robustness_cli_toggles_disturbance_rejection() -> None:
    defaults = parse_robustness_args([])
    assert defaults.disturbance_rejection is False
    assert defaults.max_duration == pytest.approx(600.0)
    assert defaults.initial_attitude_deg == pytest.approx(30.0)
    assert defaults.initial_rate_deg_s == pytest.approx(0.0)
    assert (
        parse_robustness_args(["--disturbance-rejection"]).disturbance_rejection is True
    )
    assert (
        parse_robustness_args(["--no-disturbance-rejection"]).disturbance_rejection
        is False
    )


@pytest.mark.unit
def test_robustness_scenarios_isolate_torque_and_inertia_changes() -> None:
    base = ClosedLoopTestConfig.defaults(show_progress=False)
    nominal_inertia = base.spacecraft.inertia
    scenarios = _build_scenarios(base, nominal_inertia)

    assert [scenario.name for scenario in scenarios] == [
        "ideal_reference",
        "constant_bias_reference",
        "moderate_disturbance",
        "severe_disturbance",
        "inertia_mismatch",
        "combined_uncertainty",
    ]
    reference_torque = np.array([8.0e-4, -5.0e-4, 3.0e-4])
    expected_torques = [
        np.zeros(3),
        reference_torque,
        5.0 * reference_torque,
        20.0 * reference_torque,
        reference_torque,
        20.0 * reference_torque,
    ]
    for scenario, expected_torque in zip(scenarios, expected_torques):
        np.testing.assert_allclose(
            scenario.environment.disturbance_bias, expected_torque
        )
        assert scenario.environment.disturbance_noise_std is None
        assert scenario.environment.disturbance_sinusoidal_amplitude is None
        assert scenario.environment.disturbance_sinusoidal_frequency is None

    expected_mismatched_inertia = nominal_inertia.copy()
    diagonal = np.diag_indices(3)
    expected_mismatched_inertia[diagonal] *= np.array([1.20, 0.85, 1.15])
    for scenario in scenarios[:4]:
        np.testing.assert_allclose(scenario.truth_inertia, nominal_inertia)
    for scenario in scenarios[4:]:
        np.testing.assert_allclose(scenario.truth_inertia, expected_mismatched_inertia)
        np.testing.assert_allclose(
            scenario.truth_inertia - np.diag(np.diag(scenario.truth_inertia)),
            nominal_inertia - np.diag(np.diag(nominal_inertia)),
        )


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
