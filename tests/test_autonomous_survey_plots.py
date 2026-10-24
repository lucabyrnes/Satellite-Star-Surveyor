import numpy as np
import pandas as pd
import pytest

from experiments.plot_autonomous_survey import (
    _body_x_axis_in_inertial,
    active_segment_observation_time,
    compute_summary_metrics,
    mission_completion_index,
    target_switch_indices,
)


@pytest.fixture
def survey_logs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": np.arange(8.0),
            "current_target_idx": [0, 0, 0, 1, 1, 1, 1, 1],
            "guidance_mode": [
                "SLEWING",
                "SETTLING",
                "OBSERVING",
                "SLEWING",
                "SETTLING",
                "OBSERVING",
                "OBSERVING",
                "OBSERVING",
            ],
            "target_0_obs_time": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "target_1_obs_time": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            "mpc_updated": [True, False, True, False, True, False, True, False],
            "mpc_solve_success": [True, True, True, True, False, True, True, True],
            "mpc_fallback_used": [
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
            ],
        }
    )


@pytest.mark.unit
def test_active_observation_time_resets_at_target_switch(
    survey_logs: pd.DataFrame,
) -> None:
    targets = pd.DataFrame({"req_obs_time": [1.0, 1.0]})

    observation = active_segment_observation_time(survey_logs, targets)

    np.testing.assert_allclose(observation, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(target_switch_indices(survey_logs), [3])
    assert mission_completion_index(survey_logs, targets) == 5


@pytest.mark.unit
def test_summary_reports_mission_target_and_solver_metrics(
    survey_logs: pd.DataFrame,
) -> None:
    targets = pd.DataFrame({"req_obs_time": [1.0, 1.0]})
    geometry = {
        "sun_margin_deg": np.array([5.0, 4.0, 3.0, 2.0, 1.0, -1.0, 2.0, 3.0]),
        "earth_margin_deg": np.arange(8.0),
    }

    metrics = compute_summary_metrics(
        survey_logs,
        targets,
        geometry,
        metadata={"sim_dt": 1.0},
    )

    mission = metrics["mission"]
    assert mission["targets_completed"] == 2
    assert mission["mission_completion_time_s"] == pytest.approx(5.0)
    assert mission["minimum_sun_exclusion_margin_deg"] == pytest.approx(-1.0)
    assert mission["sun_violation_sample_count"] == 1
    assert mission["mpc_update_count"] == 4
    assert mission["mpc_solve_success_rate"] == pytest.approx(0.75)
    assert mission["mpc_fallback_count"] == 1
    assert metrics["targets"][0]["slewing_time_s"] == pytest.approx(1.0)
    assert metrics["targets"][0]["settling_time_s"] == pytest.approx(1.0)
    assert metrics["targets"][1]["completion_time_s"] == pytest.approx(5.0)


@pytest.mark.unit
def test_true_boresight_comes_from_logged_attitude() -> None:
    half_angle = np.deg2rad(45.0)
    quaternions = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)],
        ]
    )

    boresight = _body_x_axis_in_inertial(quaternions)

    np.testing.assert_allclose(
        boresight,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        atol=1.0e-12,
    )
