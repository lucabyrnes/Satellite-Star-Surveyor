"""Create autonomous multi-target survey figures from a saved experiment.

Run ``experiments/experiment.py`` first, then run:

    python experiments/plot_autonomous_survey.py

The script reads the saved pickle payload; it does not rerun the simulation.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpc_spacecraft.analysis.mission_results import quaternion_angle_error_deg
from mpc_spacecraft.guidance import AstropySunDirectionModel
from mpc_spacecraft.utilities.utils import R_EARTH_M

DEFAULT_DATA_FILE = Path("experiments/closed_loop_experiment.pkl")
DEFAULT_OUTPUT_DIR = Path("experiments/figures/autonomous_survey")
MODE_ORDER = ["NO_TARGET", "SLEWING", "SETTLING", "OBSERVING"]
MODE_ABBREVIATIONS = ["—", "S", "T", "O"]
COLORS = {
    "blue": "#2F80ED",
    "orange": "#F2994A",
    "green": "#27AE60",
    "red": "#EB5757",
    "purple": "#9B51E0",
    "dark": "#111827",
    "gray": "#6B7280",
    "grid": "#D1D5DB",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", type=Path, nargs="?", default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sun-keepout-deg", type=float, default=None)
    parser.add_argument("--earth-margin-deg", type=float, default=None)
    parser.add_argument("--torque-limit-nm", type=float, default=None)
    parser.add_argument(
        "--sun-sample-interval-s",
        type=float,
        default=60.0,
        help="interval for Sun ephemeris samples interpolated over the logs",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=20_000,
        help="maximum time samples drawn in each figure",
    )
    parser.add_argument(
        "--include-post-completion",
        action="store_true",
        help="include samples logged after every target completed",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    payload = load_payload(args.data_file)
    logs = payload["logs"]
    targets = payload["targets"]
    metadata = payload.get("metadata", {})
    source_duration_s = float(logs["time"].iloc[-1] - logs["time"].iloc[0])
    if not args.include_post_completion:
        end_index = mission_completion_index(logs, targets)
        if end_index is not None:
            logs = logs.iloc[: end_index + 1].reset_index(drop=True)

    sun_keepout_deg = _resolve_setting(
        args.sun_keepout_deg, metadata, "sun_keepout_deg", 35.0
    )
    earth_margin_deg = _resolve_setting(
        args.earth_margin_deg, metadata, "earth_margin_deg", 10.0
    )
    torque_limit_nm = _resolve_setting(
        args.torque_limit_nm, metadata, "torque_limit_nm", 0.25
    )
    if args.sun_sample_interval_s <= 0.0:
        raise ValueError("--sun-sample-interval-s must be positive.")
    if args.max_plot_points <= 0:
        raise ValueError("--max-plot-points must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    geometry = compute_true_keepout_geometry(
        logs,
        sun_keepout_deg=sun_keepout_deg,
        earth_margin_deg=earth_margin_deg,
        sun_sample_interval_s=args.sun_sample_interval_s,
    )
    switches = target_switch_indices(logs)
    plot_indices = select_plot_indices(len(logs), args.max_plot_points, switches)

    paths = [
        plot_mission_timeline(logs, targets, switches, plot_indices, args.output_dir),
        plot_keepout_verification(
            logs, geometry, switches, plot_indices, args.output_dir
        ),
        plot_control_tracking(
            logs,
            switches,
            plot_indices,
            torque_limit_nm,
            args.output_dir,
        ),
    ]
    metrics = compute_summary_metrics(
        logs,
        targets,
        geometry,
        metadata=metadata,
    )
    metrics["configuration"] = {
        "sun_keepout_deg": sun_keepout_deg,
        "earth_margin_deg": earth_margin_deg,
        "torque_limit_nm": torque_limit_nm,
        "sun_ephemeris_interpolation_interval_s": args.sun_sample_interval_s,
        "keepout_evaluation": "true spacecraft trajectory and true boresight",
        "post_completion_samples_included": args.include_post_completion,
        "source_logged_duration_s": source_duration_s,
    }
    metrics_json = args.output_dir / "autonomous_survey_metrics.json"
    metrics_csv = args.output_dir / "autonomous_survey_target_metrics.csv"
    safe_metrics = _json_safe(metrics)
    metrics_json.write_text(json.dumps(safe_metrics, indent=2) + "\n")
    pd.DataFrame(safe_metrics["targets"]).to_csv(metrics_csv, index=False)

    print("Generated autonomous survey artifacts:")
    for path in paths:
        print(f"  {path}")
    print(f"  {metrics_json}")
    print(f"  {metrics_csv}")


def load_payload(path: Path) -> dict[str, Any]:
    """Load and validate the pickle payload produced by experiment.py."""
    if not path.exists():
        raise FileNotFoundError(f"Experiment data does not exist: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Experiment payload must be a dictionary.")
    logs = payload.get("logs")
    targets = payload.get("targets")
    if not isinstance(logs, pd.DataFrame) or logs.empty:
        raise TypeError("Experiment payload must contain a nonempty 'logs' DataFrame.")
    if not isinstance(targets, pd.DataFrame) or targets.empty:
        raise TypeError(
            "Experiment payload must contain a nonempty 'targets' DataFrame."
        )
    _require_columns(
        logs,
        {
            "time",
            "epoch_utc",
            "guidance_mode",
            "current_target_idx",
            "active_target_boresight_angle_deg",
            "true_position_x",
            "true_position_y",
            "true_position_z",
            "true_quat_w",
            "true_quat_x",
            "true_quat_y",
            "true_quat_z",
            "true_omega_x",
            "true_omega_y",
            "true_omega_z",
            "control_tx",
            "control_ty",
            "control_tz",
            "mpc_updated",
            "mpc_solve_success",
            "mpc_fallback_used",
        },
    )
    _require_columns(targets, {"req_obs_time"})
    return payload


def compute_true_keepout_geometry(
    logs: pd.DataFrame,
    *,
    sun_keepout_deg: float,
    earth_margin_deg: float,
    sun_sample_interval_s: float,
) -> dict[str, np.ndarray]:
    """Evaluate keepout separation along the logged true trajectory."""
    time = logs["time"].to_numpy(dtype=np.float64)
    position = logs[["true_position_x", "true_position_y", "true_position_z"]].to_numpy(
        dtype=np.float64
    )
    quaternion = logs[
        ["true_quat_w", "true_quat_x", "true_quat_y", "true_quat_z"]
    ].to_numpy(dtype=np.float64)
    boresight = _body_x_axis_in_inertial(quaternion)

    radius = np.linalg.norm(position, axis=1)
    if np.any(radius <= R_EARTH_M):
        raise ValueError("True trajectory contains a position at or inside Earth.")
    earth_direction = -position / radius[:, None]
    earth_separation_deg = _angle_deg(boresight, earth_direction)
    earth_threshold_deg = (
        np.rad2deg(np.arcsin(np.clip(R_EARTH_M / radius, -1.0, 1.0))) + earth_margin_deg
    )

    sun_direction = _interpolated_sun_directions(
        logs["epoch_utc"], time, sun_sample_interval_s
    )
    sun_separation_deg = _angle_deg(boresight, sun_direction)
    sun_threshold_deg = np.full(len(logs), sun_keepout_deg, dtype=np.float64)
    return {
        "sun_separation_deg": sun_separation_deg,
        "sun_threshold_deg": sun_threshold_deg,
        "sun_margin_deg": sun_separation_deg - sun_threshold_deg,
        "earth_separation_deg": earth_separation_deg,
        "earth_threshold_deg": earth_threshold_deg,
        "earth_margin_deg": earth_separation_deg - earth_threshold_deg,
    }


def plot_mission_timeline(
    logs: pd.DataFrame,
    targets: pd.DataFrame,
    switches: np.ndarray,
    indices: np.ndarray,
    output_dir: Path,
) -> Path:
    time = logs["time"].to_numpy(dtype=np.float64)
    target_position = _target_position_series(logs, targets)
    active_observation = active_segment_observation_time(logs, targets)
    required_observation = _active_required_observation(target_position, targets)
    mode_values = _mode_values(logs)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Autonomous Multi-Target Survey — Mission Execution Timeline")

    error = logs["active_target_boresight_angle_deg"].to_numpy(dtype=np.float64)
    axes[0].semilogy(
        time[indices], np.maximum(error[indices], 1.0e-6), color=COLORS["blue"]
    )
    axes[0].axhline(0.5, color=COLORS["orange"], linestyle="--", label="0.5° settling")
    axes[0].axhline(0.01, color=COLORS["red"], linestyle=":", label="0.01° observing")
    axes[0].set_ylabel("Boresight error [deg]")
    axes[0].legend(loc="upper right")

    mode_ax = axes[1]
    target_ax = mode_ax.twinx()
    mode_ax.step(
        time[indices],
        mode_values[indices],
        where="post",
        color=COLORS["blue"],
        label="Mode",
    )
    target_ax.step(
        time[indices],
        target_position[indices],
        where="post",
        color=COLORS["purple"],
        alpha=0.8,
        label="Target",
    )
    mode_ax.set_yticks(range(len(MODE_ORDER)), MODE_ABBREVIATIONS)
    mode_ax.set_ylabel("Guidance mode")
    target_ax.set_ylabel("Active target ID")
    finite_targets = target_position[target_position >= 0]
    if finite_targets.size:
        target_ax.set_yticks(np.unique(finite_targets).astype(int))
    handles = mode_ax.get_lines() + target_ax.get_lines()
    mode_ax.legend(handles, [line.get_label() for line in handles], loc="upper right")

    axes[2].plot(
        time[indices],
        active_observation[indices],
        color=COLORS["green"],
        label="Active-target observation",
    )
    axes[2].step(
        time[indices],
        required_observation[indices],
        where="post",
        color=COLORS["dark"],
        linestyle="--",
        label="Required duration",
    )
    axes[2].set_ylabel("Observation [s]")
    axes[2].set_xlabel("Mission time [s]")
    axes[2].legend(loc="upper right")

    for axis in axes:
        _mark_target_switches(axis, time, switches)
        axis.grid(True, alpha=0.35)
    fig.text(
        0.5,
        0.01,
        "Modes: S = Slewing, T = Settling, O = Observing; vertical lines mark target switches.",
        ha="center",
        color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    return _save(fig, output_dir / "01_mission_execution_timeline.png")


def plot_keepout_verification(
    logs: pd.DataFrame,
    geometry: dict[str, np.ndarray],
    switches: np.ndarray,
    indices: np.ndarray,
    output_dir: Path,
) -> Path:
    time = logs["time"].to_numpy(dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Keepout Constraint Verification — Logged True Trajectory")

    panels = [
        ("sun", "Sun", "Sun keepout threshold", COLORS["orange"]),
        (
            "earth",
            "Earth",
            "Apparent Earth radius + margin",
            COLORS["blue"],
        ),
    ]
    for axis, (key, label, threshold_label, color) in zip(axes, panels):
        separation = geometry[f"{key}_separation_deg"]
        threshold = geometry[f"{key}_threshold_deg"]
        violation = separation < threshold
        axis.plot(
            time[indices], separation[indices], color=color, label=f"{label} separation"
        )
        axis.plot(
            time[indices],
            threshold[indices],
            color=COLORS["dark"],
            linestyle="--",
            label=threshold_label,
        )
        axis.fill_between(
            time[indices],
            separation[indices],
            threshold[indices],
            where=violation[indices],
            color=COLORS["red"],
            alpha=0.35,
            label="Violation",
        )
        axis.set_ylabel("Separation [deg]")
        axis.legend(loc="best")
        axis.grid(True, alpha=0.35)
        _mark_target_switches(axis, time, switches)
    axes[-1].set_xlabel("Mission time [s]")
    fig.text(
        0.5,
        0.01,
        "Constraints evaluated from true attitude, true position, and Sun ephemeris; not MPC predictions.",
        ha="center",
        color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return _save(fig, output_dir / "02_keepout_constraint_verification.png")


def plot_control_tracking(
    logs: pd.DataFrame,
    switches: np.ndarray,
    indices: np.ndarray,
    torque_limit_nm: float,
    output_dir: Path,
) -> Path:
    time = logs["time"].to_numpy(dtype=np.float64)
    omega = logs[["true_omega_x", "true_omega_y", "true_omega_z"]].to_numpy(
        dtype=np.float64
    )
    omega_deg_s = np.rad2deg(omega)
    omega_norm_deg_s = np.linalg.norm(omega_deg_s, axis=1)
    control = logs[["control_tx", "control_ty", "control_tz"]].to_numpy(
        dtype=np.float64
    )
    attitude_error = quaternion_angle_error_deg(
        logs[["true_quat_w", "true_quat_x", "true_quat_y", "true_quat_z"]].to_numpy(),
        logs[
            [
                "estimated_quat_w",
                "estimated_quat_x",
                "estimated_quat_y",
                "estimated_quat_z",
            ]
        ].to_numpy(),
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Control and Tracking Performance")
    for component, color in zip(
        range(3), [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    ):
        axes[0].plot(
            time[indices],
            omega_deg_s[indices, component],
            color=color,
            alpha=0.65,
            label="xyz"[component],
        )
    axes[0].plot(
        time[indices], omega_norm_deg_s[indices], color=COLORS["dark"], label="norm"
    )
    axes[0].axhline(
        0.01, color=COLORS["red"], linestyle="--", label="0.01°/s observing limit"
    )
    axes[0].set_title("True body-frame angular velocity")
    axes[0].set_ylabel("Body rate [deg/s]")
    axes[0].legend(loc="upper right", ncols=5)

    for component, color in zip(
        range(3), [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    ):
        axes[1].plot(
            time[indices],
            control[indices, component],
            color=color,
            label=f"T{'xyz'[component]}",
        )
    axes[1].axhline(
        torque_limit_nm, color=COLORS["red"], linestyle="--", label="Actuator limits"
    )
    axes[1].axhline(-torque_limit_nm, color=COLORS["red"], linestyle="--")
    axes[1].set_title("Commanded body-frame control torque")
    axes[1].set_ylabel("Commanded torque [N·m]")
    axes[1].legend(loc="upper right", ncols=4)

    axes[2].semilogy(
        time[indices],
        np.maximum(attitude_error[indices], 1.0e-12),
        color=COLORS["purple"],
    )
    axes[2].set_title(
        "Attitude estimation error (shortest rotation from true to estimated attitude)"
    )
    axes[2].set_ylabel("Estimation error [deg]")
    axes[2].set_xlabel("Mission time [s]")

    for axis in axes:
        axis.grid(True, alpha=0.35)
        _mark_target_switches(axis, time, switches)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, output_dir / "03_control_tracking_performance.png")


def compute_summary_metrics(
    logs: pd.DataFrame,
    targets: pd.DataFrame,
    geometry: dict[str, np.ndarray],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    time = logs["time"].to_numpy(dtype=np.float64)
    durations = _sample_durations(time, metadata.get("sim_dt"))
    target_position = _target_position_series(logs, targets)
    modes = logs["guidance_mode"].astype(str).to_numpy()
    required = targets["req_obs_time"].to_numpy(dtype=np.float64)
    observed = _final_observation_times(logs, len(targets))
    completed = observed >= required
    completion_times = _target_completion_times(logs, required)
    all_complete = np.all(
        np.column_stack(
            [
                logs.get(
                    f"target_{idx}_obs_time", pd.Series(0.0, index=logs.index)
                ).to_numpy()
                >= requirement
                for idx, requirement in enumerate(required)
            ]
        ),
        axis=1,
    )
    complete_indices = np.flatnonzero(all_complete)
    mission_completion_time = (
        float(time[complete_indices[0]]) if complete_indices.size else None
    )

    target_rows: list[dict[str, Any]] = []
    for position in range(len(targets)):
        active = target_position == position
        target_rows.append(
            {
                "target_id": _target_id(targets, position),
                "completed": bool(completed[position]),
                "required_observation_s": float(required[position]),
                "observation_duration_s": float(observed[position]),
                "completion_time_s": completion_times[position],
                "slewing_time_s": float(
                    np.sum(durations[active & (modes == "SLEWING")])
                ),
                "settling_time_s": float(
                    np.sum(durations[active & (modes == "SETTLING")])
                ),
            }
        )

    update = logs["mpc_updated"].to_numpy(dtype=bool)
    success = logs["mpc_solve_success"].to_numpy(dtype=bool)[update]
    fallback = logs["mpc_fallback_used"].to_numpy(dtype=bool)[update]
    return {
        "schema_version": "mpc_spacecraft_autonomous_survey_metrics_v1",
        "mission": {
            "targets_completed": int(np.count_nonzero(completed)),
            "target_count": int(len(targets)),
            "mission_completion_time_s": mission_completion_time,
            "logged_duration_s": float(time[-1] - time[0]),
            "minimum_sun_exclusion_margin_deg": float(
                np.min(geometry["sun_margin_deg"])
            ),
            "minimum_earth_exclusion_margin_deg": float(
                np.min(geometry["earth_margin_deg"])
            ),
            "sun_violation_sample_count": int(
                np.count_nonzero(geometry["sun_margin_deg"] < 0.0)
            ),
            "earth_violation_sample_count": int(
                np.count_nonzero(geometry["earth_margin_deg"] < 0.0)
            ),
            "mpc_update_count": int(np.count_nonzero(update)),
            "mpc_solve_success_rate": float(np.mean(success)) if success.size else None,
            "mpc_fallback_count": int(np.count_nonzero(fallback)),
        },
        "targets": target_rows,
    }


def active_segment_observation_time(
    logs: pd.DataFrame, targets: pd.DataFrame
) -> np.ndarray:
    """Return active-target observation gained since each target switch."""
    target_position = _target_position_series(logs, targets)
    result = np.full(len(logs), np.nan, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(target_position) != 0) + 1]
    ends = np.r_[starts[1:], len(logs)]
    for start, end in zip(starts, ends):
        position = int(target_position[start])
        column = f"target_{position}_obs_time"
        if position < 0 or column not in logs:
            continue
        values = logs[column].to_numpy(dtype=np.float64)
        result[start:end] = values[start:end] - values[start]
    return result


def target_switch_indices(logs: pd.DataFrame) -> np.ndarray:
    target = logs["current_target_idx"].fillna(-1).to_numpy(dtype=np.int64)
    return np.flatnonzero(np.diff(target) != 0) + 1


def mission_completion_index(logs: pd.DataFrame, targets: pd.DataFrame) -> int | None:
    """Return the first row where every target has met its observation time."""
    required = targets["req_obs_time"].to_numpy(dtype=np.float64)
    complete_by_target = []
    for index, requirement in enumerate(required):
        column = f"target_{index}_obs_time"
        if column not in logs:
            return None
        complete_by_target.append(
            logs[column].to_numpy(dtype=np.float64) >= requirement
        )
    complete = np.all(np.column_stack(complete_by_target), axis=1)
    indices = np.flatnonzero(complete)
    return int(indices[0]) if indices.size else None


def select_plot_indices(
    sample_count: int, max_points: int, important_indices: np.ndarray
) -> np.ndarray:
    stride = max(1, int(np.ceil(sample_count / max_points)))
    regular = np.arange(0, sample_count, stride, dtype=np.int64)
    important = np.concatenate(
        [important_indices - 1, important_indices, important_indices + 1]
    )
    return np.unique(
        np.clip(np.r_[regular, important, sample_count - 1], 0, sample_count - 1)
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.35,
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.titlesize": 16,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
        }
    )


def _interpolated_sun_directions(
    epochs: pd.Series, time: np.ndarray, interval_s: float
) -> np.ndarray:
    anchor_indices = np.searchsorted(
        time, np.arange(time[0], time[-1] + interval_s, interval_s)
    )
    anchor_indices = np.unique(
        np.clip(np.r_[anchor_indices, len(time) - 1], 0, len(time) - 1)
    )
    model = AstropySunDirectionModel()
    directions = np.vstack(
        [model.sun_dir_eci(epochs.iloc[int(index)]) for index in anchor_indices]
    )
    interpolated = np.column_stack(
        [
            np.interp(time, time[anchor_indices], directions[:, component])
            for component in range(3)
        ]
    )
    return interpolated / np.linalg.norm(interpolated, axis=1, keepdims=True)


def _body_x_axis_in_inertial(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = quaternion.T
    return np.column_stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ]
    )


def _angle_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cosine = np.einsum("ij,ij->i", first, second)
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _mode_values(logs: pd.DataFrame) -> np.ndarray:
    mapping = {mode: index for index, mode in enumerate(MODE_ORDER)}
    return logs["guidance_mode"].map(mapping).fillna(0).to_numpy(dtype=np.int64)


def _target_position_series(logs: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    mapping = {
        int(target_id): position for position, target_id in enumerate(targets.index)
    }
    ids = logs["current_target_idx"].fillna(-1).to_numpy(dtype=np.int64)
    return np.array(
        [mapping.get(int(target_id), -1) for target_id in ids], dtype=np.int64
    )


def _active_required_observation(
    target_position: np.ndarray, targets: pd.DataFrame
) -> np.ndarray:
    required = targets["req_obs_time"].to_numpy(dtype=np.float64)
    result = np.full(len(target_position), np.nan, dtype=np.float64)
    valid = target_position >= 0
    result[valid] = required[target_position[valid]]
    return result


def _target_completion_times(
    logs: pd.DataFrame, required: np.ndarray
) -> list[float | None]:
    time = logs["time"].to_numpy(dtype=np.float64)
    completion: list[float | None] = []
    for index, requirement in enumerate(required):
        column = f"target_{index}_obs_time"
        if column not in logs:
            completion.append(None)
            continue
        reached = np.flatnonzero(logs[column].to_numpy(dtype=np.float64) >= requirement)
        completion.append(float(time[reached[0]]) if reached.size else None)
    return completion


def _final_observation_times(logs: pd.DataFrame, count: int) -> np.ndarray:
    return np.array(
        [
            (
                float(logs[f"target_{index}_obs_time"].iloc[-1])
                if f"target_{index}_obs_time" in logs
                else 0.0
            )
            for index in range(count)
        ]
    )


def _sample_durations(time: np.ndarray, configured_dt: Any) -> np.ndarray:
    fallback = float(configured_dt) if configured_dt is not None else 0.0
    if len(time) > 1:
        positive_steps = np.diff(time)
        positive_steps = positive_steps[positive_steps > 0.0]
        if positive_steps.size:
            fallback = float(np.median(positive_steps))
    return np.diff(time, append=time[-1] + fallback)


def _target_id(targets: pd.DataFrame, position: int) -> int | str:
    value = targets.index[position]
    return int(value) if isinstance(value, (int, np.integer)) else str(value)


def _mark_target_switches(
    axis: plt.Axes, time: np.ndarray, switches: np.ndarray
) -> None:
    for index in switches:
        axis.axvline(time[index], color=COLORS["gray"], linewidth=0.8, alpha=0.35)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _resolve_setting(
    cli_value: float | None,
    metadata: dict[str, Any],
    key: str,
    default: float,
) -> float:
    if cli_value is not None:
        return float(cli_value)
    return float(metadata.get(key, default))


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            f"Saved experiment data is missing columns: {', '.join(missing)}"
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return round(value, 9)
    return value


if __name__ == "__main__":
    main()
