"""Create report-ready and optional interactive plots from mission results.

Run after ``experiments/results_experiment.py`` has generated
``experiments/results/mission_results_timeseries.csv`` and
``experiments/results/mission_results_summary.json``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_PATH = Path("experiments/results/mission_plot")

COLORS = {
    "blue": "#2F80ED",
    "teal": "#00A6A6",
    "green": "#27AE60",
    "orange": "#F2994A",
    "red": "#EB5757",
    "purple": "#9B51E0",
    "gray": "#6B7280",
    "dark": "#111827",
    "grid": "#D1D5DB",
    "paper": "#F8FAFC",
}
SERIES_COLORS = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"], COLORS["teal"], COLORS["red"]]


def main() -> None:
    args = parse_args()
    result_paths = resolve_result_paths(Path(args.results_path))
    output_dir = result_paths["figures_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    logs = pd.read_csv(result_paths["timeseries_csv"])
    summary = load_summary(result_paths["summary_json"])
    metrics = build_metrics(logs, summary)
    write_metrics(metrics, result_paths["metrics_json"], result_paths["metrics_csv"])

    generated = []
    generated.append(plot_results_dashboard(logs, summary, metrics, output_dir))
    generated.append(plot_planner_timeline(logs, summary, output_dir))
    generated.append(plot_pointing_and_keepout(logs, output_dir))
    generated.append(plot_mpc_control(logs, summary, metrics, output_dir))
    generated.append(plot_estimator_errors(logs, summary, metrics, output_dir))
    generated.append(plot_position_velocity_comparison(logs, output_dir))
    generated.append(plot_attitude_rate_comparison(logs, output_dir))
    generated.append(plot_mekf_uncertainty(logs, metrics, output_dir))

    interactive_generated: list[Path] = []
    if args.interactive:
        interactive_dir = result_paths["interactive_dir"]
        interactive_dir.mkdir(parents=True, exist_ok=True)
        interactive_generated = plot_interactive_pages(logs, summary, metrics, interactive_dir)

    print("Generated result figures:")
    for path in generated:
        print(f"  {path}")
    print("Generated structured metrics:")
    print(f"  {result_paths['metrics_json']}")
    print(f"  {result_paths['metrics_csv']}")
    if interactive_generated:
        print("Generated interactive figures:")
        for path in interactive_generated:
            print(f"  {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_path",
        nargs="?",
        default=DEFAULT_RESULTS_PATH,
        help=(
            "Path to a result folder or result stem. For example, "
            "experiments/results/baseline resolves baseline/baseline_timeseries.csv and baseline/baseline_summary.json."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Also export interactive Plotly HTML pages. Requires plotly to be installed.",
    )
    return parser.parse_args()


def resolve_result_paths(results_path: Path) -> dict[str, Path]:
    """Resolve result input files and derived output paths from a folder/stem path."""
    if results_path.suffix in {".csv", ".json", ".npz"}:
        stem = results_path.name
        for suffix in ["_timeseries.csv", "_summary.json", ".npz", ".csv", ".json"]:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        result_dir = results_path.parent
    else:
        result_dir = results_path
        stem = results_path.name

    if stem == "results":
        stem = "mission_results"

    timeseries_csv = result_dir / f"{stem}_timeseries.csv"
    summary_json = result_dir / f"{stem}_summary.json"
    legacy_timeseries = result_dir / "mission_results_timeseries.csv"
    legacy_summary = result_dir / "mission_results_summary.json"
    if not timeseries_csv.exists() and legacy_timeseries.exists():
        timeseries_csv = legacy_timeseries
    if not summary_json.exists() and legacy_summary.exists():
        summary_json = legacy_summary

    return {
        "result_dir": result_dir,
        "timeseries_csv": timeseries_csv,
        "summary_json": summary_json,
        "figures_dir": result_dir / "figures",
        "interactive_dir": result_dir / "interactive",
        "metrics_json": result_dir / f"{stem}_plot_metrics.json",
        "metrics_csv": result_dir / f"{stem}_plot_metrics.csv",
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": COLORS["dark"],
            "axes.titlecolor": COLORS["dark"],
            "axes.prop_cycle": plt.cycler(color=SERIES_COLORS),
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.55,
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 18,
            "lines.linewidth": 2.0,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_metrics(logs: pd.DataFrame, summary: dict[str, object]) -> dict[str, Any]:
    completion_metrics = build_completion_time_metrics(logs, summary)
    metrics: dict[str, Any] = {
        "mission": {
            "outcome": summary.get("outcome"),
            "targets_completed": summary.get("targets_completed"),
            "target_count": summary.get("target_count"),
            "duration_s": summary.get("mission_duration_s"),
            "all_targets_completed_time_s": completion_metrics.get("all_targets_completed_time_s"),
            "observing_time_s": summary.get("observing_time_s"),
            "target_switch_count": summary.get("target_switch_count"),
            "plotting_note": (
                "Matplotlib PNGs are optimized for report-ready summaries. Use --interactive for Plotly HTML "
                "when zoom, hover, and trace toggling are needed for long simulations."
            ),
        },
        "observation": build_observation_metrics(summary, completion_metrics),
        "planner_modes": summary.get("planner_mode_seconds", {}),
        "state_estimation_error": {},
        "mpc_control": {},
        "pointing_and_keepout": {},
        "mekf_uncertainty": {},
        "measurement_updates": {},
    }

    metric_columns = {
        "state_estimation_error": {
            "attitude_error_deg": "derived_attitude_estimation_error_deg",
            "position_error_m": "derived_position_error_m",
            "velocity_error_mps": "derived_velocity_error_mps",
            "angular_rate_error_radps": "derived_angular_rate_error_radps",
        },
        "pointing_and_keepout": {
            "goal_angle_error_deg": "derived_goal_angle_error_deg",
            "boresight_target_angle_deg": "active_target_boresight_angle_deg",
            "sun_keepout_margin_deg": "active_target_sun_keepout_margin_deg",
        },
        "mpc_control": {
            "torque_norm_nm": "derived_torque_norm_nm",
            "control_tx_nm": "control_tx",
            "control_ty_nm": "control_ty",
            "control_tz_nm": "control_tz",
        },
    }
    for group, columns in metric_columns.items():
        metrics[group].update({name: summarize_series(logs[col]) for name, col in columns.items() if col in logs})

    time = logs["time"].to_numpy(dtype=float) if "time" in logs else np.arange(len(logs), dtype=float)
    if "derived_torque_norm_nm" in logs:
        torque = logs["derived_torque_norm_nm"].to_numpy(dtype=float)
        metrics["mpc_control"]["torque_integral_nm_s"] = safe_float(np.trapezoid(np.nan_to_num(torque), time))
        metrics["mpc_control"]["peak_torque_time_s"] = safe_float(time[int(np.nanargmax(torque))]) if len(torque) else None
    for flag in ["mpc_updated", "mpc_solve_success", "mpc_fallback_used"]:
        if flag in logs:
            values = logs[flag].astype(bool)
            metrics["mpc_control"][flag] = {"count": int(values.sum()), "fraction": safe_float(values.mean())}
    metrics["mpc_control"]["solve_success_rate_summary"] = summary.get("mpc_solve_success_rate")

    for flag in ["estimator_used_gnss", "estimator_used_star_tracker"]:
        if flag in logs:
            values = logs[flag].astype(bool)
            metrics["measurement_updates"][flag] = {"count": int(values.sum()), "fraction": safe_float(values.mean())}

    for col in [col for col in logs.columns if col.startswith("derived_mekf_three_sigma_")]:
        name = col.replace("derived_mekf_three_sigma_", "")
        metrics["mekf_uncertainty"][name] = summarize_series(logs[col])

    return metrics


def build_completion_time_metrics(logs: pd.DataFrame, summary: dict[str, object]) -> dict[str, Any]:
    if "time" not in logs:
        return {"all_targets_completed_time_s": None, "target_completion_time_s": []}
    time = logs["time"].to_numpy(dtype=float)
    req = np.asarray(summary.get("required_observation_time_by_target_s", []), dtype=float)
    target_completion_times: list[float | None] = []
    for idx, required in enumerate(req):
        col = f"target_{idx}_obs_time"
        if col not in logs or not np.isfinite(required):
            target_completion_times.append(None)
            continue
        observed = logs[col].to_numpy(dtype=float)
        completed_indices = np.where(observed >= required)[0]
        target_completion_times.append(safe_float(time[int(completed_indices[0])]) if completed_indices.size else None)

    finite_completion_times = [value for value in target_completion_times if value is not None]
    all_targets_completed_time = max(finite_completion_times) if len(finite_completion_times) == len(req) and len(req) else None
    return {
        "all_targets_completed_time_s": all_targets_completed_time,
        "target_completion_time_s": target_completion_times,
    }


def build_observation_metrics(summary: dict[str, object], completion_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    obs = np.asarray(summary.get("observation_time_by_target_s", []), dtype=float)
    req = np.asarray(summary.get("required_observation_time_by_target_s", []), dtype=float)
    complete_value = summary.get("target_complete", [])
    complete = complete_value if isinstance(complete_value, list) else []
    completion_times = completion_metrics.get("target_completion_time_s", [])
    rows = []
    for idx, observed in enumerate(obs):
        required = req[idx] if idx < len(req) else np.nan
        rows.append(
            {
                "target": int(idx),
                "observed_s": safe_float(observed),
                "required_s": safe_float(required),
                "completion_fraction": safe_float(observed / required) if required and np.isfinite(required) else None,
                "complete": bool(complete[idx]) if idx < len(complete) else bool(observed >= required),
                "completion_time_s": completion_times[idx] if isinstance(completion_times, list) and idx < len(completion_times) else None,
            }
        )
    return rows


def summarize_series(series: pd.Series | np.ndarray) -> dict[str, float | None]:
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: None for key in ["count", "mean", "rms", "std", "min", "max", "p50", "p95", "p99", "final"]}
    return {
        "count": int(values.size),
        "mean": safe_float(np.mean(values)),
        "rms": safe_float(np.sqrt(np.mean(values**2))),
        "std": safe_float(np.std(values)),
        "min": safe_float(np.min(values)),
        "max": safe_float(np.max(values)),
        "p50": safe_float(np.percentile(values, 50)),
        "p95": safe_float(np.percentile(values, 95)),
        "p99": safe_float(np.percentile(values, 99)),
        "final": safe_float(values[-1]),
    }


def write_metrics(metrics: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []

    def flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                flatten(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                flatten(f"{prefix}.{idx}" if prefix else str(idx), child)
        else:
            rows.append({"metric": prefix, "value": value})

    flatten("", metrics)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def plot_results_dashboard(logs: pd.DataFrame, summary: dict[str, object], metrics: dict[str, Any], output_dir: Path) -> Path:
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, height_ratios=[0.9, 1.15, 1.2, 1.2])
    fig.suptitle("Mission Results Executive Dashboard", fontweight="bold", y=0.99)

    kpis = [
        ("Targets", f"{summary.get('targets_completed', '—')}/{summary.get('target_count', '—')}", "Completed targets", COLORS["green"]),
        ("Duration", format_seconds(float_or_nan(summary.get("mission_duration_s"))), "Mission elapsed time", COLORS["blue"]),
        ("Complete by", format_seconds(float_or_nan(nested(metrics, ["mission", "all_targets_completed_time_s"]))), "All targets complete", COLORS["green"]),
        ("MPC success", format_percent(float_or_nan(summary.get("mpc_solve_success_rate"))), "Solve success rate", COLORS["teal"]),
        (
            "Estimator RMS",
            format_value(nested(metrics, ["state_estimation_error", "attitude_error_deg", "rms"]), "deg", 4),
            "Attitude error RMS",
            COLORS["purple"],
        ),
        ("Max torque", format_value(summary.get("torque_max_norm_nm"), "Nm", 3), "Peak command norm", COLORS["orange"]),
        (
            "Sun margin",
            format_value(summary.get("sun_keepout_margin_min_deg"), "deg", 1, fixed=True),
            "Minimum active keepout margin",
            COLORS["red"] if float_or_nan(summary.get("sun_keepout_margin_min_deg")) < 0 else COLORS["green"],
        ),
    ]
    for idx, (title, value, subtitle, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, idx % 4] if idx < 4 else gs[1, idx - 4])
        draw_kpi_card(ax, title, value, subtitle, color)

    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    ax = fig.add_subplot(gs[2, :2])
    plot_observation_progress(ax, summary)
    ax = fig.add_subplot(gs[2, 2:])
    plot_mode_occupancy_bars(ax, summary)
    ax = fig.add_subplot(gs[3, :2])
    plot_keepout_summary(ax, summary)
    ax = fig.add_subplot(gs[3, 2:])
    plot_error_summary_bars(ax, metrics)

    return save_page(fig, output_dir / "01_results_dashboard.png")


def plot_observation_progress(ax: plt.Axes, summary: dict[str, object]) -> None:
    obs = np.asarray(summary.get("observation_time_by_target_s", []), dtype=float)
    req = np.asarray(summary.get("required_observation_time_by_target_s", []), dtype=float)
    if len(obs) == 0:
        ax.text(0.5, 0.5, "No observation data", ha="center", va="center")
        return
    pct = np.divide(obs, req, out=np.zeros_like(obs), where=req > 0) * 100.0
    y = np.arange(len(obs))
    ax.barh(y, np.maximum(req, obs), color="#E5E7EB", label="required")
    ax.barh(y, obs, color=COLORS["green"], label="observed")
    for yi, observed, required, percent in zip(y, obs, req, pct):
        ax.text(max(observed, required) * 1.01, yi, f"{observed:.1f}/{required:.1f}s ({percent:.0f}%)", va="center", fontsize=9)
    ax.set_yticks(y, [f"Target {idx}" for idx in y])
    ax.set_title("Observation completion")
    ax.set_xlabel("Observation time [s]")
    ax.legend(loc="lower right", frameon=True)
    clean_axes(ax)


def plot_mode_occupancy_bars(ax: plt.Axes, summary: dict[str, object]) -> None:
    mode_seconds = summary.get("planner_mode_seconds", {})
    if not isinstance(mode_seconds, dict) or not mode_seconds:
        ax.text(0.5, 0.5, "No planner mode data", ha="center", va="center")
        return
    labels = list(mode_seconds.keys())
    values = np.asarray([float(mode_seconds[label]) for label in labels], dtype=float)
    pct = 100.0 * values / values.sum() if values.sum() else values
    order = np.argsort(pct)
    ax.barh(np.arange(len(labels)), pct[order], color=[SERIES_COLORS[i % len(SERIES_COLORS)] for i in order])
    ax.set_yticks(np.arange(len(labels)), [labels[i].replace("_", " ").title() for i in order])
    for yi, value, sec in zip(np.arange(len(labels)), pct[order], values[order]):
        ax.text(value + 0.7, yi, f"{value:.1f}% ({sec:.0f}s)", va="center", fontsize=9)
    ax.set_xlim(0, max(50, float(np.nanmax(pct)) * 1.2))
    ax.set_xlabel("Mission time [%]")
    ax.set_title("Planner mode occupancy")
    clean_axes(ax)


def plot_keepout_summary(ax: plt.Axes, summary: dict[str, object]) -> None:
    labels = ["Sun"]
    values = [float_or_nan(summary.get("sun_keepout_margin_min_deg"))]
    colors = [COLORS["green"] if value >= 0 else COLORS["red"] for value in values]
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color=COLORS["red"], linestyle="--", linewidth=1.2, label="safety boundary")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value / 2.0,
            f"{value:.1f}°",
            ha="center",
            va="center",
            fontweight="bold",
            color="white" if value >= 0 else COLORS["dark"],
        )
    finite_values = [value for value in values if np.isfinite(value)]
    if finite_values:
        upper = max(5.0, max(finite_values) * 1.18)
        lower = min(-5.0, min(finite_values) * 1.18) if min(finite_values) < 0 else -2.0
        ax.set_ylim(lower, upper)
    ax.set_ylabel("Minimum margin [deg]")
    ax.set_title("Active keepout safety margin")
    ax.legend(loc="best")
    clean_axes(ax)


def plot_error_summary_bars(ax: plt.Axes, metrics: dict[str, Any]) -> None:
    names = ["Attitude\n[deg]", "Position\n[m]", "Velocity\n[m/s]", "Rate\n[rad/s]"]
    keys = ["attitude_error_deg", "position_error_m", "velocity_error_mps", "angular_rate_error_radps"]
    rms = [nested(metrics, ["state_estimation_error", key, "rms"]) for key in keys]
    p95 = [nested(metrics, ["state_estimation_error", key, "p95"]) for key in keys]
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, rms, width=width, label="RMS", color=COLORS["blue"])
    ax.bar(x + width / 2, p95, width=width, label="95th percentile", color=COLORS["orange"])
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xticks(x, names)
    ax.set_title("Estimator error summary")
    ax.set_ylabel("Error magnitude (mixed units)")
    ax.legend(loc="best")
    clean_axes(ax)


def plot_planner_timeline(logs: pd.DataFrame, summary: dict[str, object], output_dir: Path) -> Path:
    time = logs["time"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Planner Timeline and Observation Progress", fontweight="bold")

    mode_series = planner_mode_series(logs)
    axes[0].step(time, mode_series.to_numpy(dtype=float), where="post", color=COLORS["blue"])
    apply_planner_mode_ticks(axes[0], logs, mode_series)
    axes[0].set_ylabel("Planner mode")
    axes[0].set_title("Guidance/planner mode")
    shade_observing(axes[0], time, logs)

    target = logs["current_target_idx"].fillna(-1).to_numpy()
    axes[1].step(time, target, where="post", color=COLORS["purple"])
    axes[1].set_ylabel("Target")
    axes[1].set_title("Active target")
    shade_observing(axes[1], time, logs)

    target_cols = sorted([col for col in logs.columns if col.startswith("target_") and col.endswith("_obs_time")])
    for idx, col in enumerate(target_cols):
        axes[2].plot(time, logs[col], label=col.replace("target_", "T").replace("_obs_time", ""), color=SERIES_COLORS[idx % len(SERIES_COLORS)])
    req = np.asarray(summary.get("required_observation_time_by_target_s", []), dtype=float)
    if len(req):
        axes[2].axhline(float(req[0]), color=COLORS["dark"], linestyle="--", linewidth=1, label="required")
    axes[2].set_title("Accumulated observation time")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Observation [s]")
    axes[2].legend(loc="lower right", ncols=4)

    for ax in axes:
        clean_axes(ax)
    return save_page(fig, output_dir / "02_planner_timeline.png")


def plot_pointing_and_keepout(logs: pd.DataFrame, output_dir: Path) -> Path:
    time = logs["time"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    fig.suptitle("Pointing Performance and Safety Geometry", fontweight="bold")

    plot_if_present(axes[0, 0], time, logs, "derived_goal_angle_error_deg", "Goal attitude error", color=COLORS["blue"])
    plot_if_present(axes[0, 0], time, logs, "active_target_boresight_angle_deg", "Boresight-target angle", color=COLORS["orange"])
    axes[0, 0].set_ylabel("Angle [deg]")
    axes[0, 0].set_yscale("symlog", linthresh=0.01)
    axes[0, 0].set_title("Pointing error (log-scaled near zero)")
    axes[0, 0].legend(loc="best")

    plot_if_present(axes[0, 1], time, logs, "active_target_sun_angle_deg", "Sun angle", color=COLORS["blue"])
    plot_if_present(axes[0, 1], time, logs, "active_target_earth_angle_deg", "Earth angle", color=COLORS["orange"])
    axes[0, 1].set_ylabel("Angle [deg]")
    axes[0, 1].set_title("Target separation from keepout bodies")
    axes[0, 1].legend(loc="best")

    plot_if_present(axes[1, 0], time, logs, "active_target_sun_keepout_margin_deg", "Sun margin", color=COLORS["green"])
    axes[1, 0].axhline(0.0, color=COLORS["red"], linestyle="--", linewidth=1.2, label="violation threshold")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Margin [deg]")
    axes[1, 0].set_title("Keepout margins")
    axes[1, 0].legend(loc="best")

    observing = logs["guidance_mode"].eq("OBSERVING") if "guidance_mode" in logs else pd.Series(False, index=logs.index)
    axes[1, 1].fill_between(time, 0, 1, where=observing.to_numpy(), step="post", alpha=0.45, color=COLORS["green"])
    axes[1, 1].set_title("Observing windows")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_yticks([0, 1], ["No", "Yes"])

    for ax in axes.ravel():
        clean_axes(ax)
    return save_page(fig, output_dir / "03_pointing_keepout.png")


def plot_mpc_control(logs: pd.DataFrame, summary: dict[str, object], metrics: dict[str, Any], output_dir: Path) -> Path:
    time = logs["time"].to_numpy()
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.0, 0.9])
    fig.suptitle("MPC Control Summary", fontweight="bold")

    ax = fig.add_subplot(gs[0, :2])
    for col, label, color in [("control_tx", "Tx", COLORS["blue"]), ("control_ty", "Ty", COLORS["orange"]), ("control_tz", "Tz", COLORS["green"] )]:
        plot_if_present(ax, time, logs, col, label, color=color)
    ax.set_ylabel("Torque [Nm]")
    ax.set_title("Torque commands")
    ax.legend(loc="best")
    shade_observing(ax, time, logs)
    clean_axes(ax)

    ax = fig.add_subplot(gs[1, :2], sharex=ax)
    plot_if_present(ax, time, logs, "derived_torque_norm_nm", "Torque norm", color=COLORS["blue"])
    ax.set_ylabel("Norm [Nm]")
    ax.set_title("Torque command norm")
    shade_observing(ax, time, logs)
    clean_axes(ax)

    ax = fig.add_subplot(gs[0:2, 2])
    ax.axis("off")
    mpc = metrics.get("mpc_control", {})
    rows = [
        ("RMS torque", format_value(nested(metrics, ["mpc_control", "torque_norm_nm", "rms"]), "Nm", 4)),
        ("Mean torque", format_value(nested(metrics, ["mpc_control", "torque_norm_nm", "mean"]), "Nm", 4)),
        ("Peak torque", format_value(nested(metrics, ["mpc_control", "torque_norm_nm", "max"]), "Nm", 3)),
        ("Peak time", format_value(mpc.get("peak_torque_time_s"), "s", 1)),
        ("Torque integral", format_value(mpc.get("torque_integral_nm_s"), "Nm·s", 1)),
        ("MPC updates", str(nested(metrics, ["mpc_control", "mpc_updated", "count"], "—"))),
        ("Solve success", format_percent(float_or_nan(summary.get("mpc_solve_success_rate")))),
        ("Fallbacks", str(nested(metrics, ["mpc_control", "mpc_fallback_used", "count"], "—"))),
    ]
    draw_metric_table(ax, "Control metrics", rows)

    ax = fig.add_subplot(gs[2, :])
    status_labels = [("mpc_updated", "MPC update", COLORS["blue"]), ("mpc_solve_success", "Solve success", COLORS["green"]), ("mpc_fallback_used", "Fallback", COLORS["red"])]
    for idx, (col, label, color) in enumerate(status_labels):
        if col in logs:
            ax.step(time, logs[col].astype(int) + idx * 1.15, where="post", label=label, color=color)
    ax.set_title("MPC status flags (vertically offset for readability)")
    ax.set_xlabel("Time [s]")
    ax.set_yticks([0, 1.15, 2.3], [label for _, label, _ in status_labels])
    ax.legend(loc="upper right", ncols=3)
    clean_axes(ax)

    return save_page(fig, output_dir / "04_mpc_control.png", tight=False)


def plot_estimator_errors(logs: pd.DataFrame, summary: dict[str, object], metrics: dict[str, Any], output_dir: Path) -> Path:
    time = logs["time"].to_numpy()
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.1, 0.95])
    fig.suptitle("Estimator Accuracy Summary", fontweight="bold")

    panels = [
        ("derived_attitude_estimation_error_deg", "Attitude estimation error", "Error [deg]", COLORS["blue"]),
        ("derived_position_error_m", "Position estimation error", "Error [m]", COLORS["orange"]),
        ("derived_velocity_error_mps", "Velocity error", "Error [m/s]", COLORS["green"]),
        ("derived_angular_rate_error_radps", "Angular-rate error", "Error [rad/s]", COLORS["purple"]),
    ]
    for idx, (col, title, ylabel, color) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        plot_if_present(ax, time, logs, col, title, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if idx >= 2:
            ax.set_xlabel("Time [s]")
        shade_observing(ax, time, logs)
        clean_axes(ax)

    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    rows = []
    for label, key, unit, digits in [
        ("Attitude RMS", "attitude_error_deg", "deg", 5),
        ("Attitude max", "attitude_error_deg", "deg", 5),
        ("Position RMS", "position_error_m", "m", 2),
        ("Position max", "position_error_m", "m", 2),
        ("Velocity RMS", "velocity_error_mps", "m/s", 3),
        ("Velocity max", "velocity_error_mps", "m/s", 3),
        ("Rate RMS", "angular_rate_error_radps", "rad/s", 6),
        ("Rate max", "angular_rate_error_radps", "rad/s", 6),
    ]:
        stat = "max" if label.endswith("max") else "rms"
        rows.append((label, format_value(nested(metrics, ["state_estimation_error", key, stat]), unit, digits)))
    rows.extend(
        [
            ("GNSS updates", str(summary.get("gnss_update_count", "—"))),
            ("Star tracker updates", str(summary.get("star_tracker_update_count", "—"))),
        ]
    )
    draw_metric_table(ax, "Numeric estimator results", rows)

    return save_page(fig, output_dir / "05_estimator_errors.png", tight=False)


def plot_position_velocity_comparison(logs: pd.DataFrame, output_dir: Path) -> Path:
    time = logs["time"].to_numpy()
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    fig.suptitle("True vs Estimated Translation State", fontweight="bold")
    components = ["x", "y", "z"]
    for row, comp in enumerate(components):
        plot_pair(axes[row, 0], time, logs, f"true_position_{comp}", f"estimated_position_{comp}", f"Position {comp.upper()}")
        axes[row, 0].set_ylabel("Position [m]")
        plot_pair(axes[row, 1], time, logs, f"true_velocity_{comp}", f"estimated_velocity_{comp}", f"Velocity {comp.upper()}")
        axes[row, 1].set_ylabel("Velocity [m/s]")
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    for ax in axes.ravel():
        clean_axes(ax)
    return save_page(fig, output_dir / "06_translation_state.png")


def plot_attitude_rate_comparison(logs: pd.DataFrame, output_dir: Path) -> Path:
    time = logs["time"].to_numpy()
    fig, axes = plt.subplots(4, 2, figsize=(15, 13), sharex=True)
    fig.suptitle("True vs Estimated Attitude and Rate", fontweight="bold")
    quat_components = ["w", "x", "y", "z"]
    for row, comp in enumerate(quat_components):
        plot_pair(axes[row, 0], time, logs, f"true_quat_{comp}", f"estimated_quat_{comp}", f"Quaternion {comp}")
        axes[row, 0].set_ylabel("Quaternion")
    for row, comp in enumerate(["x", "y", "z"]):
        plot_pair(axes[row, 1], time, logs, f"true_omega_{comp}", f"estimated_omega_{comp}", f"Omega {comp.upper()}")
        axes[row, 1].set_ylabel("Rate [rad/s]")
    plot_if_present(axes[3, 1], time, logs, "derived_true_angular_rate_norm_radps", "True norm", color=COLORS["blue"])
    plot_if_present(axes[3, 1], time, logs, "derived_estimated_angular_rate_norm_radps", "Estimated norm", linestyle="--", color=COLORS["orange"])
    axes[3, 1].set_title("Angular-rate norm")
    axes[3, 1].set_ylabel("Norm [rad/s]")
    axes[3, 1].legend(loc="best")
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    for ax in axes.ravel():
        clean_axes(ax)
    return save_page(fig, output_dir / "07_attitude_rate_state.png")


def plot_mekf_uncertainty(logs: pd.DataFrame, metrics: dict[str, Any], output_dir: Path) -> Path:
    time = logs["time"].to_numpy()
    groups = [
        ("Position", ["position_x", "position_y", "position_z"], "m"),
        ("Velocity", ["velocity_x", "velocity_y", "velocity_z"], "m/s"),
        ("Attitude", ["attitude_x", "attitude_y", "attitude_z"], "rad"),
        ("Gyro Bias", ["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"], "rad/s"),
        ("Accel Bias", ["accel_bias_x", "accel_bias_y", "accel_bias_z"], "m/s²"),
    ]
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    fig.suptitle("MEKF 3-Sigma Uncertainty Summary", fontweight="bold")

    for idx, (title, labels, unit) in enumerate(groups):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        for label_idx, label in enumerate(labels):
            col = f"derived_mekf_three_sigma_{label}"
            plot_if_present(ax, time, logs, col, label.replace("_", " "), color=SERIES_COLORS[label_idx])
        ax.set_title(f"{title} uncertainty")
        ax.set_ylabel(f"3σ [{unit}]")
        if idx >= 3:
            ax.set_xlabel("Time [s]")
        ax.legend(loc="best", ncols=3)
        clean_axes(ax)

    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    rows = []
    for label in ["position_x", "velocity_x", "attitude_x", "gyro_bias_x", "accel_bias_x"]:
        rows.append((f"{label.replace('_', ' ')} final", format_value(nested(metrics, ["mekf_uncertainty", label, "final"]), "", 3)))
        rows.append((f"{label.replace('_', ' ')} max", format_value(nested(metrics, ["mekf_uncertainty", label, "max"]), "", 3)))
    draw_metric_table(ax, "Representative uncertainty numbers", rows)
    return save_page(fig, output_dir / "08_mekf_uncertainty.png", tight=False)


def plot_interactive_pages(logs: pd.DataFrame, summary: dict[str, object], metrics: dict[str, Any], output_dir: Path) -> list[Path]:
    try:
        import plotly.graph_objects as go  # type: ignore[import-not-found]
        from plotly.subplots import make_subplots  # type: ignore[import-not-found]
    except ImportError:
        print("Plotly is not installed; skipping interactive HTML output. Install plotly>=5 to enable --interactive.")
        return []

    paths: list[Path] = []
    time = logs["time"].to_numpy(dtype=float)
    reduced = downsample_frame(logs, max_points=12000)
    rt = reduced["time"].to_numpy(dtype=float)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Estimator errors", "MPC control norm"))
    for col, name in [
        ("derived_attitude_estimation_error_deg", "Attitude error [deg]"),
        ("derived_position_error_m", "Position error [m]"),
        ("derived_velocity_error_mps", "Velocity error [m/s]"),
        ("derived_angular_rate_error_radps", "Rate error [rad/s]"),
    ]:
        if col in reduced:
            fig.add_trace(go.Scatter(x=rt, y=reduced[col], mode="lines", name=name), row=1, col=1)
    if "derived_torque_norm_nm" in reduced:
        fig.add_trace(go.Scatter(x=rt, y=reduced["derived_torque_norm_nm"], mode="lines", name="Torque norm [Nm]"), row=2, col=1)
    fig.update_layout(title="Interactive Estimator and MPC Summary", template="plotly_white", hovermode="x unified")
    path = output_dir / "interactive_estimator_mpc.html"
    fig.write_html(path)
    paths.append(path)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Pointing and keepout", "Planner state"))
    for col, name in [
        ("derived_goal_angle_error_deg", "Goal angle error [deg]"),
        ("active_target_sun_keepout_margin_deg", "Sun keepout margin [deg]"),
    ]:
        if col in reduced:
            fig.add_trace(go.Scatter(x=rt, y=reduced[col], mode="lines", name=name), row=1, col=1)
    if "guidance_mode_value" in reduced:
        fig.add_trace(go.Scatter(x=rt, y=reduced["guidance_mode_value"], mode="lines", name="Guidance mode"), row=2, col=1)
    if "current_target_idx" in reduced:
        fig.add_trace(go.Scatter(x=rt, y=reduced["current_target_idx"], mode="lines", name="Target index"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    fig.update_layout(title="Interactive Pointing, Keepout, and Planner Timeline", template="plotly_white", hovermode="x unified")
    path = output_dir / "interactive_pointing_planner.html"
    fig.write_html(path)
    paths.append(path)

    metrics_html = output_dir / "interactive_metrics_summary.html"
    metric_rows = flatten_metrics(metrics)
    table = go.Figure(
        data=[go.Table(header={"values": ["Metric", "Value"]}, cells={"values": [[row["metric"] for row in metric_rows], [row["value"] for row in metric_rows]]})]
    )
    table.update_layout(title="Structured Mission Metrics")
    table.write_html(metrics_html)
    paths.append(metrics_html)

    _ = summary, time
    return paths


def plot_pair(ax: plt.Axes, time: np.ndarray, logs: pd.DataFrame, true_col: str, est_col: str, title: str) -> None:
    plot_if_present(ax, time, logs, true_col, "true", color=COLORS["blue"])
    plot_if_present(ax, time, logs, est_col, "estimated", linestyle="--", color=COLORS["orange"])
    ax.set_title(title)
    ax.legend(loc="best")


def planner_mode_series(logs: pd.DataFrame) -> pd.Series:
    if "guidance_mode" in logs:
        mode_order = {mode: idx for idx, mode in enumerate(pd.unique(logs["guidance_mode"].dropna()))}
        return logs["guidance_mode"].map(mode_order).fillna(-1)
    return logs["guidance_mode_value"]


def apply_planner_mode_ticks(ax: plt.Axes, logs: pd.DataFrame, mode_series: pd.Series) -> None:
    if "guidance_mode" not in logs:
        return
    tick_rows = pd.DataFrame({"mode": logs["guidance_mode"], "value": mode_series}).dropna().drop_duplicates("mode")
    tick_rows = tick_rows.sort_values("value")
    labels = [str(mode).replace("_", " ").title() for mode in tick_rows["mode"]]
    ax.set_yticks(tick_rows["value"].to_numpy(dtype=float), labels)


def plot_if_present(
    ax: plt.Axes,
    time: np.ndarray,
    logs: pd.DataFrame,
    column: str,
    label: str,
    **kwargs: Any,
) -> None:
    if column not in logs:
        return
    plot_time = time
    values = logs[column].to_numpy(dtype=float)
    if len(values) > 6000:
        idx = downsample_indices(values, max_points=6000)
        plot_time = time[idx]
        values = values[idx]
    ax.plot(plot_time, values, label=label, **kwargs)


def downsample_indices(values: np.ndarray, max_points: int = 6000) -> np.ndarray:
    if len(values) <= max_points:
        return np.arange(len(values))
    base = np.linspace(0, len(values) - 1, max_points, dtype=int)
    finite = np.where(np.isfinite(values))[0]
    if finite.size:
        extrema = np.unique(np.concatenate([finite[np.argsort(values[finite])[:100]], finite[np.argsort(values[finite])[-100:]]]))
        base = np.unique(np.concatenate([base, extrema]))
    return base


def downsample_frame(logs: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(logs) <= max_points:
        return logs
    idx = np.linspace(0, len(logs) - 1, max_points, dtype=int)
    return logs.iloc[np.unique(idx)]


def draw_kpi_card(ax: plt.Axes, title: str, value: str, subtitle: str, color: str) -> None:
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white", edgecolor="#E5E7EB", linewidth=1.2))
    ax.add_patch(plt.Rectangle((0, 0), 0.025, 1, transform=ax.transAxes, facecolor=color, edgecolor=color))
    ax.text(0.08, 0.72, title, fontsize=10, fontweight="bold", color=COLORS["gray"], transform=ax.transAxes)
    ax.text(0.08, 0.38, value, fontsize=18, fontweight="bold", color=COLORS["dark"], transform=ax.transAxes)
    ax.text(0.08, 0.14, subtitle, fontsize=9, color=COLORS["gray"], transform=ax.transAxes)


def draw_metric_table(ax: plt.Axes, title: str, rows: Iterable[tuple[str, str]]) -> None:
    ax.text(0.0, 0.98, title, fontsize=12, fontweight="bold", color=COLORS["dark"], va="top", transform=ax.transAxes)
    y = 0.90
    for name, value in rows:
        ax.text(0.0, y, name, fontsize=9.5, color=COLORS["gray"], va="top", transform=ax.transAxes)
        ax.text(1.0, y, value, fontsize=9.5, color=COLORS["dark"], va="top", ha="right", fontweight="bold", transform=ax.transAxes)
        y -= 0.075


def shade_observing(ax: plt.Axes, time: np.ndarray, logs: pd.DataFrame) -> None:
    if "guidance_mode" not in logs:
        return
    observing = logs["guidance_mode"].eq("OBSERVING").to_numpy().tolist()
    if len(observing) != len(time):
        return
    ax.fill_between(time, 0, 1, where=observing, transform=ax.get_xaxis_transform(), color=COLORS["green"], alpha=0.07, step="post", linewidth=0)


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.45)


def save_page(fig: plt.Figure, path: Path, *, tight: bool = True) -> Path:
    if tight:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def flatten_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                flatten(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                flatten(f"{prefix}.{idx}" if prefix else str(idx), child)
        else:
            rows.append({"metric": prefix, "value": value})

    flatten("", metrics)
    return rows


def nested(data: dict[str, Any], keys: list[str], default: Any = np.nan) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def format_seconds(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if value >= 3600:
        return f"{value / 3600:.2f} h"
    return f"{value:.1f} s"


def format_percent(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{100.0 * value:.1f}%"


def format_value(value: object, unit: str, digits: int = 2, *, fixed: bool = False) -> str:
    number = float_or_nan(value)
    if not np.isfinite(number):
        return "—"
    suffix = f" {unit}" if unit else ""
    if fixed:
        return f"{number:.{digits}f}{suffix}"
    return f"{number:.{digits}g}{suffix}"


def safe_float(value: object) -> float | None:
    number = float_or_nan(value)
    return float(number) if np.isfinite(number) else None


def float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    main()
