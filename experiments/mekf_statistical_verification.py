"""Run Monte Carlo error, NEES, and NIS verification for the MEKF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mpc_spacecraft.analysis.mekf_verification import (
    InnovationConsistencyResult,
    MEKFVerificationConfig,
    MEKFVerificationResult,
    run_mekf_verification,
)


DEFAULT_OUTPUT_DIR = Path("./experiments/results/mekf_verification")
ERROR_LABELS = (
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "velocity_x_m_s",
    "velocity_y_m_s",
    "velocity_z_m_s",
    "attitude_x_rad",
    "attitude_y_rad",
    "attitude_z_rad",
    "gyro_bias_x_rad_s",
    "gyro_bias_y_rad_s",
    "gyro_bias_z_rad_s",
    "accel_bias_x_m_s2",
    "accel_bias_y_m_s2",
    "accel_bias_z_m_s2",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--gnss-period", type=float, default=1.0)
    parser.add_argument("--star-tracker-period", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = MEKFVerificationConfig(
        run_count=args.runs,
        duration_s=args.duration,
        dt_s=args.dt,
        gnss_period_s=args.gnss_period,
        star_tracker_period_s=args.star_tracker_period,
        base_seed=args.seed,
        confidence_level=args.confidence,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_mekf_verification(config, show_progress=True)

    _time_history(result).to_csv(args.output_dir / "time_history.csv", index=False)
    _innovation_history(result).to_csv(
        args.output_dir / "innovation_history.csv", index=False
    )
    result.run_summaries.to_csv(args.output_dir / "run_summary.csv", index=False)
    _plot_attitude_error(result, args.output_dir / "attitude_error.png")
    _plot_nees(result, args.output_dir / "anees.png")
    _plot_nis(result, args.output_dir / "nis.png")

    summary = _build_summary(result)
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(
        "MEKF verification complete: "
        f"mean ANEES={summary['consistency']['mean_anees']:.3f}, "
        f"attitude RMS={summary['accuracy']['overall_attitude_rms_deg']:.6g} deg"
    )
    print(f"Saved artifacts to {args.output_dir}")


def _time_history(result: MEKFVerificationResult) -> pd.DataFrame:
    data: dict[str, Any] = {
        "time_s": result.times,
        "attitude_rmse_rad": result.attitude_rmse_rad,
        "attitude_rmse_deg": np.rad2deg(result.attitude_rmse_rad),
        "average_nees": result.average_nees,
        "nees_lower": result.nees_lower_bound,
        "nees_upper": result.nees_upper_bound,
        "nees_sample_count": result.nees_sample_count,
    }
    for index, label in enumerate(ERROR_LABELS):
        data[f"mean_error_{label}"] = result.mean_error[:, index]
        data[f"rmse_{label}"] = result.component_rmse[:, index]
    return pd.DataFrame(data)


def _innovation_history(result: MEKFVerificationResult) -> pd.DataFrame:
    frames = [
        _innovation_frame("star_tracker", result.star_tracker_nis),
        _innovation_frame("gnss", result.gnss_nis),
    ]
    return pd.concat(frames, ignore_index=True)


def _innovation_frame(
    sensor: str, result: InnovationConsistencyResult
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor": sensor,
            "time_s": result.times,
            "average_nis": result.average_nis,
            "nis_lower": result.lower_bound,
            "nis_upper": result.upper_bound,
            "sample_count": result.sample_count,
            "measurement_dimension": result.measurement_dimension,
        }
    )


def _build_summary(result: MEKFVerificationResult) -> dict[str, Any]:
    confidence_percent = 100.0 * result.config.confidence_level
    nees_inside = (
        (result.average_nees >= result.nees_lower_bound)
        & (result.average_nees <= result.nees_upper_bound)
    )
    star_inside = _inside_bounds(result.star_tracker_nis)
    gnss_inside = _inside_bounds(result.gnss_nis)
    return {
        "schema_version": "mpc_spacecraft_mekf_verification_v1",
        "configuration": {
            "run_count": result.config.run_count,
            "duration_s": result.config.duration_s,
            "dt_s": result.config.dt_s,
            "gnss_period_s": result.config.gnss_period_s,
            "star_tracker_period_s": result.config.star_tracker_period_s,
            "base_seed": result.config.base_seed,
            "confidence_level": result.config.confidence_level,
        },
        "error_definition": {
            "state_order": ["dr", "dv", "dtheta", "dbg", "dba"],
            "attitude": "rotation_vector(conjugate(q_estimate) * q_truth)",
        },
        "accuracy": {
            "overall_attitude_rms_deg": float(
                np.rad2deg(np.sqrt(np.mean(result.attitude_rmse_rad**2)))
            ),
            "final_attitude_rmse_deg": float(
                np.rad2deg(result.attitude_rmse_rad[-1])
            ),
            "final_mean_attitude_error_deg": np.rad2deg(
                result.mean_error[-1, 6:9]
            ).tolist(),
        },
        "consistency": {
            "confidence_percent": confidence_percent,
            "expected_nees": 15,
            "mean_anees": float(np.mean(result.average_nees)),
            "anees_epoch_fraction_inside_bounds": float(np.mean(nees_inside)),
            "expected_star_tracker_nis": 3,
            "mean_star_tracker_anis": float(
                np.mean(result.star_tracker_nis.average_nis)
            ),
            "star_tracker_epoch_fraction_inside_bounds": float(
                np.mean(star_inside)
            ),
            "expected_gnss_nis": 6,
            "mean_gnss_anis": float(np.mean(result.gnss_nis.average_nis)),
            "gnss_epoch_fraction_inside_bounds": float(np.mean(gnss_inside)),
            "interpretation": (
                "Epoch fractions are descriptive because adjacent samples are "
                "temporally correlated."
            ),
        },
        "artifacts": [
            "time_history.csv",
            "innovation_history.csv",
            "run_summary.csv",
            "attitude_error.png",
            "anees.png",
            "nis.png",
        ],
    }


def _inside_bounds(result: InnovationConsistencyResult) -> np.ndarray:
    return (result.average_nis >= result.lower_bound) & (
        result.average_nis <= result.upper_bound
    )


def _plot_attitude_error(result: MEKFVerificationResult, path: Path) -> None:
    plt = _pyplot()
    scale = np.rad2deg(1.0) * 3600.0
    history = slice(1, None)
    initial_mean_arcsec = scale * result.mean_error[0, 6:9]
    initial_rmse_arcsec = scale * result.attitude_rmse_rad[0]
    fig, (mean_ax, rmse_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for offset, axis in enumerate("xyz"):
        mean_ax.plot(
            result.times[history],
            scale * result.mean_error[history, 6 + offset],
            label=rf"$\overline{{\delta\theta_{axis}}}$",
        )
    mean_ax.axhline(0.0, color="black", linewidth=0.8)
    mean_ax.set_ylabel("Mean error [arcsec]")
    mean_ax.legend()
    rmse_ax.plot(
        result.times[history],
        scale * result.attitude_rmse_rad[history],
        color="tab:red",
    )
    rmse_ax.set_xlabel("Time [s]")
    rmse_ax.set_ylabel("Rotation-vector RMSE [arcsec]")
    for axis in (mean_ax, rmse_ax):
        axis.grid(True, alpha=0.3)
    fig.suptitle("MEKF Monte Carlo Attitude Estimation Error")
    initial_components = ", ".join(
        f"{axis}={value:.1f}" for axis, value in zip("xyz", initial_mean_arcsec)
    )
    fig.text(
        0.5,
        0.01,
        (
            "Post-first-update history shown. "
            f"Initial mean error [arcsec]: {initial_components}; "
            f"initial RMSE: {initial_rmse_arcsec:.1f} arcsec."
        ),
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_nees(result: MEKFVerificationResult, path: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(
        result.times,
        result.nees_lower_bound,
        result.nees_upper_bound,
        color="tab:blue",
        alpha=0.18,
        label=f"{100 * result.config.confidence_level:.0f}% bounds",
    )
    ax.plot(result.times, result.average_nees, color="tab:blue", label="Average NEES")
    ax.axhline(15.0, color="black", linestyle="--", label="Expected value (15)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Average NEES")
    ax.set_title("MEKF Full-State Covariance Consistency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_nis(result: MEKFVerificationResult, path: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    entries = (
        (axes[0], "Star tracker", result.star_tracker_nis),
        (axes[1], "GNSS", result.gnss_nis),
    )
    for axis, label, innovation in entries:
        axis.fill_between(
            innovation.times,
            innovation.lower_bound,
            innovation.upper_bound,
            alpha=0.18,
            color="tab:purple",
            label=f"{100 * result.config.confidence_level:.0f}% bounds",
        )
        axis.plot(innovation.times, innovation.average_nis, color="tab:purple")
        axis.axhline(
            innovation.measurement_dimension,
            color="black",
            linestyle="--",
            label=f"Expected value ({innovation.measurement_dimension})",
        )
        axis.set_ylabel("Average NIS")
        axis.set_title(label)
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("MEKF Measurement Innovation Consistency", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
