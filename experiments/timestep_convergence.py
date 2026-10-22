"""Run dynamics-only convergence, then full closed-loop timestep sensitivity.

Examples:
    python experiments/timestep_convergence.py --phase all
    python experiments/timestep_convergence.py --phase dynamics --strict
    python experiments/timestep_convergence.py --phase closed-loop --duration 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mpc_spacecraft.analysis.timestep_convergence import (
    ClosedLoopConvergenceResult,
    DynamicsConvergenceResult,
    run_closed_loop_convergence,
    run_dynamics_convergence,
)
from mpc_spacecraft.config import ClosedLoopTestConfig

DEFAULT_OUTPUT_DIR = Path("./experiments/results/timestep_convergence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("dynamics", "closed-loop", "all"),
        default="all",
        help="Experiment phase to run. 'all' always runs dynamics first.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dynamics-duration", type=float, default=10.0)
    parser.add_argument(
        "--dynamics-dts", type=float, nargs="+", default=[0.08, 0.04, 0.02, 0.01]
    )
    parser.add_argument("--dynamics-reference-dt", type=float, default=0.00125)
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Closed-loop physical duration in seconds.",
    )
    parser.add_argument(
        "--closed-loop-dts", type=float, nargs="+", default=[0.05, 0.025, 0.0125]
    )
    parser.add_argument("--closed-loop-reference-dt", type=float, default=0.003125)
    parser.add_argument(
        "--minimum-dynamics-order",
        type=float,
        default=3.5,
        help="Minimum fitted attitude and rate order used by --strict.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit unsuccessfully when the measured dynamics order is below the threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": "mpc_spacecraft_timestep_convergence_v1",
        "execution_order": [],
    }
    dynamics_result: DynamicsConvergenceResult | None = None
    closed_loop_result: ClosedLoopConvergenceResult | None = None

    if args.phase in ("dynamics", "all"):
        print("[1/2] Running isolated dynamics timestep convergence...")
        dynamics_result = run_dynamics_convergence(
            timesteps=args.dynamics_dts,
            reference_dt=args.dynamics_reference_dt,
            duration=args.dynamics_duration,
        )
        dynamics_result.metrics.to_csv(
            args.output_dir / "dynamics_metrics.csv", index=False
        )
        _plot_dynamics(dynamics_result, args.output_dir / "dynamics_convergence.png")
        passed = bool(
            dynamics_result.fitted_attitude_order >= args.minimum_dynamics_order
            and dynamics_result.fitted_angular_velocity_order
            >= args.minimum_dynamics_order
        )
        summary["execution_order"].append("dynamics")
        summary["dynamics"] = {
            "duration_s": args.dynamics_duration,
            "tested_timesteps_s": args.dynamics_dts,
            "reference_timestep_s": args.dynamics_reference_dt,
            "fitted_attitude_order": dynamics_result.fitted_attitude_order,
            "fitted_angular_velocity_order": dynamics_result.fitted_angular_velocity_order,
            "principal_axis_max_error_rad": dynamics_result.principal_axis_max_error_rad,
            "minimum_order": args.minimum_dynamics_order,
            "passes_order_threshold": passed,
            "metrics": dynamics_result.metrics.to_dict(orient="records"),
        }
        print(
            "  fitted order: "
            f"attitude={dynamics_result.fitted_attitude_order:.3f}, "
            f"angular velocity={dynamics_result.fitted_angular_velocity_order:.3f}"
        )

    if args.phase in ("closed-loop", "all"):
        print("[2/2] Running deterministic full closed-loop timestep sensitivity...")
        closed_loop_result = run_closed_loop_convergence(
            base_config=ClosedLoopTestConfig.defaults(show_progress=False),
            timesteps=args.closed_loop_dts,
            reference_dt=args.closed_loop_reference_dt,
            duration=args.duration,
        )
        closed_loop_result.metrics.to_csv(
            args.output_dir / "closed_loop_metrics.csv", index=False
        )
        _plot_closed_loop(
            closed_loop_result, args.output_dir / "closed_loop_sensitivity.png"
        )
        summary["execution_order"].append("closed_loop")
        summary["closed_loop"] = {
            "duration_s": args.duration,
            "tested_timesteps_s": args.closed_loop_dts,
            "reference_timestep_s": args.closed_loop_reference_dt,
            "mpc_timestep_s": 5.0,
            "measurement_noise_enabled": False,
            "bias_random_walk_enabled": False,
            "disturbance_noise_enabled": False,
            "interpretation": (
                "Hybrid closed-loop sensitivity study; fourth-order convergence is not expected."
            ),
            "reference_mission_summary": closed_loop_result.reference_mission_summary,
            "metrics": closed_loop_result.metrics.to_dict(orient="records"),
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved timestep-convergence artifacts to {args.output_dir}")

    if args.strict and dynamics_result is not None:
        passed = summary["dynamics"]["passes_order_threshold"]
        if not passed:
            raise SystemExit(1)


def _plot_dynamics(result: DynamicsConvergenceResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = result.metrics.sort_values("sim_dt_s")
    dt = metrics["sim_dt_s"].to_numpy()
    attitude = metrics["attitude_error_max_rad"].to_numpy()
    omega = metrics["angular_velocity_error_max_radps"].to_numpy()
    ideal = attitude[-1] * (dt / dt[-1]) ** 4

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(dt, attitude, "o-", label="max attitude error [rad]")
    ax.loglog(dt, omega, "s-", label="max angular-rate error [rad/s]")
    ax.loglog(dt, ideal, "k--", label=r"ideal $O(\Delta t^4)$")
    ax.set_xlabel("Integration timestep [s]")
    ax.set_ylabel("Error relative to fine-step reference")
    ax.set_title("Isolated Rigid-Body RK4 Timestep Convergence")
    _format_timestep_axis(ax, dt)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_closed_loop(result: ClosedLoopConvergenceResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = result.metrics.sort_values("sim_dt_s")
    dt = metrics["sim_dt_s"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = (
        ("attitude_error_max_rad", "Maximum attitude difference [rad]"),
        ("angular_velocity_error_max_radps", "Maximum rate difference [rad/s]"),
        ("position_error_max_m", "Maximum position difference [m]"),
        ("torque_error_rms_nm", "RMS torque difference [N m]"),
    )
    for ax, (column, ylabel) in zip(axes.flat, panels):
        values = metrics[column].to_numpy(dtype=np.float64)
        positive = values > 0.0
        if np.any(positive):
            ax.loglog(dt[positive], values[positive], "o-")
        else:
            ax.semilogx(dt, values, "o-")
        ax.set_xlabel("Integration timestep [s]")
        ax.set_ylabel(ylabel)
        _format_timestep_axis(ax, dt)
        ax.grid(True, which="both", alpha=0.35)
    fig.suptitle("Deterministic Closed-Loop Timestep Sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _format_timestep_axis(ax: Any, timesteps: np.ndarray) -> None:
    """Show decimal timestep labels while preserving logarithmic spacing."""
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    values = np.sort(np.unique(np.asarray(timesteps, dtype=np.float64)))
    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{value:g}" for value in values]))
    ax.xaxis.set_minor_formatter(NullFormatter())


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
