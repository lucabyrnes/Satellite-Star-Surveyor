"""Run single-target controller convergence sweeps with MEKF feedback."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from mpc_spacecraft.analysis.single_target import (
    SingleTargetCaseResult,
    make_single_target_config,
    run_single_target_case,
)
from mpc_spacecraft.config import ClosedLoopTestConfig

DEFAULT_OUTPUT_DIR = Path("./experiments/results/single_target_convergence")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # argparse's default matcher does not recognize negative scientific
    # notation (for example, -5e-4) as a numeric argument.
    parser._negative_number_matcher = re.compile(  # type: ignore[attr-defined]
        r"^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-duration", type=float, default=300.0)
    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--mpc-dt", type=float, default=5.0)
    parser.add_argument(
        "--disturbance-rejection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "enable the constant-torque observer and offset-free MPC "
            "(default: disabled)"
        ),
    )
    parser.add_argument(
        "--disturbance-torque-nm",
        type=float,
        nargs=3,
        metavar=("TX", "TY", "TZ"),
        default=None,
        help=(
            "constant body-frame disturbance torque [N·m] as TX TY TZ "
            "(default: use the configured environment torque)"
        ),
    )
    parser.add_argument(
        "--attitude-offsets-deg",
        type=float,
        nargs="+",
        default=[5.0, 15.0, 30.0, 60.0],
    )
    parser.add_argument(
        "--angular-rates-deg-s",
        "--angular-rates",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5],
    )
    return parser.parse_args(argv)


def make_base_config(
    *,
    disturbance_rejection: bool,
    disturbance_torque_nm: list[float] | np.ndarray | None = None,
) -> ClosedLoopTestConfig:
    """Build the shared sweep configuration from CLI feature toggles."""
    config = ClosedLoopTestConfig.defaults(show_progress=False)
    environment = config.environment
    if disturbance_torque_nm is not None:
        environment = replace(
            environment,
            disturbance_bias=np.asarray(disturbance_torque_nm, dtype=np.float64),
        )
    return replace(
        config,
        environment=environment,
        mpc=replace(
            config.mpc,
            enable_disturbance_rejection=disturbance_rejection,
        ),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_config = make_base_config(
        disturbance_rejection=args.disturbance_rejection,
        disturbance_torque_nm=args.disturbance_torque_nm,
    )

    attitude_cases: list[SingleTargetCaseResult] = []
    for offset in args.attitude_offsets_deg:
        label = f"{offset:g}° initial offset"
        print(f"Running attitude sweep case: {label}")
        config = make_single_target_config(
            attitude_offset_deg=offset,
            angular_rate_deg_s=0.0,
            max_duration_s=args.max_duration,
            sim_dt=args.sim_dt,
            mpc_dt=args.mpc_dt,
            base_config=base_config,
        )
        attitude_cases.append(
            run_single_target_case(f"attitude_{offset:g}_deg", label, config)
        )

    rate_cases: list[SingleTargetCaseResult] = []
    for rate in args.angular_rates_deg_s:
        label = f"{rate:g}°/s initial rate"
        print(f"Running angular-rate sweep case: {label}")
        config = make_single_target_config(
            attitude_offset_deg=0.0,
            angular_rate_deg_s=rate,
            max_duration_s=args.max_duration,
            sim_dt=args.sim_dt,
            mpc_dt=args.mpc_dt,
            base_config=base_config,
        )
        rate_cases.append(run_single_target_case(f"rate_{rate:g}_deg_s", label, config))

    attitude_metrics = _metrics_table(
        attitude_cases, "initial_attitude_offset_deg", args.attitude_offsets_deg
    )
    rate_metrics = _metrics_table(
        rate_cases, "initial_angular_rate_deg_s", args.angular_rates_deg_s
    )
    attitude_metrics.to_csv(
        args.output_dir / "initial_attitude_metrics.csv", index=False
    )
    rate_metrics.to_csv(args.output_dir / "initial_rate_metrics.csv", index=False)
    _plot_attitude_cases(
        attitude_cases, args.output_dir / "initial_attitude_convergence.png"
    )
    _plot_rate_cases(rate_cases, args.output_dir / "initial_rate_convergence.png")

    summary = {
        "schema_version": "mpc_spacecraft_single_target_convergence_v1",
        "configuration": {
            "sim_dt_s": args.sim_dt,
            "mpc_dt_s": args.mpc_dt,
            "maximum_duration_s": args.max_duration,
            "required_observation_time_s": 30.0,
            "mekf_enabled": True,
            "measurement_noise_enabled": base_config.sensors.enable_measurement_noise,
            "bias_random_walk_enabled": base_config.sensors.enable_bias_random_walk,
            "disturbance_rejection_enabled": (
                base_config.mpc.enable_disturbance_rejection
            ),
            "disturbance_torque_nm": (
                base_config.environment.disturbance_bias.tolist()
                if base_config.environment.disturbance_bias is not None
                else None
            ),
            "attitude_offsets_deg": args.attitude_offsets_deg,
            "angular_rates_deg_s": args.angular_rates_deg_s,
        },
        "attitude_sweep": attitude_metrics.to_dict(orient="records"),
        "angular_rate_sweep": rate_metrics.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved single-target convergence artifacts to {args.output_dir}")


def _metrics_table(
    cases: list[SingleTargetCaseResult], parameter_name: str, values: list[float]
) -> pd.DataFrame:
    rows = []
    for case, value in zip(cases, values):
        rows.append(
            {
                "case": case.name,
                "label": case.label,
                parameter_name: value,
                **case.metrics,
            }
        )
    return pd.DataFrame(rows)


def _plot_attitude_cases(cases: list[SingleTargetCaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for case in cases:
        time = case.logs["time"].to_numpy(dtype=np.float64)
        error = case.logs["active_target_boresight_angle_deg"].to_numpy(
            dtype=np.float64
        )
        ax.semilogy(time, np.maximum(error, 1e-6), label=case.label)
    ax.axhline(0.5, color="tab:orange", linestyle="--", label="0.5° settling threshold")
    ax.axhline(
        0.01, color="tab:red", linestyle=":", label="0.01° observation threshold"
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("True boresight pointing error [deg]")
    ax.set_title("Single-Target Convergence from Initial Attitude Error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_rate_cases(cases: list[SingleTargetCaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for case in cases:
        omega = case.logs[["true_omega_x", "true_omega_y", "true_omega_z"]].to_numpy(
            dtype=np.float64
        )
        rate_deg_s = np.rad2deg(np.linalg.norm(omega, axis=1))
        ax.plot(case.logs["time"], rate_deg_s, label=case.label)
    ax.axhline(
        0.01,
        color="tab:red",
        linestyle=":",
        label="0.01°/s observation threshold",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"True angular-rate norm [$\degree$/s]")
    ax.set_title("Single-Target Convergence from Initial Angular Velocity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


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
