"""Compare nominal single-target performance against disturbances and MOI mismatch."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mpc_spacecraft.analysis.single_target import (
    SingleTargetCaseResult,
    make_single_target_config,
    perturb_inertia,
    run_single_target_case,
    summarize_disturbance_torque,
)
from mpc_spacecraft.config import (
    ClosedLoopEnvironmentConfig,
    ClosedLoopTestConfig,
    SpacecraftPhysicalConfig,
)

DEFAULT_OUTPUT_DIR = Path("./experiments/results/single_target_robustness")


@dataclass(frozen=True)
class RobustnessScenario:
    name: str
    label: str
    environment: ClosedLoopEnvironmentConfig
    truth_inertia: np.ndarray
    inertia_factors: tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-duration", type=float, default=300.0)
    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--mpc-dt", type=float, default=5.0)
    parser.add_argument("--initial-attitude-deg", type=float, default=30.0)
    parser.add_argument("--initial-rate-deg-s", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_config = ClosedLoopTestConfig.defaults(show_progress=False)
    nominal_inertia = base_config.spacecraft.inertia.copy()
    scenarios = _build_scenarios(base_config, nominal_inertia)
    disturbance_comparison = _disturbance_comparison(
        scenarios,
        base_config,
        duration_s=args.max_duration,
        sim_dt=args.sim_dt,
    )
    disturbance_comparison.to_csv(
        args.output_dir / "disturbance_torque_comparison.csv", index=False
    )
    print("Configured disturbance torque versus actuator limits:")
    print(
        disturbance_comparison[
            [
                "label",
                "disturbance_mean_norm_nm",
                "disturbance_peak_norm_nm",
                "disturbance_peak_axis_utilization",
                "disturbance_exceeds_axis_limit",
            ]
        ].to_string(index=False)
    )

    cases: list[SingleTargetCaseResult] = []
    for scenario in scenarios:
        print(f"Running robustness case: {scenario.label}")
        spacecraft = replace(
            base_config.spacecraft,
            inertia=scenario.truth_inertia.copy(),
        )
        # Keep the controller's prediction inertia nominal in every case.
        mpc = replace(base_config.mpc, model_inertia=nominal_inertia.copy())
        scenario_base = replace(
            base_config,
            environment=scenario.environment,
            spacecraft=spacecraft,
            mpc=mpc,
        )
        config = make_single_target_config(
            attitude_offset_deg=args.initial_attitude_deg,
            angular_rate_deg_s=args.initial_rate_deg_s,
            max_duration_s=args.max_duration,
            sim_dt=args.sim_dt,
            mpc_dt=args.mpc_dt,
            base_config=scenario_base,
        )
        cases.append(run_single_target_case(scenario.name, scenario.label, config))

    metrics = pd.DataFrame(
        [
            {
                "case": case.name,
                "label": case.label,
                **case.metrics,
            }
            for case in cases
        ]
    )
    metrics.to_csv(args.output_dir / "robustness_metrics.csv", index=False)
    _plot_pointing(cases, args.output_dir / "robustness_pointing_error.png")
    _plot_torque(cases, args.output_dir / "robustness_torque.png")

    summary = {
        "schema_version": "mpc_spacecraft_single_target_robustness_v1",
        "configuration": {
            "sim_dt_s": args.sim_dt,
            "mpc_dt_s": args.mpc_dt,
            "maximum_duration_s": args.max_duration,
            "required_observation_time_s": 30.0,
            "initial_attitude_offset_deg": args.initial_attitude_deg,
            "initial_angular_rate_deg_s": args.initial_rate_deg_s,
            "mekf_enabled": True,
            "mpc_model_inertia": nominal_inertia.tolist(),
            "disturbances_modeled_by_mpc": False,
        },
        "scenarios": [
            {
                "name": scenario.name,
                "label": scenario.label,
                "truth_inertia": scenario.truth_inertia.tolist(),
                "inertia_axis_factors": list(scenario.inertia_factors),
                "disturbance_bias_nm": (
                    scenario.environment.disturbance_bias.tolist()
                    if scenario.environment.disturbance_bias is not None
                    else None
                ),
                "disturbance_noise_std_nm": scenario.environment.disturbance_noise_std,
                "disturbance_sinusoidal_amplitude_nm": (
                    scenario.environment.disturbance_sinusoidal_amplitude
                ),
            }
            for scenario in scenarios
        ],
        "disturbance_torque_comparison": disturbance_comparison.to_dict(
            orient="records"
        ),
        "metrics": metrics.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved single-target robustness artifacts to {args.output_dir}")


def _build_scenarios(
    base_config: ClosedLoopTestConfig, nominal_inertia: np.ndarray
) -> list[RobustnessScenario]:
    nominal_environment = base_config.environment
    moderate_environment = replace(
        nominal_environment,
        disturbance_bias=np.array([0.01, -0.006, 0.004], dtype=np.float64),
        disturbance_noise_std=5.0e-4,
        disturbance_sinusoidal_amplitude=0.005,
    )
    severe_environment = replace(
        nominal_environment,
        disturbance_bias=np.array([0.04, -0.024, 0.016], dtype=np.float64),
        disturbance_noise_std=2.0e-3,
        disturbance_sinusoidal_amplitude=0.02,
    )
    moderate_factors = np.array([1.20, 0.85, 1.15], dtype=np.float64)
    severe_factors = np.array([1.35, 0.70, 1.25], dtype=np.float64)
    return [
        RobustnessScenario(
            "nominal",
            "Nominal",
            nominal_environment,
            nominal_inertia.copy(),
            (1.0, 1.0, 1.0),
        ),
        RobustnessScenario(
            "moderate_disturbance",
            "Moderate disturbance",
            moderate_environment,
            nominal_inertia.copy(),
            (1.0, 1.0, 1.0),
        ),
        RobustnessScenario(
            "severe_disturbance",
            "Severe disturbance",
            severe_environment,
            nominal_inertia.copy(),
            (1.0, 1.0, 1.0),
        ),
        RobustnessScenario(
            "moi_mismatch",
            "MOI mismatch",
            nominal_environment,
            perturb_inertia(nominal_inertia, moderate_factors),
            (
                float(moderate_factors[0]),
                float(moderate_factors[1]),
                float(moderate_factors[2]),
            ),
        ),
        RobustnessScenario(
            "severe_combined",
            "Severe disturbance + MOI mismatch",
            severe_environment,
            perturb_inertia(nominal_inertia, severe_factors),
            (
                float(severe_factors[0]),
                float(severe_factors[1]),
                float(severe_factors[2]),
            ),
        ),
    ]


def _disturbance_comparison(
    scenarios: list[RobustnessScenario],
    base_config: ClosedLoopTestConfig,
    *,
    duration_s: float,
    sim_dt: float,
) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        rows.append(
            {
                "case": scenario.name,
                "label": scenario.label,
                **summarize_disturbance_torque(
                    scenario.environment,
                    duration_s=duration_s,
                    sim_dt=sim_dt,
                    rng_seed=base_config.rng_seed,
                    actuator_min=base_config.spacecraft.u_min,
                    actuator_max=base_config.spacecraft.u_max,
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_pointing(cases: list[SingleTargetCaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for case in cases:
        error = case.logs["active_target_boresight_angle_deg"].to_numpy(
            dtype=np.float64
        )
        ax.semilogy(
            case.logs["time"],
            np.maximum(error, 1e-6),
            label=_case_label(case),
        )
    ax.axhline(0.5, color="tab:orange", linestyle="--", label="0.5° settling threshold")
    ax.axhline(
        0.01, color="tab:red", linestyle=":", label="0.01° observation threshold"
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("True boresight pointing error [deg]")
    ax.set_title("Single-Target Robustness: Pointing Error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_torque(cases: list[SingleTargetCaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (torque_ax, utilization_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for case in cases:
        control = case.logs[["control_tx", "control_ty", "control_tz"]].to_numpy(
            dtype=np.float64
        )
        torque_norm = np.linalg.norm(control, axis=1)
        limits = np.maximum(
            np.abs(case.result.config.spacecraft.u_min),
            np.abs(case.result.config.spacecraft.u_max),
        )
        utilization = np.max(np.abs(control) / limits, axis=1)
        torque_ax.plot(case.logs["time"], torque_norm, label=case.label)
        utilization_ax.plot(case.logs["time"], utilization, label=case.label)

    norm_limit = float(np.linalg.norm(cases[0].result.config.spacecraft.u_max))
    torque_ax.axhline(
        norm_limit,
        color="black",
        linestyle="--",
        label=r"$\|\tau_{max}\|_2$",
    )
    utilization_ax.axhline(1.0, color="black", linestyle="--", label="axis saturation")
    torque_ax.set_ylabel("Torque norm [N m]")
    utilization_ax.set_ylabel("Maximum axis utilization")
    utilization_ax.set_xlabel("Time [s]")
    torque_ax.set_title("Single-Target Robustness: Control Torque")
    for ax in (torque_ax, utilization_ax):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _case_label(case: SingleTargetCaseResult) -> str:
    settling_time = float(case.metrics["settling_time_s"])
    settling = (
        f"settled at {settling_time:.1f}s"
        if np.isfinite(settling_time)
        else "not settled"
    )
    observed = float(case.metrics["target_observation_time_s"])
    completion = (
        "observation complete"
        if case.metrics["completed_observation"]
        else f"observation incomplete ({observed:.0f}/30s)"
    )
    return f"{case.label} ({settling}; {completion})"


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
