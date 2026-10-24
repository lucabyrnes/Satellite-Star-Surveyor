"""Minimal closed-loop test example.

Run this file after installing the package in editable mode, for example:

    pip install -e .

The script stores experiment logs in ``experiments/closed_loop_experiment.pkl``
and writes figures under ``experiments/figures``.
"""

from mpc_spacecraft.simulation import (
    ClosedLoopTestConfig,
    SpacecraftPhysicalConfig,
    run_closed_loop_test,
)
import matplotlib.pyplot as plt
import numpy as np
import quaternion as qu
import pickle
from pathlib import Path


SHOW_PLOTS = False
SHOW_URSINA = True
EXPERIMENT_DIR = Path("./experiments")
FIGURE_DIR = Path("./experiments/figures")
DATA_FILE = Path("./experiments/closed_loop_experiment.pkl")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


config = ClosedLoopTestConfig.defaults(
    # spacecraft=SpacecraftPhysicalConfig.medium_space_telescope()
)

result = run_closed_loop_test(config)

# Plotting-friendly outputs.
logs = result.to_dataframe()
arrays = result.to_arrays()

experiment_payload = {
    "logs": logs,
    "targets": config.targets.targets.copy(),
    "metadata": {
        "data_format": "mpc_spacecraft_closed_loop_experiment_v1",
        "description": "Closed-loop spacecraft experiment logs and visualization target metadata.",
        "sim_dt": config.timing.sim_dt,
        "mpc_dt": config.timing.mpc_dt,
        "sim_cycles": config.timing.sim_cycles,
        "rng_seed": config.rng_seed,
        "sun_keepout_deg": config.environment.sun_keep_out_deg,
        "earth_margin_deg": config.environment.earth_margin_deg,
        "torque_limit_nm": float(
            np.max(
                np.maximum(
                    np.abs(config.spacecraft.u_min),
                    np.abs(config.spacecraft.u_max),
                )
            )
        ),
    },
}

with DATA_FILE.open("wb") as data_handle:
    pickle.dump(experiment_payload, data_handle)

print(f"Saved experiment data to {DATA_FILE}")

print(logs[["time", "guidance_mode", "is_complete", "control_tx", "control_ty", "control_tz"]].head())


def quaternion_angle_error_deg(q_current: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
    """Return shortest-axis attitude angle error between quaternion histories."""
    current = qu.as_quat_array(q_current)
    goal = qu.as_quat_array(q_goal)
    error = goal * current.conjugate()
    error_float = qu.as_float_array(error)
    error_float = error_float / np.linalg.norm(error_float, axis=1, keepdims=True)
    scalar = np.clip(np.abs(error_float[:, 0]), -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(scalar))


time = logs["time"].to_numpy()

true_quat = logs[["true_quat_w", "true_quat_x", "true_quat_y", "true_quat_z"]].to_numpy()
goal_quat = logs[["goal_quat_w", "goal_quat_x", "goal_quat_y", "goal_quat_z"]].to_numpy()
angle_error_deg = quaternion_angle_error_deg(true_quat, goal_quat)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time, angle_error_deg)
ax.set_title("Goal vs Current Attitude Angle Error")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Angle error [deg]")
ax.grid(True)
fig.tight_layout()
angle_error_figure = FIGURE_DIR / "closed_loop_goal_angle_error.png"
fig.savefig(angle_error_figure, dpi=200)
print(f"Saved figure to {angle_error_figure}")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)


state_groups = [
    (
        "Position",
        ["x", "y", "z"],
        "m",
        ["true_position_x", "true_position_y", "true_position_z"],
        ["estimated_position_x", "estimated_position_y", "estimated_position_z"],
    ),
    (
        "Velocity",
        ["x", "y", "z"],
        "m/s",
        ["true_velocity_x", "true_velocity_y", "true_velocity_z"],
        ["estimated_velocity_x", "estimated_velocity_y", "estimated_velocity_z"],
    ),
    (
        "Quaternion",
        ["w", "x", "y", "z"],
        "unitless",
        ["true_quat_w", "true_quat_x", "true_quat_y", "true_quat_z"],
        ["estimated_quat_w", "estimated_quat_x", "estimated_quat_y", "estimated_quat_z"],
    ),
    (
        "Angular Velocity",
        ["x", "y", "z"],
        "rad/s",
        ["true_omega_x", "true_omega_y", "true_omega_z"],
        ["estimated_omega_x", "estimated_omega_y", "estimated_omega_z"],
    ),
]

for title, component_labels, unit, true_cols, estimated_cols in state_groups:
    fig, axes = plt.subplots(len(component_labels), 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Estimated vs Sim State: {title}")

    for ax, component, true_col, estimated_col in zip(
        axes, component_labels, true_cols, estimated_cols
    ):
        ax.plot(time, logs[true_col], label=f"sim {component}")
        ax.plot(time, logs[estimated_col], "--", label=f"estimated {component}")
        ax.set_ylabel(f"{component} [{unit}]")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    figure_name = title.lower().replace(" ", "_")
    state_figure = FIGURE_DIR / f"closed_loop_estimated_vs_sim_{figure_name}.png"
    fig.savefig(state_figure, dpi=200)
    print(f"Saved figure to {state_figure}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

if SHOW_URSINA:
    from mpc_spacecraft.visualization import UrsinaSpacecraftVisualizer

    viz = UrsinaSpacecraftVisualizer()
    viz.visualize_closed_loop_dataframe(logs, fps=30.0, every_n=1, play=True, targets=config.targets.targets)
