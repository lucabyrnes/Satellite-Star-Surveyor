# Satellite Star Surveyor

Satellite Star Surveyor is a Python simulation project for autonomous space telescope pointing, target selection, state estimation, and attitude control. The project combines a high-level observation planner, spacecraft attitude guidance, nonlinear state estimation, and Model Predictive Control (MPC) to simulate a telescope in orbit selecting stars, slewing toward targets, stabilizing, and collecting observation time.

The system is designed around a closed-loop workflow:

1. Select a feasible star target based on Sun and Earth keepout constraints.
2. Generate a desired spacecraft attitude that points the telescope boresight at the selected target.
3. Estimate the spacecraft state using simulated sensor measurements.
4. Use MPC to compute torque commands for attitude control.
5. Propagate spacecraft dynamics and visualize the resulting mission behavior.

## Features

* **High-level star observation planner**

  * Tracks required observation time for each target.
  * Selects feasible unfinished targets.
  * Avoids targets blocked by Earth or too close to the Sun.
  * Switches between `NO_TARGET`, `SLEWING`, `SETTLING`, and `OBSERVING` modes.

* **Model Predictive Control**

  * Uses a receding-horizon MPC controller for spacecraft attitude control.
  * Supports goal regulation and trajectory tracking.
  * Uses Drake’s `MathematicalProgram` with OSQP as the optimization backend.
  * Includes warm-starting and SQP-style repeated linearized solves for goal-only control.
  * Uses time-varying linearized dynamics constraints across the prediction horizon.
  * Enforces actuator limits, attitude constraints, and Sun keepout constraints.
  * Supports offset-free regulation using an estimated constant disturbance torque.
  * Solves a constrained optimal control problem directly on a nonlinear spacecraft attitude system.

* **Multiplicative Extended Kalman Filter**

  * Estimates position, velocity, attitude, angular velocity, gyroscope bias, and accelerometer bias.
  * Uses quaternion-based attitude propagation.
  * Supports simulated IMU, gyroscope, GNSS, and star tracker updates.
  * Uses Joseph-form covariance updates for numerical stability.

* **Closed-loop simulation**

  * Combines planner, guidance, MPC, estimator, dynamics, sensor updates, and environmental disturbances.
  * Produces structured logs for plotting, debugging, and replay.
  * Includes configurable simulation timing, spacecraft inertia, torque limits, sensor sampling, target lists, and disturbance models.

* **3D visualization**

  * Uses Ursina/Panda3D to visualize closed-loop simulation results.
  * Displays Earth, Sun direction, satellite body, orbit trail, target stars, attitude frame, and boresight direction.
  * Includes playback controls, orbit camera, zoom, stepping, speed controls, and a spacecraft-fixed camera mode.

## Demo

3D visualization of the simulation. Objects are not to scale.

Axis-aligned camera view showing where the satellite is pointing.

## Project Structure

```text
Satellite-Star-Surveyor/
├── src/mpc_spacecraft/
│   ├── analysis/          # Analysis utilities and experiment helpers
│   ├── config/            # Configuration files and packaged config data
│   ├── controllers/       # MPC and error-state dynamics adapters
│   ├── dynamics/          # Spacecraft dynamics and disturbance models
│   ├── estimation/        # MEKF, sensor models, and estimated state types
│   ├── guidance/          # Guidance logic and Sun direction models
│   ├── planner/           # High-level star target planner
│   ├── scripts/           # Script entry points
│   ├── simulation/        # Closed-loop simulation and logging tools
│   ├── utilities/         # Math, state, array-view, and SO(3) utilities
│   └── visualization/     # Ursina-based 3D visualization
├── tests/                 # Unit and integration tests
├── experiments/           # Experiment outputs and figures
├── notebooks/             # Jupyter notebooks
├── Notes/                 # Project notes
├── pyproject.toml
└── README.md
```

## Core Components

### Star Planner

The star planner is responsible for deciding what the spacecraft should observe next. Targets are provided as a table containing right ascension, declination, and required observation time. The planner converts the target directions into inertial unit vectors, checks whether each target is feasible, and selects the best unfinished target.

A target is considered feasible when:

* It is outside the Earth exclusion region.
* It is farther than the Sun keepout angle.
* It still has remaining required observation time.

Planner modes:

| Mode        | Description                                                                         |
| ----------- | ----------------------------------------------------------------------------------- |
| `NO_TARGET` | No currently feasible unfinished target is available.                               |
| `SLEWING`   | The spacecraft is rotating toward the selected target.                              |
| `SETTLING`  | The spacecraft is close to the target but still stabilizing.                        |
| `OBSERVING` | The spacecraft is accurately pointed and stable enough to collect observation time. |

### Guidance

The guidance layer connects target planning to attitude control. It takes the selected target direction and produces a desired rotation state for the spacecraft. The telescope boresight is aligned with the target while respecting environmental constraints such as Sun avoidance.

The guidance system also generates attitude goals that are compatible with the MPC's safety constraints. Rather than simply commanding the shortest rotational path to a target, the guidance layer must ensure that the resulting attitude trajectory remains feasible with respect to Sun exclusion requirements and spacecraft pointing limitations.

### Model Predictive Control

The MPC controller computes torque commands for the spacecraft attitude system. It formulates a finite-horizon optimization problem in error coordinates and solves it using Drake and OSQP.

The controller supports:

* Quaternion attitude states.
* Angular velocity tracking.
* State and control quadratic costs.
* Terminal costs.
* Control bounds.
* Warm-started optimization.
* Linearized error dynamics over the prediction horizon.
* Time-varying dynamics constraints generated from successive linearizations.
* Sun keepout constraints enforced throughout the prediction horizon.

At each control update, the controller solves for a sequence of future torque commands and applies only the first command.

An optional constant-torque observer estimates slowly varying, body-frame
disturbances from angular-rate model residuals. When enabled, the estimate is
included in the prediction dynamics and subtracted from the control reference,
allowing the MPC to hold a target without requiring a steady pointing offset.
The estimate is filtered and bounded, and disturbances beyond actuator authority
cannot be fully rejected.

### MEKF State Estimation

The Multiplicative Extended Kalman Filter estimates spacecraft state using simulated sensor data. It propagates the nominal state with IMU and gyro data, then corrects it with optional GNSS and star tracker measurements.

The filter estimates:

* Position
* Velocity
* Attitude quaternion
* Angular velocity
* Gyroscope bias
* Accelerometer bias

The MEKF uses a 15-dimensional error state:

```text
[position error, velocity error, attitude error, gyro bias error, accel bias error]
```

### Closed-Loop Simulation

The closed-loop simulation ties the whole project together. It includes:

* Spacecraft rigid-body dynamics
* Gravity and disturbance torques
* Sensor bias evolution
* State estimation
* Star target planning
* Attitude guidance
* MPC torque control
* Simulation clocking
* Structured logging

The main reusable API is built around `run_closed_loop_test`, which returns a `ClosedLoopTestResult`. The result can be converted into arrays or a pandas DataFrame for plotting, analysis, or visualization.

### 3D visualization of simulation (objects not to scale)
<img width="1000" alt="Screenshot 2026-06-07 181117" src="https://github.com/user-attachments/assets/497662bd-b25d-45f2-8902-0a3f22a5e339" />


### Axis aligned camera showing where satellite is pointing
<img width="1000" alt="Screenshot 2026-06-07 181038" src="https://github.com/user-attachments/assets/3f681317-15be-4067-a441-1dc392458054" />



## Installation

This project supports Python 3.12 through 3.14.

Clone the repository:

```bash
git clone https://github.com/Makotastic/Satellite-Star-Surveyor.git
cd Satellite-Star-Surveyor
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the package and its runtime dependencies (including NumPy) in editable
mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For development, including tests, visualization, and notebook tooling, install
the `dev` extra instead:

```bash
python -m pip install -e ".[dev]"
```

All Python dependencies are declared in `pyproject.toml`; there is no separate
requirements file to keep in sync.

## Major Dependencies

The project uses:

* `numpy`
* `scipy`
* `pandas`
* `numpy-quaternion`
* `drake`
* `matplotlib`
* `ursina`
* `skyfield`
* `astropy`
* `pytest`
* `jupyter`

## Running Tests

Run the test suite with:

```bash
pytest
```

The tests cover spacecraft dynamics, error-state mapping, MPC behavior, prediction adapters, simulation clocking, state estimation simulation handling, Sun tracking, and SO(3) utilities.

## Example: Running a Closed-Loop Simulation

```python
from mpc_spacecraft.simulation.closed_loop_testing import (
    ClosedLoopTestConfig,
    run_closed_loop_test,
)

config = ClosedLoopTestConfig.defaults()
result = run_closed_loop_test(config)

logs = result.to_dataframe()
arrays = result.to_arrays()
```

The returned `result` contains:

* Per-cycle simulation records
* True spacecraft state
* Estimated spacecraft state
* Goal attitude
* Applied control torque
* Guidance/planner mode
* Target observation progress
* Final simulation state

## Example: Visualizing a Simulation

```python
from mpc_spacecraft.simulation.closed_loop_testing import run_closed_loop_test
from mpc_spacecraft.visualization.ursina_viz import UrsinaSpacecraftVisualizer

result = run_closed_loop_test()
logs = result.to_dataframe()

viz = UrsinaSpacecraftVisualizer()
viz.visualize_closed_loop_dataframe(logs)
```

Visualizer controls:

| Key / Input       | Action                                      |
| ----------------- | ------------------------------------------- |
| Space             | Play or pause                               |
| Left / Right      | Step backward or forward                    |
| PageUp / PageDown | Skip forward or backward                    |
| Home / End        | Jump to first or last frame                 |
| Mouse drag        | Orbit camera                                |
| Mouse wheel       | Zoom                                        |
| `+` / `-`         | Increase or decrease playback speed         |
| `R`               | Reset camera                                |
| `F`               | Toggle spacecraft-fixed body-forward camera |

## Configuration

Closed-loop simulations are configured through dataclasses:

* `ClosedLoopTimingConfig`
* `ClosedLoopEnvironmentConfig`
* `SpacecraftPhysicalConfig`
* `ClosedLoopMPCConfig`
* `SensorScheduleConfig`
* `InitialStateConfig`
* `TargetConfig`
* `ClosedLoopTestConfig`

Example:

```python
from mpc_spacecraft.simulation.closed_loop_testing import (
    ClosedLoopTestConfig,
    ClosedLoopTimingConfig,
    SpacecraftPhysicalConfig,
    run_closed_loop_test,
)

config = ClosedLoopTestConfig(
    timing=ClosedLoopTimingConfig(
        sim_dt=0.1,
        mpc_dt=1.0,
        sim_cycles=6000,
    ),
    spacecraft=SpacecraftPhysicalConfig.medium_space_telescope(),
)

result = run_closed_loop_test(config)
```

Disturbance rejection is opt-in and can be configured through the MPC settings:

```python
from dataclasses import replace

config = ClosedLoopTestConfig.defaults()
config = replace(
    config,
    mpc=replace(
        config.mpc,
        enable_disturbance_rejection=True,
        disturbance_observer_time_constant_s=300.0,
    ),
)
```

## Target Format

The planner expects a pandas DataFrame with the following columns:

| Column         | Description                                          |
| -------------- | ---------------------------------------------------- |
| `req_obs_time` | Required observation time for the target, in seconds |
| `Dec`          | Declination, in radians                              |
| `RA`           | Right ascension, in radians                          |

Example:

```python
import pandas as pd

targets = pd.DataFrame(
    {
        "req_obs_time": [20.0, 20.0, 20.0],
        "Dec": [0.174533, -0.349066, 0.610865],
        "RA": [0.261799, 1.396263, 2.530727],
    }
)
```

## Timestep Convergence Experiment

Run the isolated dynamics verification first, followed by deterministic
closed-loop timestep sensitivity:

```bash
python experiments/timestep_convergence.py --phase all
```

The default closed-loop experiment lasts 120 seconds and keeps the MPC and
sensor update periods fixed in physical time. Results, CSV metrics, and plots
are written to `experiments/results/timestep_convergence/`. Individual phases
can be selected with `--phase dynamics` or `--phase closed-loop`; use `--help`
to list timestep and duration overrides.

## Single-Target Performance Experiments

Run the MEKF-in-the-loop initial-condition convergence sweeps with:

```bash
python experiments/single_target_convergence.py
```

Add `--disturbance-rejection` to run the convergence sweeps with the
constant-torque observer and offset-free MPC. Use
`--no-disturbance-rejection` to explicitly select the nominal controller.

Run the nominal, disturbance, and moment-of-inertia mismatch comparison with:

```bash
python experiments/single_target_robustness.py
```

The robustness runner also accepts `--disturbance-rejection` and
`--no-disturbance-rejection`.

Both experiments use a `0.01 s` simulation timestep and a `2.0 s` MPC update
period, stop after the target has accumulated 30 seconds of observation or
after 300 seconds, and save plots plus CSV/JSON metrics under
`experiments/results/`. Use each script's `--help` option to override the
timing, initial-condition sweep, or output directory.

## Notes

This project is a simulation and controls research prototype. It is intended for experimenting with spacecraft attitude dynamics, autonomous star target selection, nonlinear state estimation, and MPC-based control. The visualized bodies are not rendered to scale.
