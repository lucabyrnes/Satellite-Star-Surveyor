"""Profile the closed-loop simulation inner loop without changing the API.

This script mirrors the setup in ``run_closed_loop_test`` and instruments the
loop body corresponding to ``src/mpc_spacecraft/simulation/closed_loop_testing.py``
lines 329-377.  It is intentionally standalone so production simulation code is
not modified while you investigate where time is spent.

Recommended quick run:

    python experiments/profile_closed_loop_inner_loop.py --cycles 200 --warmup-cycles 20

Useful variants:

    # Isolate the cost of building/storing ClosedLoopLogRecord objects.
    python experiments/profile_closed_loop_inner_loop.py --cycles 200 --disable-log-records

    # Repeat multiple deterministic runs and write machine-readable output.
    python experiments/profile_closed_loop_inner_loop.py --cycles 200 --repeat 3 --json experiments/closed_loop_profile.json

Interpretation:
    * ``pct_loop`` is the share of measured loop-body time for the run.
    * ``avg_ms`` is averaged over calls to that measured block.
    * ``mpc_update_avg_ms`` is only populated for blocks that execute during MPC
      update cycles, which prevents expensive infrequent work from being hidden
      by cheap non-update cycles.
    * Run with progress bars disabled and use ``--repeat`` when comparing code
      changes, because solver and import warmup effects can be significant.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterator, cast

import numpy as np

from mpc_spacecraft.controllers import NominalMPC
from mpc_spacecraft.controllers.error_dynamics_adapters import (
    SpacecraftErrorDynamicsProvider,
)
from mpc_spacecraft.dynamics import DisturbanceModel, SpacecraftDynamics
from mpc_spacecraft.estimation.sensor_models import GyroConfig, IMUConfig
from mpc_spacecraft.guidance import AstropySunDirectionModel, Guidance
from mpc_spacecraft.planner import StarPlanner
from mpc_spacecraft.simulation import (
    ClosedLoopLogRecord,
    ClosedLoopTestConfig,
    ClosedLoopTestResult,
    ClosedLoopTimingConfig,
)
from mpc_spacecraft.simulation.clock import SimulationClock
from mpc_spacecraft.simulation.sensor_state_updater import SensorBiasUpdater
from mpc_spacecraft.simulation.state_estimation_sim_handler import (
    StateEstimationSimHandler,
)
from mpc_spacecraft.utilities.utils import RigidBodyControl, RigidBodyState


@dataclass
class TimerStats:
    """Accumulate nanosecond timings for one named section."""

    total_ns: int = 0
    calls: int = 0
    mpc_update_total_ns: int = 0
    mpc_update_calls: int = 0

    def add(self, elapsed_ns: int, *, mpc_update: bool = False) -> None:
        self.total_ns += elapsed_ns
        self.calls += 1
        if mpc_update:
            self.mpc_update_total_ns += elapsed_ns
            self.mpc_update_calls += 1

    def reset(self) -> None:
        self.total_ns = 0
        self.calls = 0
        self.mpc_update_total_ns = 0
        self.mpc_update_calls = 0


class InnerLoopProfiler:
    """Small zero-dependency timer for named closed-loop sections."""

    def __init__(self) -> None:
        self.stats: defaultdict[str, TimerStats] = defaultdict(TimerStats)

    @contextmanager
    def section(self, name: str, *, mpc_update: bool = False) -> Iterator[None]:
        start_ns = perf_counter_ns()
        try:
            yield
        finally:
            self.stats[name].add(perf_counter_ns() - start_ns, mpc_update=mpc_update)

    def reset(self) -> None:
        for stat in self.stats.values():
            stat.reset()

    def rows(self, *, run_index: int, measured_loop_ns: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, stat in self.stats.items():
            total_ms = stat.total_ns / 1_000_000.0
            avg_ms = total_ms / stat.calls if stat.calls else 0.0
            update_avg_ms = (
                stat.mpc_update_total_ns / 1_000_000.0 / stat.mpc_update_calls
                if stat.mpc_update_calls
                else None
            )
            rows.append(
                {
                    "run": run_index,
                    "section": name,
                    "total_ms": total_ms,
                    "calls": stat.calls,
                    "avg_ms": avg_ms,
                    "pct_loop": (100.0 * stat.total_ns / measured_loop_ns)
                    if measured_loop_ns
                    else 0.0,
                    "mpc_update_calls": stat.mpc_update_calls,
                    "mpc_update_avg_ms": update_avg_ms,
                }
            )

        return sorted(rows, key=lambda row: cast(float, row["total_ms"]), reverse=True)


@dataclass(frozen=True)
class ProfileRunMetadata:
    """Metadata attached to one benchmark run."""

    run: int
    cycles: int
    warmup_cycles: int
    measured_cycles: int
    sim_dt: float
    mpc_dt: float
    disable_log_records: bool
    measured_loop_ms: float
    total_wall_ms: float
    mpc_update_cycles: int


def _build_config(args: argparse.Namespace) -> ClosedLoopTestConfig:
    return ClosedLoopTestConfig.defaults(
        timing=ClosedLoopTimingConfig(
            sim_dt=args.sim_dt,
            mpc_dt=args.mpc_dt,
            sim_cycles=args.cycles + args.warmup_cycles,
        ),
        rng_seed=args.seed,
        show_progress=False,
    )


def _run_profile_once(args: argparse.Namespace, *, run_index: int) -> tuple[ProfileRunMetadata, list[dict[str, Any]]]:
    config = _build_config(args)
    timing = config.timing
    environment = config.environment
    spacecraft = config.spacecraft
    mpc_config = config.mpc
    sensors = config.sensors

    sim_state = config.initial_state.to_full_sim_state()
    clock = SimulationClock(timing.start_epoch_utc, timing.sim_dt)
    sun_model = AstropySunDirectionModel()
    planner = StarPlanner(
        config.targets.targets.copy(),
        environment.sun_keep_out_deg,
        environment.earth_margin_deg,
    )
    guidance = Guidance(planner, sun_model)

    disturbance_model = DisturbanceModel(
        environment.disturbance_bias,
        environment.disturbance_noise_std,
        environment.disturbance_sinusoidal_amplitude,
        environment.disturbance_sinusoidal_frequency,
        config.rng_seed,
        enable_gravity=environment.enable_gravity,
    )
    dynamics = SpacecraftDynamics(spacecraft.inertia, disturbance_model, spacecraft.mass)
    error_dynamics_provider = SpacecraftErrorDynamicsProvider(
        dynamics,
        sun_model,
        np.deg2rad(environment.sun_keep_out_deg),
        timing.mpc_dt,
    )
    mpc = NominalMPC(
        mpc_config.step_horizon,
        mpc_config.q_weight,
        mpc_config.r_weight,
        error_dynamics_provider,
        mpc_config.q_terminal if mpc_config.q_terminal is not None else mpc_config.q_weight,
        spacecraft.u_min,
        spacecraft.u_max,
    )
    estimator = StateEstimationSimHandler(
        sim_state.sensor_rigid_body,
        sensors.gnss_measurement_period,
        sensors.star_tracker_measurement_period,
        config.rng_seed,
    )
    sensor_bias_updater = SensorBiasUpdater(IMUConfig(), GyroConfig(), config.rng_seed)

    result = ClosedLoopTestResult(config=config)
    profiler = InnerLoopProfiler()
    total_control = RigidBodyControl.zeros()
    previous_mpc_guidance_update: float | None = None
    goal_rotation_state = sim_state.rotation.copy()
    mode = None
    is_complete = False
    mpc_update_cycles = 0
    measured_loop_ns = 0

    total_wall_start_ns = perf_counter_ns()
    for sim_cycle in range(timing.sim_cycles):
        measuring = sim_cycle >= args.warmup_cycles
        if measuring and sim_cycle == args.warmup_cycles:
            profiler.reset()
            measured_loop_ns = 0
            mpc_update_cycles = 0

        loop_start_ns = perf_counter_ns() if measuring else 0

        with profiler.section("estimator.tick") if measuring else _null_section():
            if clock.last_dt != 0:
                estimated_state, estimator_used_star_tracker, estimator_used_gnss = estimator.tick(
                    clock.current_time, clock.last_dt, sim_state
                )
            else:
                estimated_state = sim_state.sensor_rigid_body
                estimator_used_star_tracker = False
                estimator_used_gnss = False

        with profiler.section("mpc_update_check") if measuring else _null_section():
            mpc_update = previous_mpc_guidance_update is None or (
                clock.current_time - previous_mpc_guidance_update >= timing.mpc_dt
            )

        if measuring and mpc_update:
            mpc_update_cycles += 1

        if mpc_update:
            with profiler.section("guidance.tick", mpc_update=True) if measuring else _null_section():
                goal_rotation_state, mode, is_complete = guidance.tick(
                    clock.now(), timing.sim_dt, cast(RigidBodyState, estimated_state.rigid_body)
                )
            with profiler.section("mpc.get_first_control", mpc_update=True) if measuring else _null_section():
                rotation_control = mpc.get_first_control(
                    x0=estimated_state.rotation,
                    x_goal=goal_rotation_state,
                    current_epoch_utc=clock.now(),
                )
            with profiler.section("control_assignment", mpc_update=True) if measuring else _null_section():
                total_control = RigidBodyControl.zeros()
                total_control.rotation[:] = rotation_control
                previous_mpc_guidance_update = clock.current_time

        if args.disable_log_records:
            with profiler.section("log_record_skipped") if measuring else _null_section():
                pass
        else:
            with profiler.section("log_record_append") if measuring else _null_section():
                result.append(
                    ClosedLoopLogRecord(
                        cycle=sim_cycle,
                        time=clock.current_time,
                        epoch_utc=clock.now(),
                        true_state=sim_state.sensor_rigid_body.data.copy(),
                        estimated_state=estimated_state.data.copy(),
                        goal_rotation_state=goal_rotation_state.data.copy(),
                        control=total_control.torque.copy(),
                        estimated_disturbance_torque=np.zeros(3, dtype=np.float64),
                        guidance_mode_value=int(mode) if mode is not None else -1,
                        guidance_mode_name=mode.name if mode is not None else "UNKNOWN",
                        is_complete=bool(is_complete),
                        mpc_updated=bool(mpc_update),
                        mpc_solve_success=bool(getattr(mpc, "last_solve_success", False)),
                        mpc_fallback_used=bool(getattr(mpc, "last_fallback_used", False)),
                        estimator_used_gnss=bool(estimator_used_gnss),
                        estimator_used_star_tracker=bool(estimator_used_star_tracker),
                        mekf_covariance_diag=np.diag(estimator.mekf.P).copy(),
                        current_target_idx=getattr(planner, "_current_target_idx", None),
                        target_obs_times=planner._targets["obs_time"].to_numpy().copy(),
                    )
                )

        with profiler.section("clock.advance") if measuring else _null_section():
            time_step = clock.advance(1)
        with profiler.section("dynamics.rk4_full_state") if measuring else _null_section():
            sim_state.rigid_body[:] = dynamics.discretize_dynamics_rk4_full_state(
                sim_state.rigid_body,
                total_control,
                time_step.t0,
                time_step.dt,
            ).data
        with profiler.section("sensor_bias_updater.tick") if measuring else _null_section():
            sim_state = sensor_bias_updater.tick(sim_state, time_step.dt)

        if measuring:
            measured_loop_ns += perf_counter_ns() - loop_start_ns

    total_wall_ns = perf_counter_ns() - total_wall_start_ns
    metadata = ProfileRunMetadata(
        run=run_index,
        cycles=timing.sim_cycles,
        warmup_cycles=args.warmup_cycles,
        measured_cycles=args.cycles,
        sim_dt=timing.sim_dt,
        mpc_dt=timing.mpc_dt,
        disable_log_records=args.disable_log_records,
        measured_loop_ms=measured_loop_ns / 1_000_000.0,
        total_wall_ms=total_wall_ns / 1_000_000.0,
        mpc_update_cycles=mpc_update_cycles,
    )
    return metadata, profiler.rows(run_index=run_index, measured_loop_ns=measured_loop_ns)


@contextmanager
def _null_section() -> Iterator[None]:
    yield


def _print_run(metadata: ProfileRunMetadata, rows: list[dict[str, Any]]) -> None:
    print(
        f"\nRun {metadata.run}: measured {metadata.measured_cycles} cycles "
        f"after {metadata.warmup_cycles} warmup cycles; "
        f"loop={metadata.measured_loop_ms:.3f} ms, wall={metadata.total_wall_ms:.3f} ms, "
        f"mpc_updates={metadata.mpc_update_cycles}"
    )
    print(
        f"{'section':32} {'total_ms':>12} {'calls':>8} {'avg_ms':>12} "
        f"{'pct_loop':>10} {'upd_calls':>10} {'upd_avg_ms':>12}"
    )
    print("-" * 104)
    for row in rows:
        update_avg = row["mpc_update_avg_ms"]
        update_avg_text = f"{cast(float, update_avg):.6f}" if update_avg is not None else ""
        print(
            f"{cast(str, row['section']):32} "
            f"{cast(float, row['total_ms']):12.3f} "
            f"{cast(int, row['calls']):8d} "
            f"{cast(float, row['avg_ms']):12.6f} "
            f"{cast(float, row['pct_loop']):10.2f} "
            f"{cast(int, row['mpc_update_calls']):10d} "
            f"{update_avg_text:>12}"
        )


def _write_json(path: Path, metadata: list[ProfileRunMetadata], rows: list[dict[str, Any]]) -> None:
    payload = {
        "metadata": [asdict(item) for item in metadata],
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=200, help="Measured simulation cycles.")
    parser.add_argument("--warmup-cycles", type=int, default=20, help="Unmeasured warmup cycles.")
    parser.add_argument("--sim-dt", type=float, default=0.1, help="Simulation timestep.")
    parser.add_argument("--mpc-dt", type=float, default=1.0, help="MPC update period.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of deterministic repeats.")
    parser.add_argument("--seed", type=int, default=2, help="Deterministic RNG seed.")
    parser.add_argument(
        "--disable-log-records",
        action="store_true",
        help="Skip ClosedLoopLogRecord construction/appends to isolate logging overhead.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be > 0")
    if args.warmup_cycles < 0:
        parser.error("--warmup-cycles must be >= 0")
    if args.repeat <= 0:
        parser.error("--repeat must be > 0")
    return args


def main() -> None:
    args = parse_args()
    all_metadata: list[ProfileRunMetadata] = []
    all_rows: list[dict[str, Any]] = []

    for run_index in range(1, args.repeat + 1):
        metadata, rows = _run_profile_once(args, run_index=run_index)
        all_metadata.append(metadata)
        all_rows.extend(rows)
        _print_run(metadata, rows)

    if args.json is not None:
        _write_json(args.json, all_metadata, all_rows)
        print(f"\nWrote JSON profile results to {args.json}")
    if args.csv is not None:
        _write_csv(args.csv, all_rows)
        print(f"Wrote CSV profile results to {args.csv}")


if __name__ == "__main__":
    main()
