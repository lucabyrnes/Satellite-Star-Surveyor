"""Analysis module for performance evaluation and metrics."""

from .metrics import (
    compute_tracking_error, compute_control_effort, compute_stability_metrics,
    compute_quaternion_trajectory_errors, compute_velocity_errors,
    compute_performance_index, compare_controllers
)
from .mission_results import (
    MissionArtifactPaths,
    build_derived_timeseries,
    build_mission_analysis_artifact,
    compute_mission_summary,
    quaternion_angle_error_deg,
    save_mission_artifact,
)
from .mekf_verification import (
    InnovationConsistencyResult,
    MEKFVerificationConfig,
    MEKFVerificationResult,
    average_chi_square_bounds,
    mekf_error_state,
    normalized_squared_error,
    run_mekf_verification,
)
from .stability import analyze_closed_loop_stability, compute_lyapunov_function
from .single_target import (
    SingleTargetCaseResult,
    compute_single_target_metrics,
    make_single_target_config,
    make_single_target_table,
    make_target_relative_initial_state,
    perturb_inertia,
    run_single_target_case,
    summarize_disturbance_torque,
    sustained_entry_time,
)
from .timestep_convergence import (
    ClosedLoopConvergenceResult,
    DynamicsConvergenceResult,
    estimate_observed_order,
    quaternion_angular_distance_rad,
    run_closed_loop_convergence,
    run_dynamics_convergence,
    validate_step_grid,
)

__all__ = [
    "compute_tracking_error",
    "compute_control_effort",
    "compute_stability_metrics",
    "compute_quaternion_trajectory_errors",
    "compute_velocity_errors",
    "compute_performance_index",
    "compare_controllers",
    "MissionArtifactPaths",
    "build_derived_timeseries",
    "build_mission_analysis_artifact",
    "compute_mission_summary",
    "quaternion_angle_error_deg",
    "save_mission_artifact",
    "InnovationConsistencyResult",
    "MEKFVerificationConfig",
    "MEKFVerificationResult",
    "average_chi_square_bounds",
    "mekf_error_state",
    "normalized_squared_error",
    "run_mekf_verification",
    "analyze_closed_loop_stability",
    "compute_lyapunov_function",
    "SingleTargetCaseResult",
    "compute_single_target_metrics",
    "make_single_target_config",
    "make_single_target_table",
    "make_target_relative_initial_state",
    "perturb_inertia",
    "run_single_target_case",
    "summarize_disturbance_torque",
    "sustained_entry_time",
    "ClosedLoopConvergenceResult",
    "DynamicsConvergenceResult",
    "estimate_observed_order",
    "quaternion_angular_distance_rad",
    "run_closed_loop_convergence",
    "run_dynamics_convergence",
    "validate_step_grid",
]
