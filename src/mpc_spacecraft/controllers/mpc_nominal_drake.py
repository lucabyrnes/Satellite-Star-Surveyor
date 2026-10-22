"""Nominal MPC controller using Drake as optimization backend."""

from typing import Any, cast

import numpy as np
import quaternion as qu
from pydrake.all import MathematicalProgram, OsqpSolver  

from .error_dynamics_providers import AffineErrorDynamicsStep, ErrorDynamicsProvider
from mpc_spacecraft.guidance.sun_tracker import TimeLike
from mpc_spacecraft.utilities.array_view_generic import BatchArrayView
from mpc_spacecraft.utilities.utils import (
    FloatArray, RotationState, RotationErrorState
)

IDX_STATE_QUAT = RotationState.slice_of("quat")
IDX_STATE_OMEGA = RotationState.slice_of("omega")

IDX_ROTERR_THETA = RotationErrorState.slice_of("error_angle")
IDX_ROTERR_OMEGA = RotationErrorState.slice_of("omega")


def _add_affine_dynamics_constraint(
    prog: MathematicalProgram,
    dx_kp1: Any,
    dx_k: Any,
    du_k: Any,
    step: AffineErrorDynamicsStep,
) -> None:
    """Add dx[k+1] = A dx[k] + B du[k] + c without symbolic NumPy math."""
    variables = np.concatenate((dx_kp1, dx_k, du_k))
    coefficients = np.hstack((np.eye(step.A.shape[0]), -step.A, -step.B))
    prog.AddLinearEqualityConstraint(coefficients, step.c, variables)


def _add_attitude_boundary_constraint(
    prog: MathematicalProgram,
    theta_k: Any,
    slack_k: Any,
    alpha_k: FloatArray,
    b_k: FloatArray,
) -> None:
    """Add 2 alphaᵀ theta - slack <= b without symbolic NumPy math."""
    variables = np.concatenate((theta_k, np.asarray([slack_k], dtype=object)))
    coefficients = np.concatenate((2.0 * alpha_k, np.array([-1.0])))
    prog.AddLinearConstraint(coefficients, -np.inf, float(b_k), variables)


def _add_biased_quadratic_cost(
    prog: MathematicalProgram,
    variables: Any,
    bias: FloatArray,
    weight: FloatArray,
) -> None:
    """Add (variables + bias)ᵀ weight (variables + bias)."""
    prog.AddQuadraticErrorCost(weight, -bias, variables)


class NominalMPC:
    """
    Model Predictive Control (MPC) controller with manual formulation.

    Uses Drake's MathematicalProgram as the optimization backend, but all
    constraints, costs, and dynamics are manually implemented.
    """

    def __init__(
        self,
        horizon: int,
        Q: FloatArray,
        R: FloatArray,
        error_dynamics_provider: ErrorDynamicsProvider,
        Q_terminal: FloatArray | None = None,
        u_min: FloatArray | None = None,
        u_max: FloatArray | None = None,
        max_sqp_iters: int = 1,
    ):
        """
        Initialize MPC controller.

        Args:
            horizon: Prediction horizon.
            dynamics: Instance of SpacecraftDynamics.
            dt: Sampling time.
            Q: State (error) cost matrix (6x6).
            R: Control (error) cost matrix (3x3).
            Q_terminal: Terminal state (error) cost matrix (defaults to Q).
            u_min, u_max: Control bounds (absolute).
            x_min, x_max: State bounds (absolute, currently unused).
            max_sqp_iters: Number of sequential QP iterations per MPC solve
                           for the goal-only (non-tracking) case.
        """
        self.horizon: int = horizon
        self.err_dynamics_provider = error_dynamics_provider

        self.state_dim: int = RotationState.SIZE
        self.control_dim: int = 3

        self.Q = Q
        self.R = R
        self.Q_terminal = Q_terminal if Q_terminal is not None else Q

        self.u_min = u_min
        self.u_max = u_max

        self.max_sqp_iters: int = max_sqp_iters

        self.warm_start_x : BatchArrayView[RotationState] | None = None  # shape (N+1, 7)
        self.warm_start_u = None  # shape (N, 3)

        self.solver = OsqpSolver()
        self.w_slack = 10e4
        self.last_solve_success = False
        self.last_fallback_used = False

    # -------------------------------------------------------------------------
    # Nominal trajectory builder (for linearization & warm-start)
    # -------------------------------------------------------------------------
    def _find_initial_guesses(
        self,
        x0: RotationState,
        x_goal: RotationState | None = None,
        x_ref: BatchArrayView[RotationState] | None = None,
        u_ref: FloatArray | None = None,
    ) -> tuple[BatchArrayView[RotationState], FloatArray]:
        """
        Build an initial *nominal* guess for (state, control) along the horizon.

        This is used for:
          - Nominal trajectory for dynamics linearization.
          - Initial guesses for the optimization variables (warm start).

        Returns:
            initial_x: (horizon+1, state_dim)
            initial_u: (horizon, control_dim)
        """
        if x_goal is None and x_ref is None:
            raise ValueError("Need to provide either trajectory or terminal goal")

        initial_x = RotationState.batch_zeros(self.horizon + 1)
        initial_u = np.array([])

        # 1) Try to reuse previous MPC solution as a warm start
        if self.warm_start_u is not None and self.warm_start_x is not None:
            if x_goal is not None:
                x_terminal = x_goal.data
            elif x_ref is not None:
                x_terminal = x_ref[-1].data
            else:
                raise ValueError("Need to provide either trajectory or terminal goal")

            # Shift previous solution one step and append terminal/zero
            initial_u = np.concatenate(
                [self.warm_start_u[1:], np.zeros((1, self.control_dim))], axis=0
            )

            initial_x.data[:-1] = self.warm_start_x[1:].data
            initial_x.data[-1] = x_terminal

        # 2) Full reference provided → just use it as nominal
        elif x_ref is not None:
            initial_x = x_ref.copy()
            if u_ref is not None:
                initial_u = u_ref
            else:
                initial_u = np.zeros((self.horizon, self.control_dim))

        # 3) Only terminal goal given → interpolate between x0 and x_goal
        elif x_goal is not None:
            arr_interp = np.linspace(0.0, 1.0, self.horizon + 1)

            # Quaternion slerp for attitudes
            x_q = qu.quaternion(*x0.quat).normalized()
            x_q_goal = qu.quaternion(*x_goal.quat).normalized()
            x_q_guess = qu.slerp(x_q, x_q_goal, 0.0, 1.0, arr_interp)
            x_q_guess = qu.as_float_array(x_q_guess).squeeze()

            # Linear interpolation for angular velocity
            x_omega = x0.omega
            x_omega_goal = x_goal.omega
            x_omega_guess = (1.0 - arr_interp[:, None]) * x_omega[None, :] + arr_interp[
                :, None
            ] * x_omega_goal[None, :]

            initial_x.quat[:] = x_q_guess
            initial_x.omega[:] = x_omega_guess

            initial_u = np.zeros((self.horizon, self.control_dim))

        return initial_x, initial_u

    # -------------------------------------------------------------------------
    # Cost reference builder (what the optimizer should want)
    # -------------------------------------------------------------------------
    def _build_ref_trajectory(
        self,
        x0: RotationState,
        x_goal: RotationState | None = None,
        x_ref: BatchArrayView[RotationState] | None = None,
        u_ref: FloatArray | None = None,
    ) -> tuple[BatchArrayView[RotationState], FloatArray]:
        """
        Build the *cost reference* trajectory (x_cost_traj, u_cost_traj).

        This trajectory defines what 'zero error' means in the cost and in
        the error coordinates used by state_error / state_from_error.

        Cases:
            - If x_ref is provided: track that trajectory directly.
            - If only x_goal is provided: cost reference is the constant goal.

        Returns:
            x_cost_traj: (horizon+1, state_dim)
            u_cost_traj: (horizon, control_dim)
        """
        if x_goal is None and x_ref is None:
            raise ValueError("Need to provide either trajectory or terminal goal")

        # Tracking a given reference trajectory
        if x_ref is not None:
            x_cost_traj = x_ref
            if u_ref is not None:
                u_cost_traj = u_ref
            else:
                u_cost_traj = np.zeros((self.horizon, self.control_dim))

        # Only terminal goal: cost reference is the constant goal state
        else:
            assert x_goal is not None
            x_cost_traj = RotationState.batch_zeros(self.horizon + 1)
            x_cost_traj.data = np.tile(x_goal.data, (self.horizon + 1, 1))
            u_cost_traj = np.zeros((self.horizon, self.control_dim))

        return x_cost_traj, u_cost_traj

    # -------------------------------------------------------------------------
    # Main solve method
    # -------------------------------------------------------------------------
    def _solve(
        self,
        x0: RotationState,
        x_goal: RotationState | None = None,
        x_ref: BatchArrayView[RotationState] | None = None,
        u_ref: FloatArray | None = None,
        *,
        current_epoch_utc: TimeLike,
    ) -> tuple[FloatArray, BatchArrayView[RotationState], bool]:
        """
        Solve the MPC optimization problem using multiple shooting in
        error coordinates, with optional SQP / real-time iterations.

        Args:
            x0: Initial state (7D).
            x_goal: Terminal goal state (7D), for regulation / goal-only MPC.
            x_ref: Reference state trajectory (horizon+1 x state_dim).
            u_ref: Reference control trajectory (horizon x control_dim).

        Returns:
            u_opt: Optimal control sequence (horizon x control_dim).
            x_opt: Optimal state trajectory (horizon+1 x state_dim).
            success: Whether optimization succeeded in at least one SQP iteration.
        """
        if x_goal is None and x_ref is None:
            raise ValueError("Need to provide either trajectory or terminal goal")

        n_err = 6

        # 1) Cost reference
        x_cost_traj, u_cost_traj = self._build_ref_trajectory(
            x0=x0, x_goal=x_goal, x_ref=x_ref, u_ref=u_ref
        )

        # 2) Nominal trajectory for linearization and warm-start
        x_nom_traj, u_nom_traj = self._find_initial_guesses(
            x0=x0, x_goal=x_goal, x_ref=x_ref, u_ref=u_ref
        )

        # For tracking problems (explicit x_ref) keep cost reference fixed
        # just do one linear MPC solve. For goal-only problems do SQP.
        if x_ref is not None:
            max_iters = 1
        else:
            max_iters = max(1, self.max_sqp_iters)

        best_success = False
        best_x_opt: BatchArrayView[RotationState] | None = None
        best_u_opt: FloatArray | None = None

        # SQP / real-time iteration loop
        for _ in range(max_iters):
            prog = MathematicalProgram()

            dx = prog.NewContinuousVariables(self.horizon + 1, n_err, "dx_nom")
            du = prog.NewContinuousVariables(self.horizon, self.control_dim, "du_nom")

            x_bias_k = self.err_dynamics_provider.state_error_batch(
                x_nom_traj, x_cost_traj
            )
            u_bias_k = u_nom_traj - u_cost_traj

            #    Initial condition in error coordinates (w.r.t nom reference)
            dx0_nom = self.err_dynamics_provider.state_error(x0, x_nom_traj[0])
            prog.AddLinearEqualityConstraint(dx[0], dx0_nom.data)

            #    Linearized error dynamics:
            for k in range(self.horizon):
                step = self.err_dynamics_provider.affine_error_dynamics_step(
                    x_nom_k=x_nom_traj[k],
                    x_nom_kp1=x_nom_traj[k + 1],
                    u_nom_k=u_nom_traj[k],
                )

                _add_affine_dynamics_constraint(
                    prog,
                    dx[k + 1],
                    dx[k],
                    du[k],
                    step,
                )

            # 5) Control constraints (on absolute u = u_norm_traj + du)
            #    We express bounds in terms of error input du.
            if (self.u_min is not None) or (self.u_max is not None):
                for k in range(self.horizon):
                    if self.u_min is not None:
                        lb = self.u_min - u_nom_traj[k]
                    else:
                        lb = -np.inf * np.ones(self.control_dim)

                    if self.u_max is not None:
                        ub = self.u_max - u_nom_traj[k]
                    else:
                        ub = np.inf * np.ones(self.control_dim)

                    prog.AddBoundingBoxConstraint(cast(Any, lb), cast(Any, ub), du[k])

            # OPTIONAL ADDITIONAL CONSTRAINTS (BOUNDARIES, SAFETY MARGIN)
            theta_slack = prog.NewContinuousVariables(self.horizon + 1, "theta_slack")
            prog.AddBoundingBoxConstraint(0.0, np.inf, theta_slack)

            for k in range(self.horizon):
                theta_k = dx[k][IDX_ROTERR_THETA]
                alpha_k, b_k = self.err_dynamics_provider.affine_error_theta_bc(
                    x_nom_traj[k], current_epoch_utc
                )
                _add_attitude_boundary_constraint(
                    prog,
                    theta_k,
                    theta_slack[k],
                    alpha_k,
                    b_k,
                )
                prog.AddQuadraticCost(self.w_slack * theta_slack[k] ** 2)

            #    Quadratic costs in error space
            #    - stage costs: dx(k)^T Q dx(k) + du(k)^T R du(k)
            #    - terminal cost: dx(N)^T Q_terminal dx(N)
            for k in range(self.horizon):
                _add_biased_quadratic_cost(
                    prog,
                    dx[k],
                    x_bias_k[k].data,
                    self.Q,
                )
                _add_biased_quadratic_cost(
                    prog,
                    du[k],
                    u_bias_k[k],
                    self.R,
                )

            # Terminal Cost
            _add_biased_quadratic_cost(
                prog,
                dx[self.horizon],
                x_bias_k[self.horizon].data,
                self.Q_terminal,
            )

            #    Initial guesses in error coordinates
            initial_dx = np.zeros((self.horizon + 1, n_err))
            initial_du = np.zeros((self.horizon, self.control_dim))

            for k in range(self.horizon + 1):
                initial_dx[k] = np.zeros((n_err))

            for k in range(self.horizon):
                initial_du[k] = np.zeros((self.control_dim))

            prog.SetInitialGuess(dx, cast(Any, initial_dx))
            prog.SetInitialGuess(du, cast(Any, initial_du))

            #    Solve QP
            result = self.solver.Solve(prog)

            if not result.is_success():
                # If this iteration fails but a previous one succeeded, keep the best
                if best_success:
                    break
                else:
                    # If nothing ever worked, fall back to zeros (handled after loop)
                    best_success = False
                    best_x_opt = None
                    best_u_opt = None
                    break

            #    Recover optimal error and map back to states / controls
            dx_sol = result.GetSolution(dx)
            du_sol = result.GetSolution(du)

            x_opt = self.err_dynamics_provider.state_from_error_batch(
                RotationErrorState.batch_from_array(dx_sol),
                x_nom_traj,
            )

            u_opt = du_sol + u_nom_traj

            best_success = True
            best_x_opt = x_opt
            best_u_opt = u_opt

            #     Update the NOMINAL trajectory for the next SQP iteration
            x_nom_traj = x_opt.copy()
            u_nom_traj = u_opt.copy()

        # End of SQP loop

        if not best_success:
            # Return zeros if optimization completely failed
            x_opt = RotationState.batch_zeros(self.horizon + 1)
            u_opt = np.zeros((self.horizon, self.control_dim))

            self.warm_start_x = None
            self.warm_start_u = None
            self.last_solve_success = False
            self.last_fallback_used = True
            return u_opt, x_opt, False

        # Save warm start for the next MPC call (absolute trajectories)
        assert best_x_opt is not None
        assert best_u_opt is not None
        self.warm_start_x = best_x_opt
        self.warm_start_u = best_u_opt
        self.last_solve_success = True
        self.last_fallback_used = False

        return best_u_opt, best_x_opt, True

    def get_first_control(
        self,
        x0: RotationState,
        x_ref: BatchArrayView[RotationState] | None = None,
        u_ref: FloatArray | None = None,
        x_goal: RotationState | None = None,
        *,
        current_epoch_utc: TimeLike,
    ) -> FloatArray:
        """
        Solve MPC and return only the first control input (receding horizon).

        You can either:
          - Provide x_ref / u_ref for tracking, or
          - Provide x_goal for goal-only regulation.

        Args:
            x0: Initial state.
            x_ref: Reference state trajectory.
            u_ref: Reference control trajectory.
            x_goal: Terminal goal state.

        Returns:
            First control input u[0].
        """
        u_opt, _, success = self._solve(
            x0,
            x_goal=x_goal,
            x_ref=x_ref,
            u_ref=u_ref,
            current_epoch_utc=current_epoch_utc,
        )

        if not success:
            print("Warning: MPC optimization failed, returning zero control")
            return np.zeros(self.control_dim)

        return u_opt[0]
