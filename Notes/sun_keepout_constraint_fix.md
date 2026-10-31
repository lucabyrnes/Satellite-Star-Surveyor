# Sun Keep-Out Constraint Correction

## Summary

The MPC Sun keep-out constraint previously measured the difference between two
complete spacecraft attitudes. A Sun keep-out cone, however, constrains only the
angle between the spacecraft boresight and the Sun direction. The old equation
was therefore roll-dependent and imposed a substantially larger exclusion angle
than the configured value.

This caused otherwise safe attitudes to require large soft-constraint slack.
With the slack penalty set to `1e5`, OSQP reached its iteration limit and the MPC
returned zero control.

The constraint has been replaced with a direct, linearized boresight-to-Sun
condition.

## Notation

- $R_k \in SO(3)$: nominal body-to-inertial attitude at prediction step $k$
- $R_{sun}$: an attitude whose boresight points toward the Sun
- $\hat{s}_I$: inertial-frame unit vector toward the Sun
- $e_B$: body-frame boresight unit vector
- $\delta\theta$: small right attitude perturbation used by the MPC
- $\sigma \ge 0$: soft-constraint slack
- $\gamma$: Sun keep-out half-angle
- $\operatorname{vee}(\cdot)$: conversion from a skew-symmetric matrix to a
  three-component vector

## Previous and Corrected Constraints

| Previous full-attitude constraint | Corrected boresight constraint |
| --- | --- |
| $2\,\operatorname{vee}\!\left(\frac{R_{sun}^{T}R_k-R_k^{T}R_{sun}}{2}\right)^T\delta\theta-\sigma \le 1+\cos\gamma-\operatorname{tr}(R_{sun}^{T}R_k)$ | $\left(e_B\times R_k^T\hat{s}_I\right)^T\delta\theta-\sigma \le \cos\gamma-e_B^TR_k^T\hat{s}_I$ |

Both equations have the generic affine form used by the controller:

$$
2\alpha_k^T\delta\theta-\sigma\le b_k.
$$

For the corrected equation,

$$
\alpha_k=\frac{1}{2}\left(e_B\times R_k^T\hat{s}_I\right)
$$

and

$$
b_k=\cos\gamma-e_B^TR_k^T\hat{s}_I.
$$

## Why the Previous Equation Was Wrong

### 1. It measured full attitude instead of boresight direction

The matrix

$$
R_{sun}^TR_k
$$

describes the complete relative attitude between the spacecraft and a chosen
Sun-pointing attitude. It includes roll about the boresight.

Sun safety does not depend on boresight roll. If the spacecraft rolls while its
boresight remains fixed, its angular separation from the Sun is unchanged. The
old equation could nevertheless change its constraint value because it compared
all three rotational degrees of freedom.

There is also no unique full attitude associated with a direction vector. Many
attitudes point the boresight toward the Sun but differ by roll. Constructing one
arbitrary `R_sun` introduced an unnecessary and physically incorrect roll
reference.

### 2. It did not enforce the configured angle

At the nominal attitude, set $\delta\theta=0$ and $\sigma=0$. The old constraint
then required

$$
\operatorname{tr}(R_{sun}^TR_k)\le 1+\cos\gamma.
$$

If $\phi$ is the complete relative rotation angle, then

$$
\operatorname{tr}(R_{sun}^TR_k)=1+2\cos\phi.
$$

Substitution gives

$$
2\cos\phi\le\cos\gamma.
$$

Therefore, the effective boundary was

$$
\phi\ge\cos^{-1}\!\left(\frac{\cos\gamma}{2}\right),
$$

not $\phi\ge\gamma$.

For the configured $35^\circ$ keep-out angle,

$$
\cos^{-1}\!\left(\frac{\cos 35^\circ}{2}\right)\approx65.8^\circ.
$$

Consequently, a target placed safely at $40^\circ$ from the Sun was treated as a
constraint violation.

### 3. The false violation created a numerical failure

The constraint is soft, so it remained mathematically feasible through the
slack variable $\sigma$. However, the false violation required substantial
slack, and the MPC applies a quadratic slack penalty of `1e5`.

This produced a poorly conditioned QP. OSQP terminated at its 4000-iteration
limit, after which the controller reported:

```text
Warning: MPC optimization failed, returning zero control
```

The optimization problem was not genuinely infeasible. Disabling the old Sun
constraint or drastically reducing the slack penalty allowed the same problem
to solve.

## Derivation of the Corrected Constraint

The inertial spacecraft boresight is

$$
b_I=Re_B.
$$

The cosine of the boresight-to-Sun angle $\phi$ is

$$
\cos\phi=\hat{s}_I^TRe_B.
$$

To remain outside a keep-out cone with half-angle $\gamma$, the spacecraft must
satisfy

$$
\phi\ge\gamma.
$$

Since cosine decreases over the relevant interval, this is equivalent to

$$
\hat{s}_I^TRe_B\le\cos\gamma.
$$

The MPC represents attitude near the nominal attitude using a right
perturbation:

$$
R(\delta\theta)=R_k\operatorname{Exp}([\delta\theta]_\times).
$$

For a small perturbation,

$$
\operatorname{Exp}([\delta\theta]_\times)e_B
\approx e_B+\delta\theta\times e_B.
$$

Substituting this approximation into the keep-out condition gives

$$
\hat{s}_I^TR_k(e_B+\delta\theta\times e_B)\le\cos\gamma.
$$

Using the scalar triple-product identity,

$$
(R_k^T\hat{s}_I)^T(\delta\theta\times e_B)
=\left(e_B\times R_k^T\hat{s}_I\right)^T\delta\theta,
$$

the linearized condition becomes

$$
\left(e_B\times R_k^T\hat{s}_I\right)^T\delta\theta
\le
\cos\gamma-e_B^TR_k^T\hat{s}_I.
$$

Finally, adding nonnegative slack produces the implemented soft constraint:

$$
\boxed{
\left(e_B\times R_k^T\hat{s}_I\right)^T\delta\theta-\sigma
\le
\cos\gamma-e_B^TR_k^T\hat{s}_I
}
$$

At $\delta\theta=0$ and $\sigma=0$, this reduces exactly to

$$
\hat{s}_I^TR_ke_B\le\cos\gamma,
$$

or equivalently

$$
\phi\ge\gamma.
$$

This condition depends only on boresight direction and is invariant to roll
about that boresight.

## Code Changes

The main implementation changes are in:

- `src/mpc_spacecraft/dynamics/rigid_body_error_constraints.py`
  - Linearizes the vector-based boresight keep-out condition.
- `src/mpc_spacecraft/controllers/error_dynamics_adapters.py`
  - Passes the inertial Sun direction directly to the constraint builder rather
    than converting it to an arbitrary Sun-pointing quaternion.
- `tests/test_prediction_adapters.py`
  - Verifies that $40^\circ$ is safe for a $35^\circ$ cone.
  - Verifies that $30^\circ$ violates a $35^\circ$ cone.
  - Verifies that the boundary is invariant to boresight roll.
- `experiments/experiment.py`
  - Provides a three-target mission whose first-to-second slew crosses the Sun
    keep-out region.
  - Retains the default initial +X boresight, so the spacecraft first acquires
    target 1 before beginning the intended inter-target slew.
  - Orients the target pair to retain positive Earth-clearance margin throughout
    both targets' expected acquisition and observation windows.
  - Places target 3 at the south celestial pole, keeping it clear of Earth for
    the default equatorial orbit.

## Verification

After correcting the equation:

- The original first MPC solve succeeds instead of returning zero control.
- Repeated MPC updates through the first-to-second transition succeed.
- The generated mission begins with each of its first two targets $40^\circ$
  from the Sun while the direct path between them crosses the excluded region.
- In the initial 60-second closed-loop check, the spacecraft approached the
  boundary to approximately $36.05^\circ$ without violating the configured
  $35^\circ$ keep-out angle.
- The focused MPC and prediction-adapter test suites pass.
