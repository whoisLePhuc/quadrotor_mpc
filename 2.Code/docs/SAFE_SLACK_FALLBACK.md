# Safe slack acceptance and fallback

Stage 6 adds an explicit safety supervisor between the primary NMPC controller
and the MuJoCo plant. The optimizer proposes a command; the supervisor decides
whether that command is fresh, numerically valid and acceptable under the
configured degraded-slack policy.

This layer does not prove collision avoidance. It prevents known-invalid or
stale optimizer output from being applied silently and records every degraded
decision.

## Runtime boundary

```text
beliefs + goal
      |
      v
primary NMPC solve
      |
      v
acceptance gates ---- reject ----> fallback hierarchy
      |                                 |
      +--------------- applied command -+
```

`SafeFallbackController` decorates the existing belief-based `Controller`.
Consequently, all applied commands—including fallback commands—are returned in
the same `ControlSolution` contract.

## Acceptance gates

The primary command is accepted only when all enabled checks pass:

1. The backend reports solver success and final primal/dual residuals remain
   below `maximum_solver_residual`.
2. Joint-risk accounting reports `BUDGET_OK`.
3. The command is finite and within actuator bounds, allowing only configured
   numerical tolerance. The accepted command is clipped to the exact bounds.
4. Slack is nonnegative and `chance_residual + slack` satisfies the nonlinear
   residual tolerance.
5. Maximum safety slack does not exceed
   `maximum_acceptable_slack_m`.
6. The end-to-end call returns before `solve_deadline_s` when
   `reject_on_deadline_miss` is enabled.

Solver exceptions are caught at this boundary and converted to a normalized
fallback solution. They do not terminate the plant loop.

## Slack semantics

Two thresholds have deliberately different meanings:

- `guarantee_slack_tolerance_m`: a numerical zero used only to decide whether a
  solved step is *eligible* for a chance-constraint guarantee.
- `maximum_acceptable_slack_m`: an operational degraded-mode limit. A primary
  solution below this limit may be applied, but any positive slack remains
  `NOT_GUARANTEED_POSITIVE_SLACK`.

Therefore:

```text
slack <= guarantee tolerance
    -> primary may be GUARANTEE_ELIGIBLE

guarantee tolerance < slack <= acceptable limit
    -> primary may run, explicitly not guaranteed

slack > acceptable limit
    -> reject primary and activate fallback
```

`GUARANTEE_ELIGIBLE` is not itself a proof of episode-wide safety. It only
means the local runtime gates did not invalidate the chance-constraint claim.

## Fallback hierarchy

Fallback state is reset at startup, Reset and Run again.

| Level | Command source | Activation |
|---:|---|---|
| 1 | `HOLD_LAST_ACCEPTED` | A short transient reject and a previously accepted command exists |
| 2 | `POSITION_HOLD_PD` | Continued reject or no accepted command exists |
| 3 | `EMERGENCY_HOVER` | Reject count reaches the configured emergency threshold or position hold cannot be evaluated |

Level 2 latches the estimated position and yaw at fallback entry. It uses a
bounded cascade:

\[
a_{xy}^{des}=K_p(p_{xy}^{hold}-p_{xy})-K_dv_{xy}
\]

\[
T_{dev}=m\left(K_{pz}(z^{hold}-z)-K_{dz}v_z\right)
\]

The desired horizontal acceleration is converted to bounded roll/pitch targets,
then attitude/rate PD terms produce bounded body torques.

Level 3 commands hover thrust deviation and bounded angular-rate damping. It is
an emergency containment action, not a navigation controller.

## Status and evidence

Every step records:

- primary solver status and success;
- primary solver iteration count and final primal/dual residuals;
- applied `command_source`;
- accept/reject decision;
- fallback active flag, level, reason and consecutive reject count;
- solve time and deadline miss;
- `safety_assurance_status`;
- the original chance residual, slack, risk and covariance horizons.

Recorder event logs add `fallback_entered` when the level changes and
`fallback_recovered` when primary control resumes. Replay reads the same fields
without running NMPC.

## Configuration

```yaml
controller:
  safety_fallback:
    enabled: true
    solve_deadline_s: 0.10
    reject_on_deadline_miss: true
    guarantee_slack_tolerance_m: 1.0e-6
    maximum_acceptable_slack_m: 0.08
    constraint_tolerance_m: 1.0e-6
    maximum_solver_residual: 1.0e-3
    command_bound_tolerance: 1.0e-6
    hold_last_command_steps: 1
    emergency_after_consecutive_rejections: 20
```

The supplied CC-MPC scenario enables the supervisor. Legacy deterministic and
estimator-only configurations omit the block and remain unchanged.

## Acceptance tests

Stage 6 tests inject:

- backend `success=false`;
- a solver exception;
- excessive slack;
- invalid joint-risk status;
- an out-of-bounds command;
- an artificial deadline miss;
- repeated failures through all three fallback levels;
- reset of fallback history;
- a closed-loop MuJoCo fault sequence.

The full nominal regressions must additionally show no collision, no NaN and no
unexplained fallback activation.
