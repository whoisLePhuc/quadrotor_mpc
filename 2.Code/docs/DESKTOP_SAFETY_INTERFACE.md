# Desktop safety interface

Stage 7 connects the belief-based CC-MPC, joint-risk accounting and Stage 6
safety supervisor to the existing native desktop interface. It changes only
presentation and visualization; controller, estimator, risk allocation,
acceptance gates and applied commands remain unchanged.

## Safety boundary

The Qt process receives plain telemetry dictionaries plus a validated,
read-only summary of the active controller policy. It never:

- reads MuJoCo ground truth directly;
- recomputes covariance, risk allocation or chance constraints;
- changes solver output or fallback policy;
- advances the simulation or physics clock.

`native_ui_model.py` is the presentation boundary:

```text
effective YAML policy + telemetry sample
                    |
                    v
       immutable PanelViewState
                    |
        +-----------+-----------+
        |                       |
        v                       v
  Qt status cards       transition alerts
```

Keeping this projection independent of PySide6 makes the safety labels
testable in a headless environment and ensures replay uses exactly the same
interpretation as a live run.

## Status cards

The panel renders six independent cards:

| Card | Source | Important states |
|---|---|---|
| Episode | runtime | `RUNNING`, `PAUSED`, `COMPLETED`, `COLLISION` |
| Applied control | supervisor | `PRIMARY NMPC`, `FALLBACK L1/L2/L3`, `REJECTED` |
| Safety assurance | supervisor | `GUARANTEE ELIGIBLE`, `NOT GUARANTEED`, `DETERMINISTIC` |
| Risk budget | allocator | `BUDGET_OK`, `BUDGET_EXCEEDED`, `INDIVIDUAL`, `DISABLED` |
| Chance constraint | solver/profile | `HARD-SAFE`, `DEGRADED`, `REJECT LIMIT`, `DISABLED` |
| Solver timing | supervisor | `ON TIME`, `DEADLINE MISSED`, `MONITOR ONLY` |

Color is semantic:

- green: the represented gate is currently satisfied;
- blue: informational state;
- amber: degraded or monitored state;
- red: rejected, fallback, missed deadline, invalid budget or collision;
- gray: the feature is disabled or unavailable.

`GUARANTEE ELIGIBLE` is shown only when the controller reports that exact
status. A positive slack or any fallback command is always rendered as
`NOT GUARANTEED`.

## Plots

The bounded history view contains:

1. vehicle position;
2. goal error, clearance, minimum chance residual and maximum slack;
3. terminal position sigma, maximum projected uncertainty and maximum
   tightened safety radius;
4. thrust deviation and body-torque norm;
5. solve time together with the configured acceptance deadline;
6. joint-risk fraction, accepted-command flag and fallback level.

Deterministic profiles leave unavailable chance/risk signals as gaps instead
of fabricating zeros with probabilistic meaning.

## Native 3-D cues

The MuJoCo prediction path uses the runtime assurance state:

- green: guarantee-eligible;
- amber: positive-slack degraded primary solution;
- red: fallback active;
- blue: deterministic or assurance unavailable.

An amber vehicle halo marks positive slack. A red halo marks fallback. These
are status cues only; obstacle collision and tightened safety geometry remain
the physical overlays.

## Transition log and reset

The panel logs only state transitions, not every telemetry sample:

- collision;
- fallback entry/recovery and level changes;
- deadline miss;
- joint-risk budget violation;
- loss of guarantee eligibility;
- episode completion.

Repeated samples in the same state do not duplicate alerts. `Reset` and
`Run again` clear plot history, transition state and the visible alert list,
matching the estimator/controller/supervisor episode reset.

The number of retained alerts is bounded:

```yaml
panel:
  maximum_alerts: 12
```

## Validation

`tests/test_native_ui_integration.py` verifies policy projection, deterministic
and CC-MPC labeling, positive-slack handling, fallback/deadline visibility,
risk-budget failure, transition deduplication and panel-option validation.

The complete GUI still requires the optional PySide6, pyqtgraph, MuJoCo,
CasADi and do-mpc dependencies. The pure presentation model and configuration
validation remain runnable without those packages.
