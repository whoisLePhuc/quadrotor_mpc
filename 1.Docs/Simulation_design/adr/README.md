# Architecture Decision Records

> Project: Quadrotor CC-MPC Simulation  
> Scope: Architecture Decision Record index  
> Status: Draft  
> Target location: `docs/design/ADR/README.md`

---

## 1. Purpose

This directory stores Architecture Decision Records for the refactored quadrotor CC-MPC simulation.

An ADR records an important architectural decision, the context behind it, the alternatives considered, and the consequences of the decision.

The purpose of ADRs is to make major design decisions explicit instead of leaving them hidden inside code or demo scripts.

---

## 2. What Is an ADR?

An Architecture Decision Record is a short document that answers:

```text
What decision was made?
Why was the decision needed?
What alternatives were considered?
What are the consequences?
What implementation rules follow from the decision?
```

ADRs are especially useful when a project has multiple possible designs and the final choice affects long-term maintainability.

---

## 3. When to Create an ADR

Create an ADR when a decision affects:

```text
public data contracts
module boundaries
runtime architecture
controller and engine separation
state or command definitions
threading model
physics engine abstraction
logging or validation strategy
long-term refactor direction
```

Do not create an ADR for small implementation details such as local variable names, helper function names, or formatting-only changes.

---

## 4. Current ADRs

| ADR | Status | Decision |
|---|---|---|
| `ADR-001-engine-abstraction.md` | Proposed | Use a common `PhysicsEngine` abstraction for ODE and MuJoCo engines |
| `ADR-002-single-thread-vs-mpc-thread.md` | Proposed | Use deterministic single-thread runtime as the reference mode; keep threaded MPC optional |
| `ADR-003-state-vector-definition.md` | Proposed | Use `State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]` as the canonical public state |
| `ADR-004-control-command-definition.md` | Proposed | Use `ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]` as the canonical high-level controller output |

---

## 5. Recommended Reading Order

Read ADRs in this order:

| Order | ADR | Reason |
|---:|---|---|
| 1 | `ADR-003-state-vector-definition.md` | Defines the public state contract used by every module |
| 2 | `ADR-004-control-command-definition.md` | Defines the controller command contract |
| 3 | `ADR-001-engine-abstraction.md` | Explains how ODE and MuJoCo are hidden behind a common engine boundary |
| 4 | `ADR-002-single-thread-vs-mpc-thread.md` | Explains why deterministic single-thread runtime is the reference architecture |

This order follows the dependency chain:

```text
State contract
-> command contract
-> engine boundary
-> runtime/threading policy
```

---

## 6. ADR Status Values

Use the following status values:

| Status | Meaning |
|---|---|
| `Proposed` | Decision is drafted but not yet accepted |
| `Accepted` | Decision is approved and implementation should follow it |
| `Superseded` | Decision was replaced by a newer ADR |
| `Deprecated` | Decision is no longer recommended but may still describe historical context |
| `Rejected` | Decision was considered but not adopted |

Every ADR shall include a `Status` field near the top.

---

## 7. ADR Naming Convention

ADR filenames shall follow this format:

```text
ADR-XXX-short-kebab-case-title.md
```

Examples:

```text
ADR-001-engine-abstraction.md
ADR-002-single-thread-vs-mpc-thread.md
ADR-003-state-vector-definition.md
ADR-004-control-command-definition.md
```

Rules:

```text
use three-digit numbering
do not reuse ADR numbers
use lowercase kebab-case titles
keep filenames stable after creation
```

If a decision is replaced, create a new ADR and mark the old one as `Superseded`.

---

## 8. ADR Template

New ADRs should follow this template:

```markdown
# ADR-XXX: Title

> Status: Proposed  
> Date: YYYY-MM-DD  
> Project: Quadrotor CC-MPC Simulation  
> Related documents:
>
> - `docs/design/...`

---

## 1. Context

Describe the problem and why a decision is needed.

---

## 2. Decision

State the decision clearly.

---

## 3. Alternatives Considered

List alternatives and explain why they were not chosen.

---

## 4. Consequences

Describe positive and negative consequences.

---

## 5. Implementation Rules

List rules that implementation must follow.

---

## 6. Validation

Describe tests or checks needed to verify the decision.

---

## 7. Related Documents

List related design documents.
```

---

## 9. Relationship to Design Documents

ADRs explain why major decisions were made.

Design documents define what the system shall do.

Implementation follows both.

| Design Area | Main Design Doc | ADR |
|---|---|---|
| Data model | `../04_DATA_MODEL.md` | `ADR-003`, `ADR-004` |
| Engine interface | `../05_ENGINE_INTERFACE.md` | `ADR-001` |
| Runtime flow | `../03_RUNTIME_FLOW.md` | `ADR-002` |
| Controller interface | `../06_CONTROLLER_INTERFACE.md` | `ADR-004` |
| Architecture | `../02_ARCHITECTURE.md` | `ADR-001`, `ADR-002`, `ADR-003`, `ADR-004` |

---

## 10. Rules for Modifying ADRs

After an ADR is accepted, do not rewrite its decision silently.

Allowed edits:

```text
fix typos
improve formatting
add links to related documents
clarify wording without changing meaning
```

Not allowed without a new ADR:

```text
changing State9 ordering
changing ControlCommand4 ordering
changing runtime threading decision
changing engine abstraction policy
changing command dispatch semantics
```

If a decision changes, create a new ADR and mark the previous ADR as `Superseded`.

---

## 11. Current Core Decisions

The current architecture depends on these core decisions:

```text
State9 is the canonical public state.
ControlCommand4 is the canonical high-level controller command.
ActuatorCommand4 is separate from ControlCommand4.
Physics engines are hidden behind PhysicsEngine.
ODE and MuJoCo are engine implementations, not separate architectures.
Deterministic single-thread runtime is the reference mode.
Threaded MPC runtime is optional future work.
```

These decisions shall be treated as non-negotiable unless a new ADR supersedes them.

---

## 12. Suggested Future ADRs

Possible future ADRs:

```text
ADR-005-logging-schema-versioning.md
ADR-006-scenario-config-versioning.md
ADR-007-mujoco-actuator-policy.md
ADR-008-estimator-interface.md
ADR-009-quaternion-vs-euler-planning-state.md
ADR-010-solver-backend-selection.md
ADR-011-fallback-controller-policy.md
ADR-012-validation-gate-policy.md
```

Create these only when the decision becomes necessary.

---

## 13. Acceptance Criteria

This ADR index is accepted when:

```text
all current ADRs are listed
ADR reading order is clear
ADR status values are defined
ADR naming convention is defined
ADR template is provided
rules for modifying ADRs are defined
relationship to design docs is clear
```

---

## 14. Summary

This folder records the major architectural decisions of the refactored quadrotor CC-MPC simulation.

The ADRs protect the project from hidden design drift.

Before changing a public contract or architectural boundary, check this folder first.

If the change contradicts an accepted ADR, create a new ADR instead of silently changing the implementation.
