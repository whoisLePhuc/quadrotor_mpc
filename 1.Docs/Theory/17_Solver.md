---
title: 'Chương 17: Solver & Real-Time'
chapter: 17
tags:
- ccmpc
- solver
- real-time
- clarabel
- performance
phase: implementation
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 17
- Solver & Real-Time
- Ch.17
---

## 17.1 Introduction

The CC-MPC optimization must be solved **in real time** — the solver must complete before the next control cycle begins (~60 ms at 16 Hz for the vision-based system, ~71 ms for the full framework with communication).

This chapter covers the solver selection, problem structure, and performance characteristics that make real-time operation possible.

## 17.2 Problem Type

The original papers formulate nonlinear MPC problems. The CVXPY implementation documented here uses sequential linearization and solves a **Quadratic Program (QP)** at each iMPC iteration:

$$\begin{aligned}
\min_{\mathbf{z}} \quad & \frac{1}{2}\mathbf{z}^T\mathbf{H}\mathbf{z} + \mathbf{g}^T\mathbf{z} \\
\text{s.t.} \quad & \mathbf{A}_{\text{eq}}\mathbf{z} = \mathbf{b}_{\text{eq}} \\
& \mathbf{A}_{\text{ineq}}\mathbf{z} \leq \mathbf{b}_{\text{ineq}}
\end{aligned}$$

**Problem dimensions** (for $N=30$, max_obs=2):
- Variables: ~459
- Equality constraints: $N \times 9 = 270$ (dynamics) + 9 (initial state) = 279
- Inequality constraints: $N \times \text{max\_obs} + \text{bounds}$ ≈ 60 + bounds

## 17.3 Solver Comparison

| Solver | Method | Practical note |
|--------|--------|----------------|
| **CLARABEL** | Interior-point conic solver with quadratic objectives | Project default; benchmark on target hardware |
| OSQP | ADMM / first-order QP solver | Useful comparison for repeated QPs |
| ECOS | Interior-point SOCP solver | Applicable after conic reformulation where available |
| SCS | First-order conic solver | General-purpose comparison, usually with looser accuracy/time trade-offs |
| CVXOPT | Interior-point | Additional open-source comparison |
| MOSEK | Commercial interior-point solver | Licensed comparison option |

There is no platform-independent “fastest” or “most robust” ordering; canonicalization, sparsity, scaling, tolerances, warm starts, and hardware can change the result.

### Why CLARABEL?

The CC-MPC implementation uses CLARABEL because:
1. **Native QP support**: No conversion to SOCP needed
2. **Good accuracy**: Second-order method with tight tolerances
3. **Potentially competitive speed**: must be established by benchmarking this implementation on the target hardware
4. **Open source**: No licensing issues
5. **CVXPY integration**: First-class support

**CLARABEL solver settings**:
```python
solver_opts = {
 'max_iter': 100,
 'tol_gap_abs': 1e-5,
 'tol_gap_rel': 1e-5,
 'tol_feas': 1e-5,
}
```

## 17.4 DPP and Problem Caching

**Disciplined Parametrized Programming (DPP)** is CVXPY's mechanism for efficient re-solving. A DPP-compliant problem:
1. Has fixed problem structure (same variables, constraints)
2. Only **parameter values** change between solves
3. CVXPY can **cache the canonicalization** (conversion to solver format)

This means:
- First solve: Full canonicalization + solver call
- Subsequent solves: Only update parameters + solver call (no re-canonicalization)

### DPP Constraints

To maintain DPP compliance:
- **Allowed**: `Parameter @ Variable` (e.g., `a_param @ states[:3, k+1]`)
- **Forbidden**: `Parameter @ Parameter` (would be nonlinear in parameters)
- **Forbidden**: `Variable @ Variable` (would be non-convex quadratic)

**Workaround for chance constraint RHS**: The term $\mathbf{a}^T\hat{\mathbf{p}}_o$ (parameter × parameter) is pre-computed and folded into a single scalar RHS parameter:

```python
# Instead of: a @ (p_mav - p_obs) >= rhs (a, p_obs are both params — DPP violation)
# Use: a @ p_mav - 1 + slack >= rhs_combined
# where rhs_combined = rhs + a @ p_obs (pre-computed scalar)
rhs_combined = float(rhs) + float(a_vec @ p_obs)
```

## 17.5 Solver Performance Analysis

### Timing Breakdown (from papers)

| Component | Mean Time (ms) | Framework |
|-----------|---------------|-----------|
| CCNMPC solve (2 robots) | 14.3 | Zhu 2019, FORCES Pro on Intel i7 |
| Full framework (2 robots) | 71.3 | Zhu 2019 |
| VIO (state estimation) | ~10 | Lin 2020 |
| Obstacle detection + tracking | < 8 (75th pctl) | Lin 2020 |
| MPC solve (vision) | < 22 (75th pctl) | Lin 2020 |

### Scaling with Number of Robots

| # Robots | Mean CC-MPC Time (ms) | Strategy |
|----------|----------------------|----------|
| 2 | 14.3 | Any |
| 4 | 14.4 | DC |
| 6 | 16.2 | DC |
| 16 | 24.7 | DC |

The reported distributed-planning computation grows slowly in these scenarios because each robot solves a local problem. However:
1. Each robot solves its own QP independently
2. The 2019 multi-robot results should not be explained by the 2020 vision system's separate “two closest obstacles” limit
3. A fixed `max_obs` project implementation caps QP size but may ignore relevant collision constraints and therefore requires an explicit obstacle-selection safety policy

## 17.6 Feasibility and Robustness

### Why Infeasibility Occurs

- Too many close obstacles → chance constraints cannot all be satisfied
- Large uncertainty → RHS of chance constraints too large
- Goal behind dense obstacle cluster
- Aggressive noise levels overwhelming the planner

### Mitigation Strategies

1. **Soft chance constraints**: Slack can remove infeasibility caused by those particular inequalities, but hard dynamics, state, control, and FOV constraints can still make the QP infeasible
2. **Retry with fresh initialization**: Reset the guess trajectory once
3. **Safety fallback**: Decelerate/hover and continue replanning, matching the 2019 paper; do not use an unfiltered go-to-goal PID near obstacles
4. **Source-paper context**: In the 2019 FORCES Pro experiment, 2.8% of solves were infeasible and the longest infeasible period was 0.45 s. These figures are not predictions for the CVXPY QP.

### Slack Penalty Tuning

The slack penalty $\rho$ balances safety vs. feasibility:
- Too small: Slack may be used freely, eroding the intended collision margin
- Too large: Numerical conditioning can deteriorate and progress toward feasibility may become difficult
- **Project initial value**: $\rho = 1000$; it must be tuned and validated while monitoring actual slack usage, conditioning, and collision margins

## 17.7 Alternative Solvers: ACADO and Forces Pro

The original papers used different solvers:

### ACADO Toolkit (Lin, Zhu & Alonso-Mora, 2020)

- C++ code generation for NMPC
- Generates optimized C solver from symbolic problem description
- Can use SQP (Sequential Quadratic Programming) or RTI (Real-Time Iteration)
- **Trade-off**: Fast runtime but inflexible (hard to modify problem online)

### Forces Pro (Zhu & Alonso-Mora, 2019)

- Commercial NMPC solver
- Generates tailored interior-point solver
- Supports nonlinear dynamics directly
- Very fast tailored code generation, but the 2019 paper reports a mean 14.3 ms CCNMPC planning step in its two-robot experiment; do not replace that measured result with an unsupported sub-millisecond claim
- **Trade-off**: Commercial license, less flexible than CVXPY

### CVXPY + CLARABEL (This Implementation)

- Open source, pure Python
- Flexible: Easy to modify cost/constraints
- Performance must be measured independently; the papers' FORCES Pro and ACADO runtimes are not CLARABEL results
- **Trade-off**: Slower than Forces Pro, but more maintainable

## 17.8 Real-Time Scheduling

The vision-based system runs on an NVIDIA Jetson TX2 with VIO at 15 Hz, depth processing at 60 Hz, and an MPC sampling time of 60 ms. A scheduling budget for this project should be built from measured component traces rather than inferred from one fixed timeline. An illustrative budget is:

```
Timeline (one control cycle, ~60 ms budget):
├─ latest estimator output and covariance acquisition
├─ obstacle detection/tracking update
├─ covariance and obstacle prediction
├─ iMPC/solver iterations
├─ safety validation and command transmission
└─ deadline monitoring before the next 60 ms control release
```

The paper reports component rates and runtime distributions but does not establish the exact project thread schedule above. Concurrency, data timestamps, and deadline behavior must be validated in the implementation.

## 17.9 Implementation Considerations

### Numerical Stability

1. **Normalization**: Angles are in radians, positions in meters — no extreme values
2. **Conditioning**: $\mathbf{Q}_g$, $\mathbf{R}$ are diagonal with reasonable values (0.1–30)
3. **Initialization**: Warm-start from previous solution prevents solver from starting far from optimum
4. **Scaling**: Normalize variables and residuals when state and cost magnitudes differ substantially; not every state must have artificial bounds merely to improve conditioning

### Convergence Tuning

The iMPC convergence tolerance (0.01) is set based on:
- Position accuracy needed (~1 cm)
- Solver precision (1e-5)
- Practical trade-off: tighter tolerance → more iterations → slower

The “2–3 iterations” target is an implementation hypothesis until benchmark logs show its distribution across scenarios, initializations, and obstacle geometries.

## 17.10 Prerequisites and Related Chapters

> [!info] Prerequisites
- [[16_Optimization|Ch.16: Optimization Formulation]] — QP formulation
- Convex optimization (QP, DPP)

> [!info] Used In
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — The solve loop
- [[18_Implementation_Notes|Ch.18: Implementation Notes]] — CVXPY integration
