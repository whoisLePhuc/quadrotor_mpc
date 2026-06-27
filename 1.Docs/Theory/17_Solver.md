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

The CC-MPC problem is a **Quadratic Program (QP)** with the standard form:

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

| Solver | Method | Speed | Accuracy | Robustness | Notes |
|--------|--------|-------|----------|------------|-------|
| **CLARABEL** | Interior-point | ★★★★ | ★★★★★ | ★★★★★ | Default choice |
| OSQP | ADMM (first-order) | ★★★★★ | ★★★ | ★★★★ | Very fast, less precise |
| ECOS | Interior-point (SOCP) | ★★★ | ★★★★ | ★★★★ | Converts QP to SOCP |
| SCS | ADMM | ★★★ | ★★★ | ★★★ | General cone solver |
| CVXOPT | Interior-point | ★★ | ★★★★★ | ★★★★ | Accurate but slower |
| MOSEK | Interior-point | ★★★★ | ★★★★★ | ★★★★★ | Commercial, fastest |

### Why CLARABEL?

The CC-MPC implementation uses CLARABEL because:
1. **Native QP support**: No conversion to SOCP needed
2. **Good accuracy**: Second-order method with tight tolerances
3. **Competitive speed**: ~14 ms for 2-obstacle problem
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
| CC-MPC solve (2 robots) | 14.3 | Zhu 2019 |
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

The computation time scales **sub-linearly** because:
1. Each robot solves its own QP independently
2. Only the 2 closest obstacles are included per robot
3. The QP size depends on max_obs, not total obstacles

## 17.6 Feasibility and Robustness

### Why Infeasibility Occurs

- Too many close obstacles → chance constraints cannot all be satisfied
- Large uncertainty → RHS of chance constraints too large
- Goal behind dense obstacle cluster
- Aggressive noise levels overwhelming the planner

### Mitigation Strategies

1. **Soft constraints**: Slack variables with penalty $\rho = 1000$ ensure the QP always has a solution
2. **Retry with fresh initialization**: Reset the guess trajectory once
3. **PID fallback**: If still infeasible, use a simple go-to-goal PID controller
4. **Infeasibility statistics**: Only 2.8% of solves were infeasible; longest infeasible period: 0.45 s

### Slack Penalty Tuning

The slack penalty $\rho$ balances safety vs. feasibility:
- Too small ($\rho < 100$): Slack is used freely → unsafe trajectories
- Too large ($\rho > 10000$): QP becomes ill-conditioned → solver may fail
- **Recommended**: $\rho = 1000$ — tested and validated

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
- Very fast (sub-millisecond for small problems)
- **Trade-off**: Commercial license, less flexible than CVXPY

### CVXPY + CLARABEL (This Implementation)

- Open source, pure Python
- Flexible: Easy to modify cost/constraints
- Adequate performance: 14 ms for real problems
- **Trade-off**: Slower than Forces Pro, but more maintainable

## 17.8 Real-Time Scheduling

The vision-based system (Lin et al., 2020) runs on an NVIDIA Jetson TX2:

```
Timeline (one control cycle, ~60 ms budget):
├─ 0-8 ms: VIO state estimation (15 Hz)
├─ 0-8 ms: Obstacle detection + tracking (60 Hz, runs in parallel)
├─ 8-22 ms: CC-MPC solve (main optimization)
├─ 22-24 ms: Command transmission via ROS
└─ 24-60 ms: Idle / next cycle preparation
```

**Parallel execution**: VIO and obstacle detection run in separate threads, overlapping with the MPC solve.

## 17.9 Implementation Considerations

### Numerical Stability

1. **Normalization**: Angles are in radians, positions in meters — no extreme values
2. **Conditioning**: $\mathbf{Q}_g$, $\mathbf{R}$ are diagonal with reasonable values (0.1–30)
3. **Initialization**: Warm-start from previous solution prevents solver from starting far from optimum
4. **Bounded variables**: All states and controls have explicit bounds → bounded feasible set

### Convergence Tuning

The iMPC convergence tolerance (0.01) is set based on:
- Position accuracy needed (~1 cm)
- Solver precision (1e-5)
- Practical trade-off: tighter tolerance → more iterations → slower

Typical convergence: 2–3 iterations out of max 5.

## 17.10 Prerequisites and Related Chapters

> [!info] Prerequisites
- [[16_Optimization|Ch.16: Optimization Formulation]] — QP formulation
- Convex optimization (QP, DPP)

> [!info] Used In
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — The solve loop
- [[18_Implementation_Notes|Ch.18: Implementation Notes]] — CVXPY integration
