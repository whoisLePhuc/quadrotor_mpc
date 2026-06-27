---
title: 'Chương 16: Optimization Formulation'
chapter: 16
tags:
- ccmpc
- optimization
- qp
- cvxpy
phase: implementation
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 16
- Optimization Formulation
- Ch.16
---

## 16.1 Introduction

The CC-MPC problem is ultimately a **numerical optimization** problem. This chapter details the mathematical formulation as a Quadratic Program (QP) and the techniques that make it tractable for real-time operation.

## 16.2 Problem Structure

The CC-MPC optimization at each control cycle has the structure:

$$\begin{aligned}
\min_{\mathbf{z}} \quad & \frac{1}{2}\mathbf{z}^T\mathbf{H}\mathbf{z} + \mathbf{g}^T\mathbf{z} \\
\text{s.t.} \quad & \mathbf{A}_{\text{eq}}\mathbf{z} = \mathbf{b}_{\text{eq}} \quad \text{(dynamics)} \\
& \mathbf{A}_{\text{ineq}}\mathbf{z} \leq \mathbf{b}_{\text{ineq}} \quad \text{(constraints)}
\end{aligned}$$

where $\mathbf{z} \in \mathbb{R}^{(N+1)n_x + N n_u + N n_{\text{slack}}}$ is the vector of all decision variables:

$$\mathbf{z} = \begin{bmatrix} \hat{\mathbf{x}}^0 \\ \hat{\mathbf{x}}^1 \\ \vdots \\ \hat{\mathbf{x}}^N \\ \mathbf{u}^0 \\ \mathbf{u}^1 \\ \vdots \\ \mathbf{u}^{N-1} \\ \mathbf{s}^0 \\ \vdots \\ \mathbf{s}^{N-1} \end{bmatrix}$$

## 16.3 Variable Dimensions

For the quadrotor CC-MPC:

| Variable | Symbol | Dimension | Count |
|----------|--------|-----------|-------|
| State | $\hat{\mathbf{x}}^k$ | 9 | $N+1$ |
| Control | $\mathbf{u}^k$ | 4 | $N$ |
| Slack (per obstacle) | $s_o^k$ | 1 | $N \times \text{max\_obs}$ |

**Total variables**: $(N+1) \times 9 + N \times 4 + N \times \text{max\_obs}$

For $N = 30$, max_obs = 2: $31 \times 9 + 30 \times 4 + 30 \times 2 = 279 + 120 + 60 = 459$ variables.

## 16.4 Cost Function (Quadratic Form)

### Terminal Cost

$$J^N = (\hat{\mathbf{p}}^N - \mathbf{p}_g)^T\mathbf{Q}_g(\hat{\mathbf{p}}^N - \mathbf{p}_g)$$

In CVXPY: `opt.quad_form(terminal_position, Qg)`

### Control Effort

$$J_u^k = \|\mathbf{R}^{1/2}\mathbf{u}^k\|^2 = \mathbf{u}^{kT}\mathbf{R}\mathbf{u}^k$$

In CVXPY: `opt.sum_squares(R @ controls[:, k])`

### Reference Tracking

A soft reference trajectory (straight line to goal) is used:

$$J_{\text{ref}}^k = 0.5 \|\hat{\mathbf{p}}^{k+1} - \mathbf{p}_{\text{ref}}^k\|^2$$

where $\mathbf{p}_{\text{ref}}^k = \mathbf{p}_0 + \frac{k+1}{N}(\mathbf{p}_g - \mathbf{p}_0)$.

### Yaw Alignment

$$J_\psi^k = Q_\psi(\psi^k - \bar{\psi}^k)^2$$

In CVXPY: `Q_psi * (states[8, k] - yaw_ref[k])**2`

### Slack Penalty

Soft chance constraints add a linear penalty:

$$J_{\text{slack}} = \sum_{k,o} \rho \cdot s_o^k$$

where $\rho = 1000.0$ is the penalty weight.

### Logistic Collision Cost (Linearized)

The logistic cost $J_{c,o}^k = \frac{Q_o}{1 + \exp(\lambda_o(d_o - r_o))}$ is linearized around the guess:

$$\nabla J_{c,o}^k(\hat{\mathbf{p}}_{\text{guess}}^k)^T \hat{\mathbf{p}}^k$$

This becomes a **linear term** in the QP (added to $\mathbf{g}^T\mathbf{z}$).

## 16.5 Equality Constraints: Dynamics

The linearized dynamics are **affine equality constraints**:

$$\hat{\mathbf{x}}^{k+1} = \mathbf{A}_k\hat{\mathbf{x}}^k + \mathbf{B}_k\mathbf{u}^k + \mathbf{C}_k, \quad k = 0, \ldots, N-1$$

In CVXPY:
```python
constraints.append(states[:, k+1] == A_params[k] @ states[:, k] 
 + B_params[k] @ controls[:, k] 
 + C_params[k])
```

**Initial condition**: `states[:, 0] == initial_state`

## 16.6 Inequality Constraints

### Chance Constraints

For each time step $k$ and each obstacle $o$:

$$\mathbf{a}_{k,o}^T \hat{\mathbf{p}}^{k+1} - 1 + s_{k,o} \geq \text{rhs}_{k,o}$$

In CVXPY (DPP-compliant):
```python
a_k_i @ states[:3, k+1] - 1.0 + slack_k_i >= rhs_k_i
```

where $\mathbf{a}_{k,o}$ and $\text{rhs}_{k,o}$ are **parameters** (computed before solving from the guess trajectory), not variables.

### State Bounds

| State | Lower | Upper | Purpose |
|-------|-------|-------|---------|
| $z$ (altitude) | 0.1 | — | Ground avoidance |
| $v_x, v_y$ | $-v_{\text{max}}$ | $v_{\text{max}}$ | Speed limit |
| $v_z$ | $-v_{z,\text{max}}$ | $v_{z,\text{max}}$ | Vertical speed limit |
| $\phi$ | $-0.5$ rad | $0.5$ rad | Roll limit (~29°) |
| $\theta$ | $-0.5$ rad | $0.5$ rad | Pitch limit (~29°) |

### Control Bounds

| Control | Lower | Upper | Purpose |
|---------|-------|-------|---------|
| $\phi_c$ | $-0.35$ rad | $0.35$ rad | Max roll command (~20°) |
| $\theta_c$ | $-0.35$ rad | $0.35$ rad | Max pitch command (~20°) |
| $v_{zc}$ | $-3.0$ m/s | $3.0$ m/s | Max vertical velocity |
| $\dot{\psi}_c$ | $-0.8$ rad/s | $0.8$ rad/s | Max yaw rate |

### FOV Constraints (if enabled)

For each step $k$, the worst-violated half-space:

$$(\mathbf{R}_{\text{yaw}}\mathbf{n}_j)^T \hat{\mathbf{p}}^{k+1} \leq m_j + (\mathbf{R}_{\text{yaw}}\mathbf{n}_j)^T \hat{\mathbf{p}}^k + s_{\text{fov}}$$

where $j$ is the index of the most-violated constraint at the guess trajectory.

## 16.7 DPP Compliance

CVXPY's **Disciplined Parametrized Programming (DPP)** requires that:
1. Parameters and variables cannot be multiplied together
2. The problem structure must be fixed (no changing variable/constraint counts)

To satisfy DPP, the chance constraint is reformulated:

$$\underbrace{(\mathbf{n}_o^T\boldsymbol{\Omega}^{1/2})}_{\text{Parameter}} \hat{\mathbf{p}}^{k+1} - 1 + s \geq \underbrace{\text{erf}^{-1}(1-2\delta)\sqrt{2\mathbf{n}_o^T\boldsymbol{\Omega}^{1/2}(\boldsymbol{\Sigma} + \boldsymbol{\Sigma}_o)\boldsymbol{\Omega}^{1/2\;T}\mathbf{n}_o} + (\mathbf{n}_o^T\boldsymbol{\Omega}^{1/2})\hat{\mathbf{p}}_o}_{\text{Parameter}}$$

The RHS (including $\mathbf{a}^T\hat{\mathbf{p}}_o$) is pre-computed as a scalar parameter. The constraint is then simply $\mathbf{a}^T\hat{\mathbf{p}}^{k+1} - 1 + s \geq \text{rhs}$.

## 16.8 Solver Selection

| Solver | Type | Performance | When to Use |
|--------|------|-------------|-------------|
| **CLARABEL** | Interior-point | Fast, accurate | **Default** for CC-MPC |
| OSQP | ADMM (first-order) | Very fast, less accurate | When speed > accuracy |
| ECOS | Interior-point | Moderate | Fallback option |
| SCS | ADMM (first-order) | Moderate | Conic problems |

CLARABEL is preferred because:
- Handles QP directly (no SOCP conversion)
- Good accuracy (second-order method)
- Competitive speed for problem sizes ($\sim 500$ variables)

**Solver options used**:
```python
solver_opts = {
 'max_iter': 100,
 'tol_gap_abs': 1e-5,
 'tol_gap_rel': 1e-5,
 'tol_feas': 1e-5,
}
```

## 16.9 Warm-Starting

Between consecutive MPC solves, the previous solution is used as a warm-start:

1. Shift the previous state trajectory left by 1 step
2. Duplicate the last state for the new final step
3. Shift the control sequence similarly
4. Align the first state with the current measurement

This provides an excellent initial guess, reducing iMPC iterations from 5 to 2–3.

For the first solve (no previous solution), a straight line initialization is used:
- States: Linear interpolation from current position toward goal
- Controls: All zeros

## 16.10 Feasibility Handling

When the optimization is infeasible (e.g., too many obstacles, excessive noise), the system falls back to:

1. **Retry with fresh initialization** (once)
2. If still infeasible → **PID fallback controller**
 - P controller for yaw toward goal
 - P controller for pitch (speed regulation)
 - P controller for altitude
 - Maximum safety: reduced speed, gentle maneuvers

From experiments: infeasible solutions occurred in only 2.8% of cycles, with the longest infeasible period being 9 steps (0.45 s).

## 16.11 Computational Performance

| Component | Mean Time | Notes |
|-----------|-----------|-------|
| CC-MPC solve (2 robots) | 14.3 ms | Core optimization |
| Full framework (2 robots) | 71.3 ms | Includes estimation, prediction, comm |
| CC-MPC solve (6 robots, DC) | 16.2 ms | Scales well |
| CC-MPC solve (16 robots, DC) | 24.7 ms | Linear scaling in constraints |

## 16.12 Prerequisites and Related Chapters

> [!info] Prerequisites
- [[11_MPC|Ch.11: Model Predictive Control]] — Basic MPC formulation
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — The complete problem
- Convex optimization (QP, constraints)

> [!info] Used In
- [[17_Solver|Ch.17: Solver & Real-Time]] — Solver implementation details
- [[18_Implementation_Notes|Ch.18: Implementation Notes]] — CVXPY code structure
