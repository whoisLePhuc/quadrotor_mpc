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

The CC-MPC problem is ultimately a **numerical optimization** problem. The two source papers solve nonlinear MPC problems. This chapter instead documents the project's sequential-convex approximation, which linearizes the dynamics and nonlinear terms and solves a Quadratic Program (QP) at each iteration.

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

In CVXPY, use either `opt.quad_form(controls[:, k], R)` or a square-root factor `R_sqrt` satisfying `R_sqrt.T @ R_sqrt == R`:

```python
cost += opt.quad_form(controls[:, k], R)
# equivalent: opt.sum_squares(R_sqrt @ controls[:, k])
```

`sum_squares(R @ u)` equals $u^T R^T R u$, not $u^T R u$, unless `R` is already the square-root factor.

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

$$\mathbf{a}_{k,o}^T \hat{\mathbf{p}}^{k+1} - 1 + s_{k,o} \geq \text{rhs}_{k,o},\qquad \mathbf{a}_{k,o}=\mathbf{U}_{k,o}^T\mathbf{n}_{k,o}$$

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

The FOV is an intersection, so all five half-spaces must be enforced at every constrained prediction step:

$$(\mathbf{R}_{\text{cam}}\mathbf{n}_j)^T \hat{\mathbf{p}}^{k+1} \leq m_j + (\mathbf{R}_{\text{cam}}\mathbf{n}_j)^T \mathbf{p}_{\text{cam}},\qquad j=1,\ldots,5.$$

For the paper's current-view formulation, $\mathbf{R}_{\text{cam}}$ and $\mathbf{p}_{\text{cam}}$ are fixed from the current measured pose. A moving cone anchored at $\hat{\mathbf{p}}^k$ is a different, project-specific sequential-convex approximation and must not be attributed directly to Equation (17). If an active-set method is used, the final solution must be checked against all five inequalities.

## 16.7 DPP Compliance

CVXPY's **Disciplined Parametrized Programming (DPP)** permits a parameter-affine expression to multiply a parameter-free expression, which is why `Parameter @ Variable` is allowed here. Products of two parameterized expressions and products of decision variables are not DPP-compliant. The variable and constraint structure must also remain fixed between solves.

To satisfy DPP, the chance constraint is reformulated:

Choose $\mathbf{U}^T\mathbf{U}=\boldsymbol{\Omega}$, $\mathbf{n}=\mathbf{U}(\hat{\mathbf{p}}_{\text{guess}}-\hat{\mathbf{p}}_o)/\|\mathbf{U}(\hat{\mathbf{p}}_{\text{guess}}-\hat{\mathbf{p}}_o)\|$, and $\mathbf{a}=\mathbf{U}^T\mathbf{n}$. Then:

$$\mathbf{a}^T\hat{\mathbf{p}}^{k+1}-1+s\geq \underbrace{z_{1-\delta}\sqrt{\mathbf{n}^T\mathbf{U}(\boldsymbol{\Sigma}+\boldsymbol{\Sigma}_o)\mathbf{U}^T\mathbf{n}}+\mathbf{a}^T\hat{\mathbf{p}}_o}_{\text{precomputed scalar Parameter}}.$$

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

This often provides a useful initial guess and may reduce the number of iMPC iterations. The actual reduction must be measured from solver logs rather than assumed.

For the first solve (no previous solution), a straight line initialization is used:
- States: Linear interpolation from current position toward goal
- Controls: All zeros

## 16.10 Feasibility Handling

When the optimization is infeasible or returns excessive slack, use a safety-oriented recovery sequence:

1. **Retry with fresh initialization** (once)
2. If still infeasible, command deceleration/hover as in the 2019 paper, maintain a validated altitude policy, and continue sensing and replanning.
3. A go-to-goal PID must not be used as the safety fallback unless a separate collision-avoidance safety filter validates its command.

The 2.8% and 0.45 s figures belong to the 2019 FORCES Pro experiment with hard nonlinear constraints; they are context from the paper, not validation of this CVXPY soft-QP recovery policy.

## 16.11 Computational Performance

| Paper/framework result | Reported time | Scope |
|------------------------|---------------|-------|
| CCNMPC solve, 2 robots | 14.3 ms | 2019, FORCES Pro on an Intel i7 |
| Full framework, 2 robots | 71.3 ms | 2019, including estimation, prediction, communication and both solves |
| DC planning, 6 robots | 16.2 ms | 2019 simulation |
| DC planning, 16 robots | 24.7 ms | 2019 simulation |

These are reference-paper measurements, not benchmarks of CVXPY + CLARABEL. The project implementation requires its own benchmark with hardware, solver version, canonicalization time, iteration count, and scenario recorded.

## 16.12 Prerequisites and Related Chapters

> [!info] Prerequisites
- [[11_MPC|Ch.11: Model Predictive Control]] — Basic MPC formulation
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — The complete problem
- Convex optimization (QP, constraints)

> [!info] Used In
- [[17_Solver|Ch.17: Solver & Real-Time]] — Solver implementation details
- [[18_Implementation_Notes|Ch.18: Implementation Notes]] — CVXPY code structure
