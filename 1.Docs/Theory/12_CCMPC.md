---
title: 'Chương 12: Chance-Constrained MPC'
chapter: 12
tags:
- ccmpc
- chance-constrained
- collision-avoidance
- optimization
phase: control-theory
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 12
- Chance-Constrained MPC
- Ch.12
---

## 12.1 Introduction

Chance-Constrained Model Predictive Control (CC-MPC) extends standard MPC by incorporating **probabilistic constraints** that account for uncertainty in state estimation, motion disturbances, and obstacle sensing.

Instead of requiring *deterministic* collision avoidance (impossible with unbounded Gaussian noise), CC-MPC guarantees:

> The probability of collision with each obstacle at each planning step is below $\delta$.

This is the central algorithm in both papers:
- Zhu & Alonso-Mora (2019): CC-MPC for inter-robot + robot-obstacle collision avoidance
- Lin, Zhu & Alonso-Mora (2020): Vision-based CC-MPC with FOV constraints

## 12.2 Problem Formulation

### Stochastic Dynamics

$$\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) + \boldsymbol{\omega}_k, \quad \mathbf{x}_0 \sim \mathcal{N}(\hat{\mathbf{x}}_0, \boldsymbol{\Gamma}_0)$$

where $\boldsymbol{\omega}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$ is the process noise.

### Chance-Constrained Optimization Problem

For a single robot $i$, over horizon $N$:

$$\begin{aligned}
\min_{\hat{\mathbf{x}}^{1:N}, \mathbf{u}^{0:N-1}} \quad & \sum_{k=0}^{N-1} J^k(\hat{\mathbf{x}}^k, \mathbf{u}^k) + J^N(\hat{\mathbf{x}}^N) \\
\text{s.t.} \quad & \mathbf{x}^0 = \hat{\mathbf{x}}(0), \quad \hat{\mathbf{x}}^k = \mathbf{f}(\hat{\mathbf{x}}^{k-1}, \mathbf{u}^{k-1}) \\
& \mathbb{P}(\mathbf{x}^k \notin \mathcal{C}_{ij}^k) \geq 1 - \delta_r, \quad \forall j \neq i \\
& \mathbb{P}(\mathbf{x}^k \notin \mathcal{C}_{io}^k) \geq 1 - \delta_o, \quad \forall o \\
& \mathbf{u}^{k-1} \in \mathcal{U}, \quad \hat{\mathbf{x}}^k \in \mathcal{X} \\
& \forall k \in \{1, \ldots, N\}
\end{aligned}$$

where:
- $J^k$: Stage cost at step $k$
- $J^N$: Terminal cost
- $\delta_r, \delta_o$: Collision probability thresholds for robots and obstacles
- $\mathcal{C}_{ij}, \mathcal{C}_{io}$: Collision regions
- $\mathcal{U}, \mathcal{X}$: Admissible control and state sets

## 12.3 Cost Function

### Terminal Cost (Goal Navigation)

$$J^N(\hat{\mathbf{x}}^N) = \ell^N \|\mathbf{p}_g - \hat{\mathbf{p}}^N\|$$

where $\mathbf{p}_g$ is the goal position and $\ell^N$ is the terminal weight.

In the CC-MPC implementation, this becomes a **quadratic cost** on position error:

$$J^N = (\hat{\mathbf{p}}^N - \mathbf{p}_g)^T \mathbf{Q}_g (\hat{\mathbf{p}}^N - \mathbf{p}_g)$$

with $\mathbf{Q}_g = \text{diag}(30, 30, 20)$ (higher penalty on horizontal error).

### Stage Costs

The total stage cost has four components:

$$J^k = J_u^k + J_c^k + J_\psi^k$$

#### Control Effort

$$J_u^k(\mathbf{u}^k) = \|\mathbf{u}^k\|_{\mathbf{R}}^2 = \mathbf{u}^{kT}\mathbf{R}\mathbf{u}^k$$

where $\mathbf{R} = \text{diag}(0.1, 0.1, 0.5, 0.1)$ penalizes control usage.

#### Collision Potential Field

To improve flight safety, a smooth logistic cost pushes the MAV away from obstacles:

$$J_{c,o}^k(\hat{\mathbf{p}}^k) = \frac{Q_o}{1 + \exp(\lambda_o(d_o^k - r_o))}$$

where:
- $d_o^k = \|\hat{\mathbf{p}}^k - \hat{\mathbf{p}}_o^k\|$: Distance to obstacle $o$
- $Q_o$: Maximum cost (when very close)
- $\lambda_o$: Steepness (sharpness of transition)
- $r_o$: Distance at which cost = $Q_o/2$ (threshold)

**Properties**:
- $J_{c,o}^k = Q_o/2$ at $d = r_o$
- $J_{c,o}^k \to Q_o$ as $d \to -\infty$ (inside obstacle)
- $J_{c,o}^k \to 0$ as $d \to \infty$ (far away)
- Gradient: $\nabla_{\hat{\mathbf{p}}} J_{c,o}^k = -\lambda_o Q_o \frac{\exp(\lambda_o(d-r_o))}{(1+\exp(\lambda_o(d-r_o)))^2} \mathbf{n}_o$
- Smooth derivative → well-behaved in optimization

**Linearization for QP**: In the iMPC framework, this cost is linearized around the guess trajectory:

$$J_{c,o}^k(\hat{\mathbf{p}}^k) \approx J_{c,o}^k(\hat{\mathbf{p}}_{\text{guess}}^k) + \nabla J_{c,o}^k(\hat{\mathbf{p}}_{\text{guess}}^k)^T (\hat{\mathbf{p}}^k - \hat{\mathbf{p}}_{\text{guess}}^k)$$

Only the gradient term affects the QP (constant term dropped). The gradient is clipped to max 5.0 to prevent solver instability.

#### Yaw Alignment

$$J_\psi^k(\psi^k) = Q_\psi (\psi^k - \bar{\psi}^k)^2$$

where $\bar{\psi}^k = \arctan2(\hat{v}_y^k, \hat{v}_x^k)$ is the motion direction angle. This encourages the camera to face the direction of travel, which is important for vision-based obstacle detection.

## 12.4 Collision Conditions

### Inter-Robot Collision

Robot $i$ and robot $j$ (both modeled as spheres with radii $r_i, r_j$) are in collision if:

$$\mathcal{C}_{ij} = \{\mathbf{x}_i \mid \|\mathbf{p}_i - \mathbf{p}_j\| \leq r_i + r_j\}$$

### Robot-Obstacle Collision

Robot $i$ and obstacle $o$ (ellipsoid with axes $(a_o, b_o, c_o)$ and rotation $\mathbf{R}_o$) are in collision if:

$$\mathcal{C}_{io} = \{\mathbf{x}_i \mid \|\mathbf{p}_i - \mathbf{p}_o\|_{\boldsymbol{\Omega}_{io}} \leq 1\}$$

where:

$$\boldsymbol{\Omega}_{io} = \mathbf{R}_o^T \text{diag}\left(\frac{1}{(a_o+r_i)^2}, \frac{1}{(b_o+r_i)^2}, \frac{1}{(c_o+r_i)^2}\right)\mathbf{R}_o$$

Note: The robot radius $r_i$ is incorporated into the ellipsoid (enlarged ellipsoid approximation).

## 12.5 Deterministic Reformulation

The stochastic problem is transformed into a **deterministic QP** by:

1. **Linearizing** nonlinear collision conditions
2. **Applying** Lemma 2 (Gaussian chance constraint → deterministic inequality)
3. **Propagating** uncertainty using last-loop trajectory

### Inter-Robot Constraint

$$\mathbf{a}_{ij}^T(\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_j) - b_{ij} \geq \text{erf}^{-1}(1-2\delta_r)\sqrt{2\mathbf{a}_{ij}^T(\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_j)\mathbf{a}_{ij}}$$

where:
- $\mathbf{a}_{ij} = \frac{\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_j}{\|\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_j\|}$ (unit normal)
- $b_{ij} = r_i + r_j$ (combined radius)

### Robot-Obstacle Constraint

$$\mathbf{n}_o^T\boldsymbol{\Omega}_{io}^{1/2}(\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_o) - 1 \geq \text{erf}^{-1}(1-2\delta_o)\sqrt{2\mathbf{n}_o^T\boldsymbol{\Omega}_{io}^{1/2}(\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_o)\boldsymbol{\Omega}_{io}^{1/2\;T}\mathbf{n}_o}$$

where:
- $\mathbf{n}_o = \frac{\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_o}{\|\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_o\|}$
- $\boldsymbol{\Omega}_{io}^{1/2}$: Cholesky factor ($\boldsymbol{\Omega}_{io}^{1/2}\boldsymbol{\Omega}_{io}^{1/2\;T} = \boldsymbol{\Omega}_{io}$)

### Implementation Form (DPP-Compliant)

For CVXPY optimization (DPP = Disciplined Parametrized Programming), the constraint is structured as:

$$\underbrace{(\mathbf{n}_o^T\boldsymbol{\Omega}^{1/2})}_{\text{parameter } \mathbf{a}} \hat{\mathbf{p}}_i - 1 + s \geq \underbrace{\text{erf}^{-1}(1-2\delta)\sqrt{2\mathbf{n}_o^T\boldsymbol{\Omega}^{1/2}(\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_o)\boldsymbol{\Omega}^{1/2\;T}\mathbf{n}_o} + (\mathbf{n}_o^T\boldsymbol{\Omega}^{1/2})\hat{\mathbf{p}}_o}_{\text{parameter } \text{rhs}}$$

where $s \geq 0$ is a slack variable (soft constraint) with penalty weight 1000.0.

## 12.6 Complete Deterministic MPC

The final tractable optimization problem:

$$\begin{aligned}
\min_{\hat{\mathbf{x}}^{1:N}, \mathbf{u}^{0:N-1}} \quad & \underbrace{(\hat{\mathbf{p}}^N - \mathbf{p}_g)^T\mathbf{Q}_g(\hat{\mathbf{p}}^N - \mathbf{p}_g)}_{\text{Terminal cost}} \\
& + \sum_{k=0}^{N-1} \underbrace{\|\mathbf{R}^{1/2}\mathbf{u}^k\|^2}_{\text{Control effort}} \\
& + \sum_{k=1}^{N} \underbrace{\nabla J_c^k \cdot \hat{\mathbf{p}}^k}_{\text{Collision grad (linearized)}} \\
& + \sum_{k=0}^{N-1} \underbrace{Q_\psi(\psi^k - \bar{\psi}^k)^2}_{\text{Yaw alignment}} \\
& + \sum_{k=1}^{N} \sum_{o} \underbrace{1000 \cdot s_o^k}_{\text{Slack penalty}} \\
\text{s.t.} \quad & \hat{\mathbf{x}}^{k+1} = \mathbf{A}^k\hat{\mathbf{x}}^k + \mathbf{B}^k\mathbf{u}^k + \mathbf{C}^k \quad \text{(linearized dynamics)} \\
& \mathbf{a}_{io}^{k\;T}\hat{\mathbf{p}}^k - 1 + s_o^k \geq \text{rhs}_{io}^k \quad \text{(chance constraint)} \\
& \mathbf{u}^k \in \mathcal{U}, \quad \hat{\mathbf{x}}^k \in \mathcal{X} \\
& s_o^k \geq 0
\end{aligned}$$

## 12.7 Field of View (FOV) Constraints

For vision-based obstacle avoidance, the planned trajectory must keep obstacles within the camera's FOV:

$$\text{FOV}^k = \{\mathbf{p} \mid \mathbf{n}_j^k \cdot \mathbf{p} \leq m_j^k, \; j = 1,\ldots,5\}$$

The five half-spaces in the **body frame** (camera forward = +x):

| # | Half-space | Meaning | $\mathbf{n}_{\text{body}}$ | $m$ |
|---|-----------|---------|--------------------------|-----|
| 1 | Left bound | $y \geq -x \tan(\alpha_h/2)$ | $[-\tan(\alpha_h/2), -1, 0]$ | 0 |
| 2 | Right bound | $y \leq x \tan(\alpha_h/2)$ | $[-\tan(\alpha_h/2), 1, 0]$ | 0 |
| 3 | Bottom bound | $z \geq -x \tan(\alpha_v/2)$ | $[-\tan(\alpha_v/2), 0, -1]$ | 0 |
| 4 | Top bound | $z \leq x \tan(\alpha_v/2)$ | $[-\tan(\alpha_v/2), 0, 1]$ | 0 |
| 5 | Max depth | $x \leq d_{\text{max}}$ | $[1, 0, 0]$ | $d_{\text{max}}$ |

where $\alpha_h, \alpha_v$ are horizontal/vertical FOV angles (e.g., 87° × 58° for Intel RealSense D435i).

**Transformation to world frame**: For a body-frame point $\mathbf{p}_{\text{body}} = \mathbf{R}_{\text{yaw}}^T(\mathbf{p} - \mathbf{p}_{\text{cam}})$, the constraint becomes:

$$\mathbf{n}_j^T\mathbf{R}_{\text{yaw}}^T(\mathbf{p} - \mathbf{p}_{\text{cam}}) \leq m_j \implies (\mathbf{R}_{\text{yaw}}\mathbf{n}_j)^T\mathbf{p} \leq m_j + (\mathbf{R}_{\text{yaw}}\mathbf{n}_j)^T\mathbf{p}_{\text{cam}}$$

Implementation: To reduce constraint count, only the **worst-violated half-space** is enforced at each step.

## 12.8 Multi-Robot Coordination

Three strategies for obtaining other robots' future positions (from Zhu & Alonso-Mora, 2019):

### 1. Constant Velocity (CV) — No Communication

Each robot predicts others using a constant velocity model:

$$\begin{bmatrix} \hat{\mathbf{p}}_j^k \\ \hat{\mathbf{v}}_j^k \end{bmatrix} = \begin{bmatrix} \mathbf{I} & \Delta t\mathbf{I} \\ \mathbf{0} & \mathbf{I} \end{bmatrix}^k \begin{bmatrix} \hat{\mathbf{p}}_j^0 \\ \hat{\mathbf{v}}_j^0 \end{bmatrix}$$

$$\boldsymbol{\Sigma}_{j,\text{pv}}^k = \mathbf{F}_j^k \boldsymbol{\Sigma}_{j,\text{pv}}^0 \mathbf{F}_j^{k\;T} + \mathbf{Q}_{j,\text{pv}}^k$$

**Pros**: Fully distributed, no communication needed 
**Cons**: Prediction mismatch → collisions (0.56 m minimum distance vs 0.6 m safe distance)

### 2. Sequential Planning (SP) — With Communication

Robots plan sequentially by priority: robot $i$ avoids plans $\mathcal{T}_j$ of robots $j < i$.

**Pros**: High coordination, cooperative trajectories 
**Cons**: Computation grows linearly with number of robots (115 ms for 6 robots)

### 3. Distributed with Communication (DC)

At each time step, each robot avoids the **previous** plans of all others:

Robot $i$ at time $t$ avoids $\mathcal{T}_j^{t-\Delta t}$ for all $j \neq i$.

**Pros**: Scalable computation (16.2 ms for 6 robots, 24.7 ms for 16 robots), safe trajectories 
**Cons**: Less cooperative than SP (longer trajectories)

## 12.9 Algorithm Summary

```
Algorithm: CC-MPC (one control cycle)
──────────────────────────────────────
Input: x̂₀ (state estimate), Γ₀ (covariance), 
 obstacles (p̂ₒ, v̂ₒ, Σₒ), goal (p_g)

1. Build initial guess from previous solution (warm-start)
 - Shift previous trajectory by 1 step
 - If no previous, use straight line toward goal

2. Propagate uncertainty (Eq. 19):
 For k = 0, ..., N-1:
 Γ_{k+1} = F_k Γ_k F_k^T + W·dt
 Σ_{k+1} = Γ_{k+1}[0:3, 0:3]

3. iMPC loop (max 5 iterations):
 a. Linearize dynamics at guess: A_k, B_k, C_k
 b. Compute Ω^{1/2} for each obstacle
 c. Predict obstacle positions over horizon
 d. Compute chance constraint params (a, rhs)
 e. Compute logistic cost gradient
 f. Compute FOV constraint params
 g. Solve QP (CVXPY + CLARABEL)
 h. If converged (max change < 0.01), break

4. Return: x*_trajectory, u*_sequence

5. Apply first control u*₀ to quadrotor
```

## 12.10 Implementation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $N$ | 30 | Horizon steps (1.8 s / 0.06 s) |
| $\Delta t$ | 0.06 s | Time step |
| $\delta_r, \delta_o$ | 0.03 | Collision threshold (97% confidence) |
| $r_i$ | 0.4 m | MAV collision radius |
| Max iMPC iters | 5 | Iterative MPC convergence |
| Convergence tol | 0.01 | Max state change for convergence |
| Slack penalty | 1000 | Soft constraint weight |
| Max obstacles | 2 | Closest obstacles fed to MPC |

## 12.11 Prerequisites and Related Chapters

> [!info] Prerequisites
- [[02_Quadrotor_Dynamics|Ch.2: Quadrotor Dynamics]] (Quadrotor Dynamics)
- [[08_Linearization|Ch.8: Linearization]] (Linearization)
- [[10_State_Space_Model|Ch.10: State-Space Model]] (State-Space Model)
- [[11_MPC|Ch.11: Model Predictive Control]] (MPC)
- [[13_Chance_Constraints|Ch.13: Chance Constraints]] (Chance Constraints)
- [[14_Covariance_Propagation|Ch.14: Covariance Propagation]] (Covariance Propagation)

> [!info] Used In
- [[15_Obstacle_Avoidance|Ch.15: Obstacle Avoidance]] (Obstacle Avoidance)
- [[16_Optimization|Ch.16: Optimization Formulation]] (Optimization Formulation)
- [[18_Implementation_Notes|Ch.18: Implementation Notes]] (Implementation Notes)
