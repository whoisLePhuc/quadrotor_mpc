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

> With hard constraints and under the Gaussian, independence, prediction, and linearization assumptions, the constructed upper bound on collision probability for each modeled obstacle at each planning step is at most $\delta$.

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

For Zhu & Alonso-Mora (2019), the source expression is normalized by the initial goal distance:

$$J_i^N(\hat{\mathbf{x}}_i^N) = \ell_i^N \frac{\|\mathbf{p}_{ig} - \hat{\mathbf{p}}_i^N\|}{\|\mathbf{p}_{ig} - \hat{\mathbf{p}}_i^0\|}$$

where $\mathbf{p}_g$ is the goal position and $\ell^N$ is the terminal weight.

In the CC-MPC implementation, this becomes a **quadratic cost** on position error:

$$J^N = (\hat{\mathbf{p}}^N - \mathbf{p}_g)^T \mathbf{Q}_g (\hat{\mathbf{p}}^N - \mathbf{p}_g)$$

with $\mathbf{Q}_g = \text{diag}(30, 30, 20)$ (higher penalty on horizontal error).

### Stage Costs

The 2020 stage cost has three components, in addition to the terminal goal cost:

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
- $J_{c,o}^k$ approaches $Q_o$ as its argument becomes strongly negative; because Euclidean distance is nonnegative, the attainable maximum depends on $r_o$ and $\lambda_o$
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

Robot $i$ and obstacle $o$ (ellipsoid with axes $(a_o, b_o, c_o)$) are in collision if:

$$\mathcal{C}_{io} = \{\mathbf{x}_i \mid \|\mathbf{p}_i - \mathbf{p}_o\|_{\boldsymbol{\Omega}_{io}} \leq 1\}$$

where:

$$\boldsymbol{\Omega}_{io} = \mathbf{R}_{OW}^T \text{diag}\left(\frac{1}{(a_o+r_i)^2}, \frac{1}{(b_o+r_i)^2}, \frac{1}{(c_o+r_i)^2}\right)\mathbf{R}_{OW},$$

where $\mathbf{R}_{OW}$ maps a world-frame displacement into obstacle principal-axis coordinates. Equivalently, if $\mathbf{R}_{WO}$ maps obstacle coordinates to world coordinates, then $\mathbf{R}_{OW}=\mathbf{R}_{WO}^T$.

Note: The robot radius $r_i$ is incorporated into the ellipsoid (enlarged ellipsoid approximation).

## 12.5 Deterministic Reformulation

The chance constraints are transformed into deterministic inequalities by the following steps. The source papers retain nonlinear dynamics and solve an NMPC/NLP; the project obtains a QP only after additionally linearizing dynamics and nonlinear costs.

1. **Linearizing** nonlinear collision conditions
2. **Applying** Lemma 2 (Gaussian chance constraint → deterministic inequality)
3. **Propagating** uncertainty using last-loop trajectory

### Inter-Robot Constraint

$$\mathbf{a}_{ij}^T(\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_j) - b_{ij} \geq \text{erf}^{-1}(1-2\delta_r)\sqrt{2\mathbf{a}_{ij}^T(\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_j)\mathbf{a}_{ij}}$$

where:
- $\mathbf{a}_{ij} = \frac{\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_j}{\|\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_j\|}$ (unit normal)
- $b_{ij} = r_i + r_j$ (combined radius)

### Robot-Obstacle Constraint

Choose a transformation $\mathbf{U}_{io}$ satisfying

$$\boxed{\mathbf{U}_{io}^T\mathbf{U}_{io}=\boldsymbol{\Omega}_{io}}.$$

For example, if $\mathbf{L}=\operatorname{chol}(\boldsymbol{\Omega}_{io})$ is lower triangular with $\mathbf{L}\mathbf{L}^T=\boldsymbol{\Omega}_{io}$, use $\mathbf{U}_{io}=\mathbf{L}^T$. Define the tight transformed-space normal used in the 2019 derivation:

$$\mathbf{n}_{io}=\frac{\mathbf{U}_{io}(\hat{\mathbf{p}}_i-\hat{\mathbf{p}}_o)}{\|\mathbf{U}_{io}(\hat{\mathbf{p}}_i-\hat{\mathbf{p}}_o)\|}.$$

Then the deterministic upper-bound constraint is

$$\boxed{\mathbf{n}_{io}^T\mathbf{U}_{io}(\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_o) - 1 \geq z_{1-\delta_o}\sqrt{\mathbf{n}_{io}^T\mathbf{U}_{io}(\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_o)\mathbf{U}_{io}^T\mathbf{n}_{io}}}$$

where $z_{1-\delta_o}=\Phi^{-1}(1-\delta_o)=\sqrt{2}\,\operatorname{erf}^{-1}(1-2\delta_o)$. The 2020 paper writes the normal from the untransformed mean direction; that remains a unit-half-space bound but is generally less tight for anisotropic ellipsoids. The implementation should state explicitly which variant it reproduces.

### Implementation Form (DPP-Compliant)

For CVXPY optimization (DPP = Disciplined Parametrized Programming), the constraint is structured as:

$$\underbrace{(\mathbf{U}^T\mathbf{n})^T}_{\mathbf{a}^T}\hat{\mathbf{p}}_i - 1 + s \geq \underbrace{z_{1-\delta}\sqrt{\mathbf{n}^T\mathbf{U}(\boldsymbol{\Sigma}_i+\boldsymbol{\Sigma}_o)\mathbf{U}^T\mathbf{n}}+\mathbf{a}^T\hat{\mathbf{p}}_o}_{\text{precomputed rhs}}$$

where $s\geq0$ is an optional project-specific relaxation. Whenever $s>0$, the original probability bound is not guaranteed; therefore slack must be monitored and must not be described as hard CC-MPC safety.

## 12.6 Project-Specific Sequential Convex MPC

The following QP is the project's sequential-convex implementation, not the nonlinear program solved directly by the two papers:

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

For vision-based obstacle avoidance, the 2020 paper constrains predicted MAV positions to remain inside the currently sensed FOV/depth volume. This is not a constraint that forces every obstacle to remain visible.

$$\text{FOV}^k = \{\mathbf{p} \mid \mathbf{n}_j^k \cdot \mathbf{p} \leq m_j^k, \; j = 1,\ldots,5\}$$

The five half-spaces are most naturally written in the **camera frame** ($X_C$ forward, $Y_C$ right, $Z_C$ down):

| # | Half-space | Meaning | $\mathbf{n}_{\text{camera}}$ | $m$ |
|---|-----------|---------|--------------------------|-----|
| 1 | Left bound | $y \geq -x \tan(\alpha_h/2)$ | $[-\tan(\alpha_h/2), -1, 0]$ | 0 |
| 2 | Right bound | $y \leq x \tan(\alpha_h/2)$ | $[-\tan(\alpha_h/2), 1, 0]$ | 0 |
| 3 | Bottom bound | $z \geq -x \tan(\alpha_v/2)$ | $[-\tan(\alpha_v/2), 0, -1]$ | 0 |
| 4 | Top bound | $z \leq x \tan(\alpha_v/2)$ | $[-\tan(\alpha_v/2), 0, 1]$ | 0 |
| 5 | Max depth | $x \leq d_{\text{max}}$ | $[1, 0, 0]$ | $d_{\text{max}}$ |

where $\alpha_h, \alpha_v$ are horizontal/vertical FOV angles (e.g., 87° × 58° for Intel RealSense D435i).

**Transformation to world frame**: Let $\mathbf{R}_C^W$ include the calibrated camera-to-body extrinsic and current body attitude. For $\mathbf{p}_C=\mathbf{R}_C^{W\,T}(\mathbf{p}_W-\mathbf{p}_{\text{cam}}^W)$, the constraint becomes:

$$\mathbf{n}_j^T\mathbf{R}_C^{W\,T}(\mathbf{p}_W - \mathbf{p}_{\text{cam}}^W) \leq m_j \implies (\mathbf{R}_C^W\mathbf{n}_j)^T\mathbf{p}_W \leq m_j + (\mathbf{R}_C^W\mathbf{n}_j)^T\mathbf{p}_{\text{cam}}^W.$$

The paper uses a simplified pose-based construction; a yaw-only approximation is acceptable only when roll/pitch and camera extrinsics are intentionally neglected and that approximation is documented.

All five inequalities define an intersection and must be enforced to guarantee FOV membership. A worst-violated-only active-set implementation is acceptable only if it iterates and verifies all five inequalities before accepting the final trajectory.

## 12.8 Multi-Robot Coordination

Three strategies for obtaining other robots' future positions (from Zhu & Alonso-Mora, 2019):

### 1. Constant Velocity (CV) — No Communication

Each robot predicts others using a constant velocity model:

$$\begin{bmatrix} \hat{\mathbf{p}}_j^k \\ \hat{\mathbf{v}}_j^k \end{bmatrix} = \begin{bmatrix} \mathbf{I} & \Delta t\mathbf{I} \\ \mathbf{0} & \mathbf{I} \end{bmatrix}^k \begin{bmatrix} \hat{\mathbf{p}}_j^0 \\ \hat{\mathbf{v}}_j^0 \end{bmatrix}$$

$$\boldsymbol{\Sigma}_{j,\text{pv}}^k = \mathbf{F}_j^k \boldsymbol{\Sigma}_{j,\text{pv}}^{k-1} \mathbf{F}_j^{k\;T} + \mathbf{Q}_{j,\text{pv}}^k$$

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
 Γ_{k+1} = F_k Γ_k F_k^T + W_{d,k}
 Σ_{k+1} = Γ_{k+1}[0:3, 0:3]

3. iMPC loop (max 5 iterations):
 a. Linearize dynamics at guess: A_k, B_k, C_k
 b. Compute U for each obstacle such that U^T U = Ω
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
| $N$ | 30 | Project configuration; papers use 20 steps/1.0 s and 25 steps/1.5 s |
| $\Delta t$ | 0.06 s | Time step |
| $\delta_r, \delta_o$ | 0.03 | Collision threshold (97% confidence) |
| $r_i$ | 0.4 m | MAV collision radius |
| Max iMPC iters | 5 | Iterative MPC convergence |
| Convergence tol | 0.01 | Max state change for convergence |
| Slack penalty | 1000 | Project tuning value, not a paper-derived guarantee |
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
