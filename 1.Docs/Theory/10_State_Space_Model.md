---
title: 'Chương 10: State-Space Model'
chapter: 10
tags:
- quadrotor
- control
- state-space
- ltv
phase: foundations
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 10
- State-Space Model
- Ch.10
---

## 10.1 Introduction

The **state-space representation** is the standard framework for describing dynamical systems in control theory. It compactly expresses the quadrotor's evolution:

$$\begin{aligned}
\mathbf{x}_{k+1} &= \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) + \boldsymbol{\omega}_k \\
\mathbf{y}_k &= \mathbf{h}(\mathbf{x}_k) + \boldsymbol{\nu}_k
\end{aligned}$$

where:
- $\mathbf{x}_k \in \mathbb{R}^{n_x}$: State vector
- $\mathbf{u}_k \in \mathbb{R}^{n_u}$: Control input
- $\mathbf{y}_k \in \mathbb{R}^{n_y}$: Measurement/output
- $\mathbf{f}(\cdot)$: State transition function (dynamics)
- $\mathbf{h}(\cdot)$: Observation function
- $\boldsymbol{\omega}_k, \boldsymbol{\nu}_k$: Process and measurement noise

## 10.2 Quadrotor State-Space Definition

### State Vector $\mathbf{x} \in \mathbb{R}^9$

$$\mathbf{x} = \begin{bmatrix} \mathbf{p} \\ \mathbf{v} \\ \boldsymbol{\eta} \end{bmatrix} = \begin{bmatrix} x \\ y \\ z \\ v_x \\ v_y \\ v_z \\ \phi \\ \theta \\ \psi \end{bmatrix}$$

| Index | Symbol | Name | Units |
|-------|--------|------|-------|
| 0 | $x$ | X position (world frame) | m |
| 1 | $y$ | Y position (world frame) | m |
| 2 | $z$ | Z position (world frame) | m |
| 3 | $v_x$ | X velocity (world frame) | m/s |
| 4 | $v_y$ | Y velocity (world frame) | m/s |
| 5 | $v_z$ | Z velocity (world frame) | m/s |
| 6 | $\phi$ | Roll angle | rad |
| 7 | $\theta$ | Pitch angle | rad |
| 8 | $\psi$ | Yaw angle | rad |

### Control Vector $\mathbf{u} \in \mathbb{R}^4$

$$\mathbf{u} = \begin{bmatrix} \phi_c \\ \theta_c \\ v_{zc} \\ \dot{\psi}_c \end{bmatrix}$$

## 10.3 Continuous Dynamics Function

The function $\mathbf{f}_c: \mathbb{R}^9 \times \mathbb{R}^4 \to \mathbb{R}^9$ returns the continuous derivative and consists of three blocks. The discrete transition $\mathbf{f}_d$ is obtained by numerical or exact integration.

$$\dot{\mathbf{x}}=\mathbf{f}_c(\mathbf{x}, \mathbf{u}) = \begin{bmatrix} \mathbf{f}_p(\mathbf{x}) \\ \mathbf{f}_v(\mathbf{x}, \mathbf{u}) \\ \mathbf{f}_\eta(\mathbf{x}, \mathbf{u}) \end{bmatrix}$$

### Position Block $\mathbf{f}_p$

$$\mathbf{f}_p(\mathbf{x}) = \mathbf{v} = \begin{bmatrix} v_x \\ v_y \\ v_z \end{bmatrix}$$

### Velocity Block $\mathbf{f}_v$

$$\mathbf{f}_v(\mathbf{x}, \mathbf{u}) = \begin{bmatrix}
g(\tan\theta \cos\psi + \tan\phi \sin\psi) - k_D v_x \\
g(\tan\theta \sin\psi - \tan\phi \cos\psi) - k_D v_y \\
\frac{1}{\tau_{vz}}(k_{vz} v_{zc} - v_z)
\end{bmatrix}$$

### Attitude Block $\mathbf{f}_\eta$

$$\mathbf{f}_\eta(\mathbf{x}, \mathbf{u}) = \begin{bmatrix}
\frac{1}{\tau_\phi}(k_\phi \phi_c - \phi) \\
\frac{1}{\tau_\theta}(k_\theta \theta_c - \theta) \\
\dot{\psi}_c
\end{bmatrix}$$

## 10.4 Stochastic Extension

For the CC-MPC formulation, we extend the deterministic model with **additive Gaussian process noise**:

$$\boxed{\mathbf{x}_{k+1} = \mathbf{f}_d(\mathbf{x}_k, \mathbf{u}_k) + \boldsymbol{\omega}_k, \quad \boldsymbol{\omega}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)}$$

with initial state uncertainty:

$$\mathbf{x}_0 \sim \mathcal{N}(\hat{\mathbf{x}}_0, \boldsymbol{\Gamma}_0)$$

The zero-mean process noise is omitted from nominal mean propagation, so $\hat{\mathbf{x}}_{k+1}=\mathbf{f}_d(\hat{\mathbf{x}}_k,\mathbf{u}_k)$ is used as an approximation. It still affects covariance propagation and therefore the chance-constraint margin.

## 10.5 Linear Time-Varying (LTV) Approximation

For optimization, we linearize around a nominal trajectory:

$$\mathbf{x}_{k+1} \approx \mathbf{A}_k\mathbf{x}_k + \mathbf{B}_k\mathbf{u}_k + \mathbf{C}_k$$

This is a **Linear Time-Varying (LTV)** model because $\mathbf{A}_k, \mathbf{B}_k$ change at each step (they depend on the linearization point).

### At Hover

At the hover equilibrium ($\mathbf{v} = 0, \phi = \theta = 0, \mathbf{u} = 0$):

$$\mathbf{A}_{\text{hover}} \approx \begin{bmatrix}
\mathbf{I}_3 & \mathbf{I}_3\Delta t & \mathbf{0} \\
\mathbf{0} & \operatorname{diag}(1-k_D\Delta t,1-k_D\Delta t,1-\Delta t/\tau_{vz}) & \begin{bmatrix} 0 & g\Delta t & 0 \\ -g\Delta t & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} \\
\mathbf{0} & \mathbf{0} & \text{diag}(1-\Delta t/\tau_\phi, 1-\Delta t/\tau_\theta, 1)
\end{bmatrix}$$

For the optional second-order frozen-Jacobian input approximation, the important non-zero entries of $\mathbf{B}_{\text{hover}}$ are:

$$\begin{aligned}
B[v_x,\theta_c] &= \frac{g\Delta t^2}{2}\frac{k_\theta}{\tau_\theta}, &
B[v_y,\phi_c] &= -\frac{g\Delta t^2}{2}\frac{k_\phi}{\tau_\phi},\\
B[v_z,v_{zc}] &= \Delta t\frac{k_{vz}}{\tau_{vz}}, &
B[z,v_{zc}] &= \frac{\Delta t^2}{2}\frac{k_{vz}}{\tau_{vz}},\\
B[\phi,\phi_c] &= \Delta t\frac{k_\phi}{\tau_\phi}, &
B[\theta,\theta_c] &= \Delta t\frac{k_\theta}{\tau_\theta},\\
B[\psi,\dot\psi_c] &= \Delta t.
\end{aligned}$$

Thus pitch command affects $v_x$, while positive roll command affects $v_y$ with a negative sign under the FLU convention. Differentiating the actual RK4 map is preferable when an exact discrete Jacobian is needed.

## 10.6 Observability

The quadrotor model with position and attitude measurements is **locally observable**. The state estimator (VIO or UKF) provides estimates $\hat{\mathbf{x}}$ of the full state.

For the CC-MPC, we use the full state estimate (not just position), because the dynamics model needs velocity and attitude for prediction.

## 10.7 Controllability

The hover-linearized reduced model is controllable under non-zero gains and finite time constants. This should be checked from the rank of its discrete controllability matrix. The input chains are:

1. Horizontal position is controlled **indirectly** through attitude ($\phi_c, \theta_c \to$ tilt $\to$ acceleration)
2. Vertical position is controlled through $v_{zc}$
3. Yaw is controlled directly through $\dot{\psi}_c$
4. The attitude dynamics are stable (first-order response to commands)

For the nine-state reduced hover model, these chains can yield controllability rank 9. The physical system having four simultaneous input channels does not limit the multi-step controllability rank to four or six.

## 10.8 Measurement Model

For analysis, a linearized measurement model may be written as:

$$\mathbf{y}_k = \mathbf{H}\mathbf{x}_k + \boldsymbol{\nu}_k$$

where $\mathbf{H} = [\mathbf{I}_{3\times 3}, \mathbf{0}_{3\times 6}]$ represents an idealized position-only sensor. Motion capture normally measures pose rather than all nine states directly; velocity and possibly angular quantities are produced by an estimator, so $\mathbf{H}=\mathbf{I}_9$ should be used only for a genuinely full-state synthetic measurement.

In practice, VIO provides a full-state estimate:

$$\mathbf{x}_k\mid\mathcal{Y}_{0:k} \approx \mathcal{N}(\hat{\mathbf{x}}_k, \boldsymbol{\Gamma}_k^{\text{VIO}})$$

## 10.9 Prerequisites and Related Chapters

> [!info] Prerequisites
- [[02_Quadrotor_Dynamics|Ch.2: Quadrotor Dynamics]] — The dynamics function
- [[08_Linearization|Ch.8: Linearization]] — LTV matrices
- [[09_Discretization|Ch.9: Discretization]] — Discrete-time form
- Linear systems theory

> [!info] Used In
- [[11_MPC|Ch.11: Model Predictive Control]] — Prediction model
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — Stochastic formulation
- [[14_Covariance_Propagation|Ch.14: Covariance Propagation]] — Uncertainty evolution

> [!info] See Also
- [[03_Coordinate_Frames|Ch.3: Coordinate Frames]] — Frame definitions
- [[07_Newton_Euler|Ch.7: Newton-Euler Dynamics]] — Alternative dynamics derivation
