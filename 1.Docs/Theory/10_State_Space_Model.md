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

## 10.3 State Transition Function

The function $\mathbf{f}: \mathbb{R}^9 \times \mathbb{R}^4 \to \mathbb{R}^9$ consists of three blocks:

$$\mathbf{f}(\mathbf{x}, \mathbf{u}) = \begin{bmatrix} \mathbf{f}_p(\mathbf{x}) \\ \mathbf{f}_v(\mathbf{x}, \mathbf{u}) \\ \mathbf{f}_\eta(\mathbf{x}, \mathbf{u}) \end{bmatrix}$$

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

$$\boxed{\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) + \boldsymbol{\omega}_k, \quad \boldsymbol{\omega}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)}$$

with initial state uncertainty:

$$\mathbf{x}_0 \sim \mathcal{N}(\hat{\mathbf{x}}_0, \boldsymbol{\Gamma}_0)$$

The process noise **does not affect the mean propagation** in the MPC (we use $\hat{\mathbf{x}}_{k+1} = \mathbf{f}(\hat{\mathbf{x}}_k, \mathbf{u}_k)$ for the nominal trajectory), but it **does affect the covariance propagation** (Eq. 19) and thus the chance constraint tightness.

## 10.5 Linear Time-Varying (LTV) Approximation

For optimization, we linearize around a nominal trajectory:

$$\mathbf{x}_{k+1} \approx \mathbf{A}_k\mathbf{x}_k + \mathbf{B}_k\mathbf{u}_k + \mathbf{C}_k$$

This is a **Linear Time-Varying (LTV)** model because $\mathbf{A}_k, \mathbf{B}_k$ change at each step (they depend on the linearization point).

### At Hover

At the hover equilibrium ($\mathbf{v} = 0, \phi = \theta = 0, \mathbf{u} = 0$):

$$\mathbf{A}_{\text{hover}} = \begin{bmatrix}
\mathbf{I}_3 & \mathbf{I}_3\Delta t & \mathbf{0} \\
\mathbf{0} & (1-k_D\Delta t)\mathbf{I}_3 & \begin{bmatrix} 0 & g\Delta t & 0 \\ -g\Delta t & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} \\
\mathbf{0} & \mathbf{0} & \text{diag}(1-\Delta t/\tau_\phi, 1-\Delta t/\tau_\theta, 1)
\end{bmatrix}$$

$$\mathbf{B}_{\text{hover}} = \begin{bmatrix}
\mathbf{0} & \mathbf{0} & 0 & 0 \\
\mathbf{0} & \mathbf{0} & 0 & 0 \\
\mathbf{0} & \mathbf{0} & 0 & 0 \\
\begin{matrix} \frac{g\Delta t^2 k_\phi}{2\tau_\phi} \\ 0 \\ 0 \end{matrix} & \begin{matrix} 0 \\ \frac{g\Delta t^2 k_\theta}{2\tau_\theta} \\ 0 \end{matrix} & \begin{matrix} 0 \\ 0 \\ \frac{k_{vz}\Delta t}{\tau_{vz}} \end{matrix} & \begin{matrix} 0 \\ 0 \\ 0 \end{matrix} \\
\frac{k_\phi\Delta t}{\tau_\phi} & 0 & 0 & 0 \\
0 & \frac{k_\theta\Delta t}{\tau_\theta} & 0 & 0 \\
0 & 0 & 0 & \Delta t
\end{bmatrix}$$

(Note: $\mathbf{B}_{\text{hover}}$ uses the second-order correction; first-order would have zeros in row 3–4 columns 0–1.)

## 10.6 Observability

The quadrotor model with position and attitude measurements is **locally observable**. The state estimator (VIO or UKF) provides estimates $\hat{\mathbf{x}}$ of the full state.

For the CC-MPC, we use the full state estimate (not just position), because the dynamics model needs velocity and attitude for prediction.

## 10.7 Controllability

The quadrotor is **controllable** — any state can be reached from any initial state with appropriate control inputs, despite the under-actuation (4 inputs for 6 DOF). This is because:

1. Horizontal position is controlled **indirectly** through attitude ($\phi_c, \theta_c \to$ tilt $\to$ acceleration)
2. Vertical position is controlled through $v_{zc}$
3. Yaw is controlled directly through $\dot{\psi}_c$
4. The attitude dynamics are stable (first-order response to commands)

The controllable subspace has dimension 6 (position + velocity + yaw), matching the number of independent DOF we can control.

## 10.8 Measurement Model

The simplest measurement model is:

$$\mathbf{y}_k = \mathbf{H}\mathbf{x}_k + \boldsymbol{\nu}_k$$

where $\mathbf{H} = [\mathbf{I}_{3\times 3}, \mathbf{0}_{3\times 6}]$ for position-only measurements, or $\mathbf{H} = \mathbf{I}_9$ for full-state measurement (motion capture).

In practice, VIO provides a full-state estimate:

$$\hat{\mathbf{x}}_k \sim \mathcal{N}(\mathbf{x}_k^{\text{true}}, \boldsymbol{\Gamma}_k^{\text{VIO}})$$

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
