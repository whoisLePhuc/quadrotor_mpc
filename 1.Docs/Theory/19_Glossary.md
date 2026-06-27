---
title: 'Chương 19: Glossary'
chapter: 19
tags:
- reference
- glossary
- notation
phase: reference
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 19
- Glossary
- Ch.19
---

## Mathematical Notation

| Symbol | Meaning | Units / Dimension |
|--------|---------|-------------------|
| $\mathbf{x}$ | State vector $[x, y, z, v_x, v_y, v_z, \phi, \theta, \psi]^T$ | $\mathbb{R}^9$ |
| $\mathbf{p}$ | Position $[x, y, z]^T$ | m |
| $\mathbf{v}$ | Velocity $[v_x, v_y, v_z]^T$ | m/s |
| $\boldsymbol{\eta}$ | Attitude $[\phi, \theta, \psi]^T$ | rad |
| $\mathbf{u}$ | Control input $[\phi_c, \theta_c, v_{zc}, \dot{\psi}_c]^T$ | $\mathbb{R}^4$ |
| $\mathbf{f}(\cdot)$ | Nonlinear dynamics function | $\mathbb{R}^9 \times \mathbb{R}^4 \to \mathbb{R}^9$ |
| $\boldsymbol{\omega}$ | Process noise vector | $\mathbb{R}^9$ |
| $\mathbf{Q}$ | Process noise covariance | $\mathbb{R}^{9\times 9}$ |
| $\mathbf{W}$ | Process noise covariance matrix | $\mathbb{R}^{9\times 9}$ |
| $\hat{\mathbf{x}}$ | State mean (estimate) | $\mathbb{R}^9$ |
| $\boldsymbol{\Gamma}$ | Full state uncertainty covariance | $\mathbb{R}^{9\times 9}$ |
| $\boldsymbol{\Sigma}$ | Position uncertainty covariance | $\mathbb{R}^{3\times 3}$ |
| $\mathbf{F}$ | State transition Jacobian ($\partial\mathbf{f}/\partial\mathbf{x}$) | $\mathbb{R}^{9\times 9}$ |
| $\mathbf{A}_k$ | Discrete-time state matrix | $\mathbb{R}^{9\times 9}$ |
| $\mathbf{B}_k$ | Discrete-time control matrix | $\mathbb{R}^{9\times 4}$ |
| $\mathbf{C}_k$ | Affine dynamics offset | $\mathbb{R}^9$ |
| $N$ | Prediction horizon steps | $\mathbb{N}$ |
| $\Delta t$ | Discretization time step | s |
| $\tau$ | Planning horizon ($N\Delta t$) | s |
| $k$ | Time step index | $\mathbb{N}$ |
| $i, j$ | Robot indices | $\mathbb{N}$ |
| $o$ | Obstacle index | $\mathbb{N}$ |
| $\delta$ | Collision probability threshold | $[0, 0.5]$ |
| $\delta_r$ | Inter-robot collision threshold | $[0, 0.5]$ |
| $\delta_o$ | Robot-obstacle collision threshold | $[0, 0.5]$ |
| $r_i$ | Robot $i$ collision radius | m |
| $a_o, b_o, c_o$ | Obstacle ellipsoid semi-principal axes | m |
| $l_o, w_o, h_o$ | Obstacle box dimensions (length, width, height) | m |
| $\mathbf{R}_o$ | Obstacle orientation (rotation matrix) | $\mathbb{R}^{3\times 3}$ |
| $\boldsymbol{\Omega}_{io}$ | Collision matrix (sphere-ellipsoid) | $\mathbb{R}^{3\times 3}$ |
| $\mathbf{n}$ | Unit normal vector | $\mathbb{R}^3$ |
| $\mathbf{a}$ | Linearized constraint normal vector | $\mathbb{R}^3$ |
| $b$ | Linearized constraint scalar | $\mathbb{R}$ |
| $\text{erf}(x)$ | Error function | $\mathbb{R} \to [-1, 1]$ |
| $\text{erf}^{-1}(x)$ | Inverse error function | $[-1, 1] \to \mathbb{R}$ |
| $\mathcal{C}$ | Collision region | Set |
| $\tilde{\mathcal{C}}$ | Linearized collision region (half-space) | Set |
| $\mathbb{P}[\cdot]$ | Probability measure | $[0, 1]$ |
| $p[\cdot]$ | Probability density function | — |
| $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | Multivariate Gaussian distribution | — |
| $s$ | Slack variable (soft constraint) | $\mathbb{R}_{\geq 0}$ |
| $\rho$ | Slack penalty weight | $\mathbb{R}_{> 0}$ |
| $J^k$ | Stage cost at step $k$ | $\mathbb{R}$ |
| $J^N$ | Terminal cost | $\mathbb{R}$ |
| $\mathbf{Q}_g$ | Terminal cost weight matrix (on position) | $\mathbb{R}^{3\times 3}$ |
| $\mathbf{R}$ | Control cost weight matrix | $\mathbb{R}^{4\times 4}$ |
| $Q_\psi$ | Yaw alignment cost weight | $\mathbb{R}$ |
| $Q_o$ | Logistic cost maximum value | $\mathbb{R}$ |
| $\lambda_o$ | Logistic cost steepness | $\mathbb{R}$ |
| $r_o$ | Logistic cost threshold distance | m |
| $\mathcal{U}$ | Admissible control set | $\mathbb{R}^4$ |
| $\mathcal{X}$ | Admissible state set | $\mathbb{R}^9$ |

## Dynamical Parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| $g$ | 9.81 m/s² | Gravitational acceleration |
| $k_D$ | 0.5 s⁻¹ | Aerodynamic drag coefficient |
| $k_\phi$ | 1.0 | Roll command gain |
| $k_\theta$ | 1.0 | Pitch command gain |
| $k_{vz}$ | 3.0 | Vertical velocity command gain |
| $\tau_\phi$ | 0.2 s | Roll time constant |
| $\tau_\theta$ | 0.2 s | Pitch time constant |
| $\tau_{vz}$ | 0.4 s | Vertical velocity time constant |

## Camera Parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| $f$ | 600 px | Camera focal length |
| $\alpha_h$ | 87° | Horizontal field of view |
| $\alpha_v$ | 58° | Vertical field of view |
| $d_{\text{max}}$ | 5 m | Maximum depth sensing range |

## Acronyms

| Acronym | Full Name |
|---------|-----------|
| CC-MPC | Chance-Constrained Model Predictive Control |
| CCNMPC | Chance-Constrained Nonlinear Model Predictive Control |
| MPC | Model Predictive Control |
| NMPC | Nonlinear Model Predictive Control |
| iMPC | Iterative Model Predictive Control |
| QP | Quadratic Program |
| NLP | Nonlinear Program |
| EKF | Extended Kalman Filter |
| UKF | Unscented Kalman Filter |
| VIO | Visual-Inertial Odometry |
| FOV | Field of View |
| POI | Point of Interest |
| DPP | Disciplined Parametrized Programming |
| CV | Constant Velocity (model) |
| SP | Sequential Planning |
| DC | Distributed with Communication |
| MAV | Micro Aerial Vehicle |
| ROS | Robot Operating System |
| LTV | Linear Time-Varying |
| LHS | Left-Hand Side (of equation) |
| RHS | Right-Hand Side (of equation) |
| RK4 | Runge-Kutta 4th order |
| S-MSCKF | Square-root Multi-State Constraint Kalman Filter |

## Frame Conventions

| Frame | Description | Axes Convention |
|-------|-------------|-----------------|
| World ($W$) | Inertial reference frame | Z-up (NED-like) |
| Body ($B$) | Quadrotor body-fixed frame | X-forward, Y-left, Z-up |
| Camera ($C$) | Stereo camera frame | X-forward (depth), Y-right, Z-down |

## Probability Distributions

| Notation | Meaning |
|----------|---------|
| $\mathbf{x} \sim \mathcal{N}(\hat{\mathbf{x}}, \boldsymbol{\Gamma})$ | State is Gaussian with mean $\hat{\mathbf{x}}$ and covariance $\boldsymbol{\Gamma}$ |
| $\boldsymbol{\omega}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$ | Process noise is zero-mean Gaussian |
| $\mathbf{x}_0 \sim \mathcal{N}(\hat{\mathbf{x}}_0, \boldsymbol{\Gamma}_0)$ | Initial state uncertainty |
| $\mathbf{p}_i \sim \mathcal{N}(\hat{\mathbf{p}}_i, \boldsymbol{\Sigma}_i)$ | Position is Gaussian |
