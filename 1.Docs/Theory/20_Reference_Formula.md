---
title: 'Chương 20: Reference Formulas'
chapter: 20
tags:
- reference
- formulas
- cheatsheet
phase: reference
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 20
- Reference Formulas
- Ch.20
---

## Complete Formula Index

All formulas are numbered by order of appearance in the source papers.

---

### Vision & Detection (Lin, Zhu & Alonso-Mora, 2020)

| Eq | Formula | Description | Verified |
|----|---------|-------------|----------|
| (1) | $T_{\text{POI}} = f \cdot T_{h_o} / d_{\text{bin}}$ | U-depth POI threshold | ✅ |
| (2) | $x_o^B = d_b, \; y_o^B = \frac{(u_l+u_r)d_b}{2f}, \; l_o^B = 2(d_b-d_t), \; w_o^B = \frac{(u_r-u_l)d_b}{f}$ | Horizontal position/size | ✅ |
| (3) | $z_o^B = \frac{(h_t+h_b)d_b}{2f}, \; h_o^B = \frac{|h_t-h_b|d_b}{f}$ | Vertical position/height | ✅ |
| (4) | $\mathbf{p}_o^W = \mathbf{R}_B^W\mathbf{p}_o^B + \mathbf{p}^W, \; \boldsymbol{\Sigma}_o^W = \mathbf{R}_B^{W\;T}\boldsymbol{\Sigma}_o^B\mathbf{R}_B^W + \boldsymbol{\Sigma}^W$ | World frame transform | ✅ |
| (5) | $p_d = p_G(\mathbf{x}_o^m \mid \hat{\mathbf{x}}_o^{m|m-1}, \mathbf{P}_o^{m|m-1})$ | Gaussian data association | ✅ |
| (6) | $\hat{\mathbf{p}}_o^{k+1} = \hat{\mathbf{p}}_o^{k} + \hat{\mathbf{v}}_o^{k}\Delta t, \; \hat{\mathbf{v}}_o^{k+1} = \hat{\mathbf{v}}_o^{k}, \; \boldsymbol{\Sigma}_o^{k+1} = \boldsymbol{\Sigma}_o^{k} + \boldsymbol{\Sigma}_{o,v}\Delta t^2$ | Constant velocity prediction | ✅ |
| (7) | $(a_o, b_o, c_o) = \frac{\sqrt{3}}{2}(l_o, w_o, h_o)$ | Box-to-ellipsoid bounding | ✅ |

### Dynamics Model

| Eq | Formula | Description |
|----|---------|-------------|
| (8) | $\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) + \boldsymbol{\omega}_k, \; \mathbf{x}_0 \sim \mathcal{N}(\hat{\mathbf{x}}_0, \boldsymbol{\Gamma}_0)$ | Stochastic discrete dynamics |
| — | $\dot{v}_x = g(\tan\theta\cos\psi + \tan\phi\sin\psi) - k_D v_x$ | Forward acceleration |
| — | $\dot{v}_y = g(\tan\theta\sin\psi - \tan\phi\cos\psi) - k_D v_y$ | Lateral acceleration |
| — | $\dot{v}_z = \frac{1}{\tau_{vz}}(k_{vz}v_{zc} - v_z)$ | Vertical acceleration |
| — | $\dot{\phi} = \frac{1}{\tau_\phi}(k_\phi\phi_c - \phi)$ | Roll dynamics |
| — | $\dot{\theta} = \frac{1}{\tau_\theta}(k_\theta\theta_c - \theta)$ | Pitch dynamics |
| — | $\dot{\psi} = \dot{\psi}_c$ | Yaw dynamics |

### MPC Problem Formulation

| Eq | Formula | Description |
|----|---------|-------------|
| (9a) | $\min \sum_{k=0}^{N-1} J^k(\hat{\mathbf{x}}^k, \mathbf{u}^k) + J^N(\hat{\mathbf{x}}^N)$ | MPC objective |
| (9b) | $\hat{\mathbf{x}}^0 = \hat{\mathbf{x}}(t)$ | Initial condition |
| (9c) | $\hat{\mathbf{x}}^{k} = \mathbf{f}(\hat{\mathbf{x}}^{k-1}, \mathbf{u}^{k-1})$ | Dynamics |
| (9d) | $\mathbf{G}(\hat{\mathbf{x}}^k, \boldsymbol{\Gamma}^k) \leq 0$ | Constraints |
| (9e) | $\mathbf{u}^{k-1} \in \mathcal{U}, \; \hat{\mathbf{x}}^k \in \mathcal{X}$ | Bounds |

### Cost Terms

| Eq | Formula | Description |
|----|---------|-------------|
| (10) | $J^N(\hat{\mathbf{x}}^N) = \|\hat{\mathbf{p}}^N - \mathbf{p}_g\|_{\mathbf{Q}_g}$ | Terminal cost |
| (11) | $J_u^k(\mathbf{u}^k) = \|\mathbf{u}^k\|_{\mathbf{Q}_u}$ | Control cost |
| (12) | $J_{c,o}^k = \frac{Q_o}{1 + \exp(\lambda_o(d_o^k - r_o))}$ | Logistic collision cost |
| (13) | $J_\psi^k = Q_\psi(\psi^k - \bar{\psi}^k)^2$ | Yaw cost |
| (14) | $J^k = J_u^k + J_c^k + J_\psi^k$ | Total stage cost |

### Chance Constraints

| Eq | Formula | Description |
|----|---------|-------------|
| (15) | $\mathbb{P}(\mathcal{C}_o^k) \leq \delta$ | Chance constraint |
| (16) | $\mathbf{n}_o^T\boldsymbol{\Omega}^{1/2}(\hat{\mathbf{p}} - \hat{\mathbf{p}}_o) - 1 \geq \text{erf}^{-1}(1-2\delta)\sqrt{2\mathbf{n}_o^T\boldsymbol{\Omega}^{1/2}(\boldsymbol{\Sigma} + \boldsymbol{\Sigma}_o)\boldsymbol{\Omega}^{1/2\;T}\mathbf{n}_o}$ | Deterministic reformulation |
| L1 | $\mathbb{P}(\mathbf{a}^T\mathbf{x} \leq b) = \frac{1}{2} + \frac{1}{2}\text{erf}\left(\frac{b - \mathbf{a}^T\hat{\mathbf{x}}}{\sqrt{2\mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a}}}\right)$ | Gaussian probability |
| L2 | $\mathbf{a}^T\hat{\mathbf{x}} - b \geq \text{erf}^{-1}(1-2\delta)\sqrt{2\mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a}}$ | Deterministic reform |

### FOV Constraints

| Eq | Formula | Description |
|----|---------|-------------|
| (17) | $\text{FOV}^k = \{\mathbf{p} \mid \mathbf{n}_j^k \cdot \mathbf{p} \leq m_j^k, \; j = 1,\ldots,5\}$ | FOV half-spaces |
| (18) | $\mathbf{p}^k \in \text{FOV}^k, \; \forall k = 1,\ldots,N$ | FOV constraint |

### Uncertainty Propagation

| Eq | Formula | Description |
|----|---------|-------------|
| (19) | $\boldsymbol{\Gamma}^{k+1} = \mathbf{F}^k\boldsymbol{\Gamma}^k\mathbf{F}^{k\;T} + \mathbf{W}^k$ | EKF covariance propagation |

### Collision Geometry

| Formula | Description |
|---------|-------------|
| $\mathcal{C}_{ij} = \{\mathbf{x}_i \mid \|\mathbf{p}_i - \mathbf{p}_j\| \leq r_i + r_j\}$ | Sphere-sphere collision |
| $\mathcal{C}_{io} = \{\mathbf{x}_i \mid \|\mathbf{p}_i - \mathbf{p}_o\|_{\boldsymbol{\Omega}_{io}} \leq 1\}$ | Sphere-ellipsoid collision |
| $\boldsymbol{\Omega}_{io} = \mathbf{R}_o^T \text{diag}\left(\frac{1}{(a_o+r_i)^2}, \frac{1}{(b_o+r_i)^2}, \frac{1}{(c_o+r_i)^2}\right)\mathbf{R}_o$ | Collision matrix |

### Multi-Robot Prediction (Zhu & Alonso-Mora, 2019)

| Formula | Description |
|---------|-------------|
| $[\hat{\mathbf{p}}_j^k, \hat{\mathbf{v}}_j^k]^T = \mathbf{F}_j^k [\hat{\mathbf{p}}_j^{k-1}, \hat{\mathbf{v}}_j^{k-1}]^T$ | CV prediction |
| $\boldsymbol{\Sigma}_{j,\text{pv}}^k = \mathbf{F}_j^k \boldsymbol{\Sigma}_{j,\text{pv}}^{k-1} \mathbf{F}_j^{k\;T} + \mathbf{Q}_{j,\text{pv}}^k$ | CV covariance |
| $\mathbf{F}_j = \begin{bmatrix} \mathbf{I}_3 & \Delta t\mathbf{I}_3 \\ \mathbf{0} & \mathbf{I}_3 \end{bmatrix}$ | CV state transition |

### Key Values

| Parameter | Value | Source |
|-----------|-------|--------|
| $\delta$ (collision threshold) | 0.03 | Both papers |
| $\Delta t$ (MPC step) | 0.06 s (vision), 0.05 s (mocap) | Config |
| $N$ (horizon steps) | 30 (1.8 s) | Config |
| $r_i$ (MAV radius) | 0.4 m (vision), 0.3 m (mocap) | Config |
| $\text{erf}^{-1}(1-2\delta)$ for $\delta=0.03$ | 1.3293 | Mathematical |
| $\tau_\phi, \tau_\theta$ | 0.2 s | Model |
| $\tau_{vz}$ | 0.4 s | Model |
| $k_D$ | 0.5 | Model |
| $k_{vz}$ | 3.0 | Model |
| Max speed | 8.0 m/s (config) | Config |
| Max roll/pitch command | 0.35 rad (20°) | Config |

---

## Implementation Checkpoints

- [ ] Dynamics implemented (RK4 integration)
- [ ] Linearization verified (finite-difference Jacobian vs analytical)
- [ ] Second-order B-matrix correction included
- [ ] Affine offset C_k computed correctly ($\mathbf{x}_{\text{next}} - \mathbf{A}\bar{\mathbf{x}} - \mathbf{B}\bar{\mathbf{u}}$)
- [ ] Uncertainty propagation: $\boldsymbol{\Gamma}_{k+1} = \mathbf{F}\boldsymbol{\Gamma}_k\mathbf{F}^T + \mathbf{W}\Delta t$
- [ ] Chance constraint RHS: $\text{erf}^{-1}(1-2\delta)\sqrt{2\mathbf{n}^T\mathbf{L}(\boldsymbol{\Sigma}+\boldsymbol{\Sigma}_o)\mathbf{L}^T\mathbf{n}}$
- [ ] $\boldsymbol{\Omega}^{1/2}$ computed via Cholesky (lower triangular)
- [ ] Logistic cost gradient clipped to 5.0
- [ ] Slack variables with penalty 1000.0
- [ ] Warm-start from shifted previous trajectory
- [ ] Fallback PID controller for infeasible QP
- [ ] All 13 verification tests pass
