---
title: 'Chương 15: Obstacle Avoidance'
chapter: 15
tags:
- ccmpc
- collision-avoidance
- vision
- detection
phase: implementation
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 15
- Obstacle Avoidance
- Ch.15
---

## 15.1 Introduction

Obstacle avoidance for MAVs involves three tasks:
1. **Detection**: Perceiving obstacles from sensor data
2. **Prediction**: Estimating future obstacle positions and uncertainties
3. **Avoidance**: Planning trajectories that maintain safe separation

The CC-MPC framework addresses all three, with the vision pipeline handling detection and the MPC handling prediction and avoidance.

## 15.2 Obstacle Representation

### Detection: From Depth Images to Boxes

The vision pipeline (Lin, Zhu & Alonso-Mora, 2020, Sec. III) processes stereo depth images:

```
Depth Image → U-Depth Map → POI Detection → Bounding Box → Ellipsoid
```

#### U-Depth Map (Eq. 1)

A U-depth map is a histogram of depth values for each image column. A bin $(u, d)$ is a **point of interest (POI)** if:

$$T_{\text{POI}} = \frac{f \cdot T_{h_o}}{d_{\text{bin}}}$$

where:
- $f$: Camera focal length (pixels)
- $T_{h_o}$: Predefined obstacle height threshold (m)
- $d_{\text{bin}}$: Depth of the bin (m)

**Physical meaning**: At closer depth, an object of fixed height occupies more image rows, so the bin count is larger.

#### Box Detection (Eqs. 2–3)

From neighboring POIs, a bounding box is extracted:

**Horizontal (from U-depth map)**:

$$\begin{aligned}
x_o^B &= d_b \\
y_o^B &= \frac{(u_l + u_r)d_b}{2f} \\
l_o^B &= 2(d_b - d_t) \\
w_o^B &= \frac{(u_r - u_l)d_b}{f}
\end{aligned}$$

where $(u_l, d_t)$ is top-left corner, $(u_r, d_b)$ is bottom-right corner in the U-depth map.

**Vertical (from depth image)**:

$$\begin{aligned}
z_o^B &= \frac{(h_t + h_b)d_b}{2f} \\
h_o^B &= \frac{|h_t - h_b| \cdot d_b}{f}
\end{aligned}$$

where $(h_t, h_b)$ are top and bottom row indices in the depth image.

#### World Frame Transformation (Eq. 4)

The detection in body (camera) frame is transformed to world frame:

$$\begin{aligned}
\mathbf{p}_o^W &= \mathbf{R}_B^W \mathbf{p}_o^B + \mathbf{p}^W \\
\boldsymbol{\Sigma}_o^W &= \mathbf{R}_B^{W\;T} \boldsymbol{\Sigma}_o^B \mathbf{R}_B^W + \boldsymbol{\Sigma}^W
\end{aligned}$$

where $\mathbf{R}_B^W$ is the MAV's rotation matrix (from VIO) and $\boldsymbol{\Sigma}^W$ is the MAV position uncertainty.

For the size, a compensation matrix accounts for MAV pitch/roll:

$$\mathbf{R}_{B,s}^W = \text{diag}(\cos\theta, \cos\phi, \frac{1}{\cos\theta\cos\phi})$$

### Ellipsoidal Model (Eq. 7)

For optimization (smooth constraints), the detected box is converted to a bounding ellipsoid:

$$\boxed{(a_o, b_o, c_o) = \frac{\sqrt{3}}{2}(l_o, w_o, h_o)}$$

**Justification**: This ensures the box corners lie on the ellipsoid surface:

For a corner at $(l/2, w/2, h/2)$:

$$\frac{(l/2)^2}{a^2} + \frac{(w/2)^2}{b^2} + \frac{(h/2)^2}{c^2} = \frac{(l/2)^2}{(\sqrt{3}l/2)^2} + \cdots = \frac{1}{3} + \frac{1}{3} + \frac{1}{3} = 1$$

The factor $\sqrt{3}/2$ makes the ellipsoid the smallest one that circumscribes the box.

### Obstacle Model Parameters

| Parameter | Symbol | Source |
|-----------|--------|--------|
| Position | $\hat{\mathbf{p}}_o$ | Detection + tracking |
| Velocity | $\hat{\mathbf{v}}_o$ | Kalman filter tracking |
| Size (axes) | $(a_o, b_o, c_o)$ | Detection (Eq. 7) |
| Orientation | $\mathbf{R}_o$ | Detection yaw angle |
| Position uncertainty | $\boldsymbol{\Sigma}_o$ | Detection covariance |
| Velocity uncertainty | $\boldsymbol{\Sigma}_{o,v}$ | Tracking covariance |

## 15.3 Obstacle Tracking

### Data Association (Eq. 5)

To track obstacles across frames, detections are associated with existing tracks using Gaussian probability density:

$$p_d = p_G(\mathbf{x}_o^m \mid \hat{\mathbf{x}}_o^{m|m-1}, \mathbf{P}_o^{m|m-1})$$

where:
- $\mathbf{x}_o = (\mathbf{p}_o^W, \mathbf{s}_o^B)^T$: Obstacle state (position + size)
- $\hat{\mathbf{x}}_o^{m|m-1}$: Predicted state from previous frame
- $\mathbf{P}_o^{m|m-1}$: Predicted covariance

If $p_d$ exceeds a threshold, the detection is associated and fed to a Kalman filter update.

### Constant Velocity Prediction (Eq. 6)

For collision avoidance, future obstacle positions are predicted:

$$\begin{aligned}
\hat{\mathbf{p}}_o^{k+1} &= \hat{\mathbf{p}}_o^k + \hat{\mathbf{v}}_o^k \Delta t \\
\hat{\mathbf{v}}_o^{k+1} &= \hat{\mathbf{v}}_o^k \\
\boldsymbol{\Sigma}_o^{k+1} &= \boldsymbol{\Sigma}_o^k + \boldsymbol{\Sigma}_{o,v} \Delta t^2
\end{aligned}$$

Important: The obstacle size is assumed constant ($\hat{\mathbf{s}}_o^{k+1} = \hat{\mathbf{s}}_o^k$) and its uncertainty is not considered in collision avoidance.

## 15.4 Collision Conditions

### Inter-Robot Collision

Two spherical robots $i$ and $j$ collide if:

$$\|\mathbf{p}_i - \mathbf{p}_j\| \leq r_i + r_j$$

### Robot-Obstacle Collision

A spherical robot $i$ (radius $r_i$) and ellipsoidal obstacle $o$ collide if:

$$\|\mathbf{p}_i - \mathbf{p}_o\|_{\boldsymbol{\Omega}_{io}} \leq 1$$

where $\boldsymbol{\Omega}_{io}$ is the **collision matrix**:

$$\boxed{\boldsymbol{\Omega}_{io} = \mathbf{R}_o^T \text{diag}\left(\frac{1}{(a_o+r_i)^2}, \frac{1}{(b_o+r_i)^2}, \frac{1}{(c_o+r_i)^2}\right)\mathbf{R}_o}$$

> [!note] 
> The robot radius $r_i$ is incorporated by **enlarging** the obstacle ellipsoid. This is an approximation that avoids computing the true sphere-ellipsoid distance (which has no closed form).

## 15.5 Collision Chance Constraints

Since positions are random variables (Gaussian), collision avoidance must be probabilistic:

$$\mathbb{P}(\mathbf{x}_i^k \in \mathcal{C}_{io}^k) \leq \delta_o$$

With $\delta_o = 0.03$: "At most 3% chance of collision with this obstacle at this step."

### Deterministic Reformulation

Following the linearization procedure ([[13_Chance_Constraints|Ch.13]]), the chance constraint becomes:

$$\mathbf{n}_o^T\boldsymbol{\Omega}_{io}^{1/2}(\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_o) - 1 \geq \text{erf}^{-1}(1-2\delta_o)\sqrt{2\mathbf{n}_o^T\boldsymbol{\Omega}_{io}^{1/2}(\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_o)\boldsymbol{\Omega}_{io}^{1/2\;T}\mathbf{n}_o}$$

where $\mathbf{n}_o = \frac{\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_o}{\|\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_o\|}$.

### Interpretation

The left-hand side (LHS) is the **transformed nominal distance**:

$$\text{LHS} = \mathbf{n}_o^T\boldsymbol{\Omega}_{io}^{1/2}(\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_o) - 1$$

- LHS > 0: Nominal positions are collision-free
- LHS = 0: On the boundary
- LHS < 0: Nominally in collision

The right-hand side (RHS) is the **uncertainty margin**:

$$\text{RHS} = \text{erf}^{-1}(1-2\delta_o)\sqrt{2\mathbf{n}_o^T\boldsymbol{\Omega}_{io}^{1/2}(\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_o)\boldsymbol{\Omega}_{io}^{1/2\;T}\mathbf{n}_o}$$

- Large $\boldsymbol{\Sigma}$: Large RHS → need more nominal clearance
- Small $\delta_o$: Large $\text{erf}^{-1}(1-2\delta_o)$ → more conservative

The constraint requires: $\text{LHS} \geq \text{RHS}$ — the nominal clearance must exceed the uncertainty margin.

## 15.6 Comparison of Collision Probability Methods

From Zhu & Alonso-Mora (2019), Table I:

| Method | Computed $P$(collision) | Time (ms) | Result |
|--------|------------------------|-----------|--------|
| **Numerical integration** (truth) | 0.011 | 258.7 | Feasible |
| Bounding volume (3σ) | 1.000 | 0.011 | **Infeasible** |
| Center point PDF [12] | 3.6×10⁻¹⁸ | 0.016 | Feasible (**unsafe**) |
| Cube approximation [11] | 0.100 | 0.044 | **Infeasible** |
| **CC-MPC (our method)** | **0.017** | **0.011** | **Feasible** |

- **Bounding volume**: Massively overestimates collision probability → infeasible
- **Center point PDF**: Severely underestimates → unsafe
- **Cube approximation**: Overestimates → infeasible
- **CC-MPC**: Tight bound, computationally efficient → optimal

## 15.7 Obstacle Sensing Uncertainty

The depth camera measurement model has quadratic range error:

$$\sigma_{\text{depth}} \propto d^2$$

In practice, an empirically determined detection covariance is used:

$$\boldsymbol{\Sigma}_o^B = \text{diag}(\sigma_{x}^2, \sigma_{y}^2, \sigma_{z}^2)$$

with $\sigma \approx 0.05$ m at typical ranges (2–3 m).

When velocity estimation is noisy (large $\boldsymbol{\Sigma}_{o,v}$), the predicted covariance grows rapidly. In this case, $\boldsymbol{\Sigma}_{o,v}$ is bounded to prevent unrealistic covariance growth.

## 15.8 Field of View (FOV) Constraints

For vision-based avoidance, the planned trajectory must keep the camera pointing at obstacles. The FOV constraint ensures:

1. Obstacles remain within camera field of view
2. Camera faces direction of motion (via yaw cost)

**Half-space representation** (in body frame):

| Constraint | Body-frame inequality |
|-----------|----------------------|
| Left | $-\tan(\alpha_h/2) \cdot x - y \leq 0$ |
| Right | $-\tan(\alpha_h/2) \cdot x + y \leq 0$ |
| Bottom | $-\tan(\alpha_v/2) \cdot x - z \leq 0$ |
| Top | $-\tan(\alpha_v/2) \cdot x + z \leq 0$ |
| Max depth | $x \leq d_{\text{max}}$ |

where $\alpha_h, \alpha_v$ are the FOV angles (e.g., 87° × 58° for RealSense D435i).

**Implementation simplification**: Only the worst-violated constraint is enforced at each step to keep the QP size manageable.

## 15.9 Experimental Validation

From Lin, Zhu & Alonso-Mora (2020):

**Scenario 1** (Lab space, 2 walking humans):
- Minimum safe separation: 0.4 m achieved in all trials
- Maximum MAV speed: 1.6 m/s
- Obstacle detection + tracking: < 8 ms (75th percentile)
- MPC solve time: < 22 ms (75th percentile)

**Scenario 2** (Long corridor, static + moving obstacles):
- Maximum MAV speed: 2.4 m/s
- Successful navigation in narrow space

## 15.10 Prerequisites and Related Chapters

> [!info] Prerequisites
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — The MPC formulation
- [[13_Chance_Constraints|Ch.13: Chance Constraints]] — Linearization of collision conditions
- [[14_Covariance_Propagation|Ch.14: Covariance Propagation]] — Uncertainty evolution

> [!info] Used In
- [[16_Optimization|Ch.16: Optimization Formulation]] — Constraints in QP
- [[18_Implementation_Notes|Ch.18: Implementation Notes]] — ObstacleManager class
