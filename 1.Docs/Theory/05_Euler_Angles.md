---
title: 'Chương 5: Euler Angles'
chapter: 5
tags:
- quadrotor
- kinematics
- euler-angles
- attitude
phase: foundations
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 5
- Euler Angles
- Ch.5
---

## 5.1 Introduction

**Euler angles** $(\phi, \theta, \psi)$ are a minimal (3-parameter) representation of 3D orientation. They are the attitude parameters used in the quadrotor state vector.

The convention used is **ZYX (yaw-pitch-roll)**, also called Tait-Bryan angles:
1. **Yaw** $\psi$: Rotation about world Z-axis
2. **Pitch** $\theta$: Rotation about intermediate Y-axis
3. **Roll** $\phi$: Rotation about body X-axis

## 5.2 Definitions

### Roll $\phi$

Rotation about the body's forward (X) axis. Positive = right side down.

- Range: $[-\pi, \pi]$ or typically $[-0.5, 0.5]$ rad (≈ ±29°) in practice
- Controls lateral (side-to-side) motion
- Commanded via $\phi_c$

### Pitch $\theta$

Rotation about the body's lateral (Y) axis. Positive = nose up.

- Range: $[-\pi/2, \pi/2]$ (for non-singular representation)
- Typically $[-0.5, 0.5]$ rad in practice
- Controls forward/backward motion
- Commanded via $\theta_c$

### Yaw $\psi$

Rotation about the body's vertical (Z) axis. Positive = nose right.

- Range: $[-\pi, \pi]$ (full circle)
- Controls heading direction
- Commanded via $\dot{\psi}_c$ (rate control)

## 5.3 Rotation Matrix from Euler Angles

$$\mathbf{R}_B^W = \mathbf{R}_Z(\psi)\mathbf{R}_Y(\theta)\mathbf{R}_X(\phi)$$

Explicitly:

$$\mathbf{R}_B^W = \begin{bmatrix}
c_\psi c_\theta & c_\psi s_\theta s_\phi - s_\psi c_\phi & c_\psi s_\theta c_\phi + s_\psi s_\phi \\
s_\psi c_\theta & s_\psi s_\theta s_\phi + c_\psi c_\phi & s_\psi s_\theta c_\phi - c_\psi s_\phi \\
-s_\theta & c_\theta s_\phi & c_\theta c_\phi
\end{bmatrix}$$

where $c_\alpha = \cos\alpha$, $s_\alpha = \sin\alpha$.

### Column Interpretation

- Column 1: Body X-axis expressed in world frame (forward direction)
- Column 2: Body Y-axis expressed in world frame (left direction)
- Column 3: Body Z-axis expressed in world frame (up direction, thrust direction)

## 5.4 Euler Angles from Rotation Matrix

Given $\mathbf{R} = [r_{ij}]$:

$$\begin{aligned}
\theta &= -\arcsin(r_{31}) \quad \text{or} \quad \theta = \arctan2\left(-r_{31}, \sqrt{r_{11}^2 + r_{21}^2}\right) \\
\phi &= \arctan2(r_{32}, r_{33}) \\
\psi &= \arctan2(r_{21}, r_{11})
\end{aligned}$$

The $\arcsin$ form has a **gimbal lock singularity** at $\theta = \pm \pi/2$, where roll and yaw become indistinguishable. This is not an issue for quadrotors (they don't fly at 90° pitch).

## 5.5 Small-Angle Linearization

Near hover ($\phi, \theta \approx 0$):

$$\mathbf{R}_B^W \approx \begin{bmatrix}
\cos\psi & -\sin\psi & \theta\cos\psi + \phi\sin\psi \\
\sin\psi & \cos\psi & \theta\sin\psi - \phi\cos\psi \\
-\theta & \phi & 1
\end{bmatrix}$$

The thrust direction in world frame (third column) simplifies to:

$$\mathbf{t}^W \approx \begin{bmatrix} \theta\cos\psi + \phi\sin\psi \\ \theta\sin\psi - \phi\cos\psi \\ 1 \end{bmatrix}$$

At $\psi = 0$: $\mathbf{t}^W \approx [\theta, -\phi, 1]^T$ — pitch tilts forward, roll tilts right.

## 5.6 Velocity Dynamics Through Euler Angles

The key relationship in the quadrotor model:

$$\begin{bmatrix} \dot{v}_x \\ \dot{v}_y \end{bmatrix} = \mathbf{R}_Z(\psi) \begin{bmatrix} g\tan\theta \\ -g\tan\phi \end{bmatrix} - k_D\begin{bmatrix} v_x \\ v_y \end{bmatrix}$$

**Small-angle approximation** (valid for $\phi, \theta < 0.3$ rad ≈ 17°):

$$\begin{bmatrix} \dot{v}_x \\ \dot{v}_y \end{bmatrix} \approx \mathbf{R}_Z(\psi) \begin{bmatrix} g\theta \\ -g\phi \end{bmatrix} - k_D\begin{bmatrix} v_x \\ v_y \end{bmatrix}$$

This is the linearized form used in the Jacobian at hover.

## 5.7 Euler Angle Rates vs. Body Angular Velocity

The Euler angle derivatives are NOT the same as body-frame angular velocities:

$$\begin{bmatrix} \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \end{bmatrix} = \begin{bmatrix}
1 & \sin\phi\tan\theta & \cos\phi\tan\theta \\
0 & \cos\phi & -\sin\phi \\
0 & \sin\phi/\cos\theta & \cos\phi/\cos\theta
\end{bmatrix} \begin{bmatrix} p \\ q \\ r \end{bmatrix}$$

where $(p, q, r)$ are body-frame angular velocities.

For small angles ($\phi, \theta \approx 0$): $\dot{\phi} \approx p$, $\dot{\theta} \approx q$, $\dot{\psi} \approx r$.

The CC-MPC model uses the simplified first-order attitude dynamics directly in Euler angle space, avoiding this transformation.

## 5.8 Attitude Dynamics in CC-MPC

The attitude is modeled as first-order systems:

$$\begin{aligned}
\dot{\phi} &= \frac{1}{\tau_\phi}(k_\phi\phi_c - \phi) \\
\dot{\theta} &= \frac{1}{\tau_\theta}(k_\theta\theta_c - \theta) \\
\dot{\psi} &= \dot{\psi}_c
\end{aligned}$$

This is a simplification that works well for the Bebop 2's onboard controller, which already handles the low-level attitude stabilization.

## 5.9 Advantages and Limitations

### Advantages
- **Minimal**: Only 3 parameters (vs. 4 for quaternion, 9 for rotation matrix)
- **Intuitive**: Direct physical interpretation of each angle
- **Simple dynamics**: First-order decoupled dynamics

### Limitations
- **Gimbal lock**: Singularity at $\theta = \pm \pi/2$ (not an issue in practice)
- **Non-unique**: Multiple Euler angle triples can represent the same orientation
- **Nonlinear**: Trigonometric functions in dynamics and Jacobians

## 5.10 Prerequisites and Related Chapters

> [!info] Prerequisites

> [!info] Used In
- [[02_Quadrotor_Dynamics|Ch.2: Quadrotor Dynamics]] — State vector components
- [[07_Newton_Euler|Ch.7: Newton-Euler Dynamics]] — Orientation parameterization
- [[08_Linearization|Ch.8: Linearization]] — Attitude Jacobian

> [!info] See Also
- [[06_Quaternion|Ch.6: Quaternion]] — Alternative representation
