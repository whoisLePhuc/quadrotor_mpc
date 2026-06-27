---
title: 'Chương 2: Quadrotor Dynamics'
chapter: 2
tags:
- quadrotor
- dynamics
- model
- bebop2
phase: foundations
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 2
- Quadrotor Dynamics
- Ch.2
---

## 2.1 Introduction

A quadrotor is a rotorcraft with four rotors arranged in a cross configuration. Its motion is governed by:
1. **Newton's laws** for translational motion (position $\mathbf{p}$, velocity $\mathbf{v}$)
2. **Euler's equations** for rotational motion (attitude $\phi, \theta, \psi$)

The key property that makes quadrotor control challenging is **under-actuation**: it has 6 degrees of freedom (3 position + 3 attitude) but only 4 control inputs (4 rotor thrusts).

## 2.2 The Simplified Model (Bebop 2)

The CC-MPC papers use a **first-order low-pass Euler approximation** of the Parrot Bebop 2 dynamics. This model captures the essential physics while being simple enough for real-time optimization.

### State Vector

$$\mathbf{x} = \begin{bmatrix} \mathbf{p} \\ \mathbf{v} \\ \boldsymbol{\eta} \end{bmatrix} = \begin{bmatrix} x \\ y \\ z \\ v_x \\ v_y \\ v_z \\ \phi \\ \theta \\ \psi \end{bmatrix} \in \mathbb{R}^9$$

- $\mathbf{p} = [x, y, z]^T$: Position in world frame (m)
- $\mathbf{v} = [v_x, v_y, v_z]^T$: Velocity in world frame (m/s)
- $\boldsymbol{\eta} = [\phi, \theta, \psi]^T$: Roll, pitch, yaw Euler angles (rad)

### Control Input

$$\mathbf{u} = \begin{bmatrix} \phi_c \\ \theta_c \\ v_{zc} \\ \dot{\psi}_c \end{bmatrix} \in \mathbb{R}^4$$

- $\phi_c$: Commanded roll angle (rad) — controls lateral motion
- $\theta_c$: Commanded pitch angle (rad) — controls forward motion
- $v_{zc}$: Commanded vertical velocity (m/s) — controls altitude
- $\dot{\psi}_c$: Commanded yaw rate (rad/s) — controls heading

> [!important] 
> $\phi_c$ and $\theta_c$ are **commanded angles**, not torques. The Bebop 2's low-level controller handles the rotor speed mixing.

## 2.3 Continuous-Time Dynamics

The continuous-time dynamics $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$ are:

### Position Kinematics

$$\dot{\mathbf{p}} = \mathbf{v}$$

Simply: position changes at velocity.

$$\begin{bmatrix} \dot{x} \\ \dot{y} \\ \dot{z} \end{bmatrix} = \begin{bmatrix} v_x \\ v_y \\ v_z \end{bmatrix}$$

### Velocity Dynamics

This is the core of the quadrotor model. The horizontal acceleration comes from tilting the thrust vector:

$$\begin{bmatrix} \dot{v}_x \\ \dot{v}_y \end{bmatrix} = \mathbf{R}_Z(\psi) \begin{bmatrix} g \tan\theta \\ -g \tan\phi \end{bmatrix} - k_D \begin{bmatrix} v_x \\ v_y \end{bmatrix}$$

where:

$$\mathbf{R}_Z(\psi) = \begin{bmatrix} \cos\psi & -\sin\psi \\ \sin\psi & \cos\psi \end{bmatrix}$$

is the rotation matrix about the z-axis (yaw).

**Expanded form**:

$$\begin{aligned}
\dot{v}_x &= g(\tan\theta \cos\psi + \tan\phi \sin\psi) - k_D v_x \\
\dot{v}_y &= g(\tan\theta \sin\psi - \tan\phi \cos\psi) - k_D v_y
\end{aligned}$$

**Physical interpretation**:
- $g \tan\theta$ gives forward acceleration when pitching forward
- $g \tan\phi$ gives lateral acceleration when rolling
- $\mathbf{R}_Z(\psi)$ rotates these accelerations from body frame to world frame
- $-k_D \mathbf{v}$ is aerodynamic drag (proportional to velocity)

**Vertical velocity**:

$$\dot{v}_z = \frac{1}{\tau_{vz}}(k_{vz} v_{zc} - v_z)$$

This is a first-order system:
- $k_{vz}$: DC gain (typically 3.0)
- $\tau_{vz}$: Time constant (typically 0.4 s)
- $v_{zc}$: Commanded vertical velocity

### Attitude Dynamics

The attitude response is also first-order:

$$\begin{aligned}
\dot{\phi} &= \frac{1}{\tau_\phi}(k_\phi \phi_c - \phi) \\
\dot{\theta} &= \frac{1}{\tau_\theta}(k_\theta \theta_c - \theta) \\
\dot{\psi} &= \dot{\psi}_c
\end{aligned}$$

- $k_\phi = k_\theta = 1.0$: Unity gains
- $\tau_\phi = \tau_\theta = 0.2$ s: Time constants
- Yaw rate is directly commanded (integrator)

### Complete State Derivatives

$$\dot{\mathbf{x}} = \begin{bmatrix}
v_x \\
v_y \\
v_z \\
g(\tan\theta \cos\psi + \tan\phi \sin\psi) - k_D v_x \\
g(\tan\theta \sin\psi - \tan\phi \cos\psi) - k_D v_y \\
\frac{1}{\tau_{vz}}(k_{vz} v_{zc} - v_z) \\
\frac{1}{\tau_\phi}(k_\phi \phi_c - \phi) \\
\frac{1}{\tau_\theta}(k_\theta \theta_c - \theta) \\
\dot{\psi}_c
\end{bmatrix}$$

## 2.4 Model Parameters

| Parameter | Symbol | Typical Value | Units |
|-----------|--------|---------------|-------|
| Gravity | $g$ | 9.81 | m/s² |
| Drag coefficient | $k_D$ | 0.5 | 1/s |
| Roll gain | $k_\phi$ | 1.0 | — |
| Pitch gain | $k_\theta$ | 1.0 | — |
| Vertical velocity gain | $k_{vz}$ | 3.0 | — |
| Roll time constant | $\tau_\phi$ | 0.2 | s |
| Pitch time constant | $\tau_\theta$ | 0.2 | s |
| Vertical time constant | $\tau_{vz}$ | 0.4 | s |

## 2.5 Stochastic Dynamics

The dynamics are **stochastic** due to unmodeled effects:

$$\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) + \boldsymbol{\omega}_k$$

where:

$$\boldsymbol{\omega}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$$

is zero-mean Gaussian process noise with diagonal covariance:

$$\mathbf{Q}_k = \text{diag}\left([\sigma_p^2]_{1\times 3}, [\sigma_v^2]_{1\times 3}, [\sigma_a^2]_{1\times 3}\right)$$

Typical values: $\sigma_p = 0.01$ m, $\sigma_v = 0.1$ m/s, $\sigma_a = 0.02$ rad.

## 2.6 Intuitive Understanding

### How Tilting Produces Motion

A quadrotor cannot translate without tilting (unlike a car). To move forward:

1. The rear rotors spin faster → pitch forward ($\theta > 0$)
2. Thrust vector tilts forward → horizontal acceleration component $g\tan\theta$
3. As velocity increases → drag $-k_D v_x$ opposes motion
4. At steady state: $g\tan\theta = k_D v_x$, so $v_x = g\tan\theta / k_D$

For $\theta = 10°$: $v_x \approx 9.81 \times 0.176 / 0.5 \approx 3.45$ m/s steady-state.

### Why First-Order Attitude Response?

The Bebop 2 uses an onboard PID controller to track commanded angles. The first-order model captures the closed-loop response:

$$\dot{\phi} = \frac{1}{\tau_\phi}(\phi_c - \phi)$$

This is equivalent to: $\phi(t) = \phi_c(1 - e^{-t/\tau_\phi})$, i.e., the angle exponentially approaches the commanded value with time constant $\tau_\phi = 0.2$ s.

### The Yaw Rotation $\mathbf{R}_Z(\psi)$

When the quadrotor yaws, its body-fixed axes rotate. The accelerations $g\tan\theta$ (forward) and $-g\tan\phi$ (right) must be rotated to world frame:

$$\mathbf{a}_{\text{world}} = \mathbf{R}_Z(\psi) \begin{bmatrix} g\tan\theta \\ -g\tan\phi \end{bmatrix}$$

At $\psi = 0$: Forward pitch → +x acceleration (moving north)
At $\psi = 90°$: Forward pitch → +y acceleration (moving east)

## 2.7 Model Limitations

1. **Small angle approximation**: The model uses $\tan\phi \approx \phi$ for small angles. For aggressive maneuvers ($\phi > 30°$), this becomes inaccurate.

2. **Neglected coupling**: In reality, yaw rotation couples with pitch/roll through gyroscopic effects (not modeled here).

3. **Constant drag**: $k_D$ is assumed constant, but drag depends on the quadrotor's orientation relative to airflow.

4. **Ground effect**: When flying close to the ground (< 0.5 m), extra lift reduces the effective thrust needed.

5. **First-order attitude**: This is a simplification of the Bebop's internal controller. Different quadrotors may have different attitude dynamics.

## 2.8 Verification

The model has been verified against real flight data. Key results:
- Position tracking error: ~0.05 m (with external motion capture)
- Velocity tracking: matches within 0.1 m/s
- Attitude response: time constant $\tau_\phi \approx 0.2$ s confirmed

## 2.9 Prerequisites and Related Chapters

> [!info] Prerequisites
- Calculus (derivatives, ODEs)
- Linear algebra (rotation matrices — see [[04_Rotation_Matrix|Ch.4]])

> [!info] Used In
- [[08_Linearization|Ch.8: Linearization]] — Jacobians of this model
- [[09_Discretization|Ch.9: Discretization]] — RK4 integration
- [[10_State_Space_Model|Ch.10: State-Space Model]] — Canonical form
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — Dynamics constraint in MPC

> [!info] See Also
- [[03_Coordinate_Frames|Ch.3: Coordinate Frames]] — Body vs. world frame
- [[04_Rotation_Matrix|Ch.4: Rotation Matrix]] — $\mathbf{R}_Z(\psi)$
- [[07_Newton_Euler|Ch.7: Newton-Euler Dynamics]] — Full 6-DOF derivation
