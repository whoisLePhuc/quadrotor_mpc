---
title: 'Chương 7: Newton-Euler Dynamics'
chapter: 7
tags:
- quadrotor
- dynamics
- newton-euler
- rigid-body
phase: foundations
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 7
- Newton-Euler Dynamics
- Ch.7
---

## 7.1 Introduction

The **Newton-Euler** formulation provides a first-principles derivation of quadrotor dynamics from forces and torques. While the CC-MPC uses a simplified first-order model ([[02_Quadrotor_Dynamics|Ch.2]]), understanding the full Newton-Euler derivation provides insight into:
- Where the simplified model comes from
- What physics is being approximated
- How to extend the model for other quadrotors

## 7.2 Newton's Second Law (Translation)

$$\sum \mathbf{F} = m\ddot{\mathbf{p}}$$

For a quadrotor, the forces are:
1. **Thrust**: $\mathbf{T} = T\mathbf{R}_B^W\mathbf{e}_3$ (along body Z-axis, rotated to world)
2. **Gravity**: $\mathbf{G} = -mg\mathbf{e}_3$ (downward in world frame)
3. **Drag**: $\mathbf{D} = -k_D\dot{\mathbf{p}}$ (opposing velocity)

$$\ddot{\mathbf{p}} = \frac{1}{m}\left(T\mathbf{R}_B^W\mathbf{e}_3 - mg\mathbf{e}_3 - k_D\dot{\mathbf{p}}\right)$$

where $T = c_T(\omega_1^2 + \omega_2^2 + \omega_3^2 + \omega_4^2)$ is the total thrust from all four rotors.

## 7.3 Euler's Equation (Rotation)

$$\mathbf{J}\dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (\mathbf{J}\boldsymbol{\omega}) = \boldsymbol{\tau}$$

where:
- $\mathbf{J}$: Inertia matrix (diagonal for symmetric quadrotor)
- $\boldsymbol{\omega} = [p, q, r]^T$: Body-frame angular velocity
- $\boldsymbol{\tau}$: External torques

For a quadrotor with "+" configuration:

$$\mathbf{J} = \begin{bmatrix} J_x & 0 & 0 \\ 0 & J_y & 0 \\ 0 & 0 & J_z \end{bmatrix}$$

The torques are:

$$\begin{aligned}
\tau_x &= l c_T(\omega_4^2 - \omega_2^2) \quad \text{(roll torque, from differential thrust)} \\
\tau_y &= l c_T(\omega_1^2 - \omega_3^2) \quad \text{(pitch torque)} \\
\tau_z &= c_Q(\omega_1^2 - \omega_2^2 + \omega_3^2 - \omega_4^2) \quad \text{(yaw torque, from differential drag)}
\end{aligned}$$

where $l$ is the arm length and $c_T, c_Q$ are thrust and torque coefficients.

## 7.4 Relationship to the Simplified Model

The simplified Bebop 2 model ([[02_Quadrotor_Dynamics|Ch.2]]) is a **first-order low-pass Euler approximation** of the full Newton-Euler dynamics. The key simplications:

### Thrust → Velocity

The full model: $\ddot{\mathbf{p}} = \frac{T}{m}\mathbf{R}_B^W\mathbf{e}_3 - g\mathbf{e}_3 - \frac{k_D}{m}\dot{\mathbf{p}}$

At hover: $T = mg$, so the thrust vector exactly balances gravity. When tilting:

$$\mathbf{R}_B^W\mathbf{e}_3 = \begin{bmatrix} \cos\psi\sin\theta\cos\phi + \sin\psi\sin\phi \\ \sin\psi\sin\theta\cos\phi - \cos\psi\sin\phi \\ \cos\theta\cos\phi \end{bmatrix}$$

For small angles ($\phi, \theta \ll 1$):

$$\mathbf{R}_B^W\mathbf{e}_3 \approx \begin{bmatrix} \theta\cos\psi + \phi\sin\psi \\ \theta\sin\psi - \phi\cos\psi \\ 1 \end{bmatrix}$$

So horizontal acceleration $\approx \frac{T}{m}(\theta\cos\psi + \phi\sin\psi)$.

The simplified model replaces this with:

$$\begin{bmatrix} \dot{v}_x \\ \dot{v}_y \end{bmatrix} = \mathbf{R}_Z(\psi) \begin{bmatrix} g\tan\theta \\ -g\tan\phi \end{bmatrix} - k_D\begin{bmatrix} v_x \\ v_y \end{bmatrix}$$

Note: The simplified model uses $g\tan\theta$ instead of $\frac{T}{m}\theta$. This is equivalent when $T \approx mg$ (near hover) since $g\tan\theta \approx g\theta \approx \frac{T}{m}\theta$ for small angles.

### Attitude Dynamics

The full model requires computing torques from rotor speeds, then integrating Euler's equation. The simplified model replaces this with first-order systems:

$$\dot{\phi} = \frac{1}{\tau_\phi}(\phi_c - \phi), \quad \dot{\theta} = \frac{1}{\tau_\theta}(\theta_c - \theta), \quad \dot{\psi} = \dot{\psi}_c$$

This captures the **closed-loop response** of the Bebop 2's onboard attitude controller.

## 7.5 Rotor Speed Mixing

For completeness, the mapping from commanded attitude to rotor speeds (for a "+" configuration):

$$\begin{bmatrix} \omega_1^2 \\ \omega_2^2 \\ \omega_3^2 \\ \omega_4^2 \end{bmatrix} = \begin{bmatrix}
1 & 0 & -1 & 1 \\
1 & -1 & 0 & -1 \\
1 & 0 & 1 & 1 \\
1 & 1 & 0 & -1
\end{bmatrix} \begin{bmatrix} T/(4c_T) \\ \tau_x/(2lc_T) \\ \tau_y/(2lc_T) \\ \tau_z/(4c_Q) \end{bmatrix}$$

The CC-MPC does not use this directly — the Bebop 2 handles mixing internally.

## 7.6 State Vector in Newton-Euler Form

The full 12-state Newton-Euler model:

$$\mathbf{x}_{\text{NE}} = \begin{bmatrix} \mathbf{p} \\ \dot{\mathbf{p}} \\ \mathbf{q} \\ \boldsymbol{\omega} \end{bmatrix} = \begin{bmatrix} x \\ y \\ z \\ \dot{x} \\ \dot{y} \\ \dot{z} \\ w \\ x_q \\ y_q \\ z_q \\ p \\ q \\ r \end{bmatrix} \in \mathbb{R}^{13}$$

The CC-MPC uses a reduced 9-state model with Euler angles instead of quaternion + angular velocity.

## 7.7 Why the Simplified Model?

1. **Lower dimension**: 9 states vs. 13 — quadratic reduction in QP size
2. **Simpler linearization**: Euler angle Jacobians are cleaner
3. **Matched to Bebop 2 API**: The Bebop accepts $\phi_c, \theta_c, v_{zc}, \dot{\psi}_c$ directly
4. **Adequate accuracy**: For the short prediction horizon (1.8 s), the first-order model captures the essential dynamics
5. **Real-time requirement**: The simpler model = faster optimization

## 7.8 Prerequisites and Related Chapters

> [!info] Prerequisites
- Physics (Newton's laws, rigid body dynamics)
- [[04_Rotation_Matrix|Ch.4: Rotation Matrix]] (Rotation Matrix)
- [[05_Euler_Angles|Ch.5: Euler Angles]] (Euler Angles)

> [!info] Used In
- [[02_Quadrotor_Dynamics|Ch.2: Quadrotor Dynamics]] — The simplified model derived from this

> [!info] See Also
- [[03_Coordinate_Frames|Ch.3: Coordinate Frames]] — Frame definitions
