---
title: 'Chương 8: Linearization'
chapter: 8
tags:
- quadrotor
- control
- linearization
- jacobian
- taylor
phase: foundations
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 8
- Linearization
- Ch.8
---

## 8.1 Introduction

The quadrotor dynamics $\mathbf{f}(\mathbf{x}, \mathbf{u})$ are **nonlinear** due to:
- Trigonometric functions (sin, cos, tan of Euler angles)
- Product terms ($v_x \cos\psi$, etc.)
- Aerodynamic drag ($-k_D v_x$)

However, standard MPC solvers work best with **linear (or quadratic) constraints and costs**. Linearization approximates the nonlinear system as a linear time-varying (LTV) system around a nominal trajectory:

$$\mathbf{x}_{k+1} \approx \mathbf{A}_k \mathbf{x}_k + \mathbf{B}_k \mathbf{u}_k + \mathbf{C}_k$$

This is the standard **iterative MPC (iMPC)** approach: solve a QP, update the linearization point, repeat until convergence.

## 8.2 Taylor Series Expansion

### First-Order (Standard Linearization)

For a general nonlinear system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$:

$$\mathbf{f}(\mathbf{x}, \mathbf{u}) \approx \mathbf{f}(\bar{\mathbf{x}}, \bar{\mathbf{u}}) + \underbrace{\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\bigg|_{\bar{\mathbf{x}},\bar{\mathbf{u}}}}_{\mathbf{A}_{\text{cont}}}(\mathbf{x} - \bar{\mathbf{x}}) + \underbrace{\frac{\partial \mathbf{f}}{\partial \mathbf{u}}\bigg|_{\bar{\mathbf{x}},\bar{\mathbf{u}}}}_{\mathbf{B}_{\text{cont}}}(\mathbf{u} - \bar{\mathbf{u}})$$

where $(\bar{\mathbf{x}}, \bar{\mathbf{u}})$ is the linearization point (guess trajectory).

### Discrete-Time Linearization

For discrete-time with step $\Delta t$, the standard first-order discretization is:

$$\begin{aligned}
\mathbf{A}_k &= \mathbf{I} + \Delta t \cdot \mathbf{A}_{\text{cont}} \\
\mathbf{B}_k &= \Delta t \cdot \mathbf{B}_{\text{cont}}
\end{aligned}$$

This is the **forward Euler** discretization of the Jacobians.

## 8.3 State Jacobian $\frac{\partial \mathbf{f}}{\partial \mathbf{x}}$

The continuous-time state Jacobian $\mathbf{A}_{\text{cont}}$ is a 9×9 matrix. Let's examine each block.

### Position Row (rows 0–2)

$$\dot{\mathbf{p}} = \mathbf{v}$$

$$\frac{\partial \dot{\mathbf{p}}}{\partial \mathbf{x}} = \begin{bmatrix} \mathbf{0}_{3\times 3} & \mathbf{I}_{3\times 3} & \mathbf{0}_{3\times 3} \end{bmatrix}$$

### Velocity Row (rows 3–5)

$$\begin{aligned}
\dot{v}_x &= g(\tan\theta \cos\psi + \tan\phi \sin\psi) - k_D v_x \\
\dot{v}_y &= g(\tan\theta \sin\psi - \tan\phi \cos\psi) - k_D v_y \\
\dot{v}_z &= \frac{1}{\tau_{vz}}(k_{vz} v_{zc} - v_z)
\end{aligned}$$

**With respect to velocity**:

$$\frac{\partial \dot{\mathbf{v}}}{\partial \mathbf{v}} = \begin{bmatrix} -k_D & 0 & 0 \\ 0 & -k_D & 0 \\ 0 & 0 & -1/\tau_{vz} \end{bmatrix}$$

**With respect to attitude** (the critical coupling):

$$\begin{aligned}
\frac{\partial \dot{v}_x}{\partial \phi} &= g \cdot \frac{\sin\psi}{\cos^2\phi} \\
\frac{\partial \dot{v}_x}{\partial \theta} &= g \cdot \frac{\cos\psi}{\cos^2\theta} \\
\frac{\partial \dot{v}_x}{\partial \psi} &= g(-\tan\theta \sin\psi + \tan\phi \cos\psi) \\
\\
\frac{\partial \dot{v}_y}{\partial \phi} &= -g \cdot \frac{\cos\psi}{\cos^2\phi} \\
\frac{\partial \dot{v}_y}{\partial \theta} &= g \cdot \frac{\sin\psi}{\cos^2\theta} \\
\frac{\partial \dot{v}_y}{\partial \psi} &= g(\tan\theta \cos\psi + \tan\phi \sin\psi)
\end{aligned}$$

**At hover** ($\phi = \theta = \psi = 0$):

$$\frac{\partial \dot{\mathbf{v}}}{\partial \boldsymbol{\eta}}\bigg|_{\text{hover}} = \begin{bmatrix}
0 & g & 0 \\
-g & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}$$

This shows the fundamental coupling: pitch $\theta$ affects $\dot{v}_x$, roll $\phi$ affects $\dot{v}_y$ (with gravity $g$ as the gain).

### Attitude Row (rows 6–8)

$$\begin{aligned}
\dot{\phi} &= \frac{1}{\tau_\phi}(k_\phi \phi_c - \phi) \\
\dot{\theta} &= \frac{1}{\tau_\theta}(k_\theta \theta_c - \theta) \\
\dot{\psi} &= \dot{\psi}_c
\end{aligned}$$

$$\frac{\partial \dot{\boldsymbol{\eta}}}{\partial \boldsymbol{\eta}} = \begin{bmatrix}
-1/\tau_\phi & 0 & 0 \\
0 & -1/\tau_\theta & 0 \\
0 & 0 & 0
\end{bmatrix}$$

## 8.4 Control Jacobian $\frac{\partial \mathbf{f}}{\partial \mathbf{u}}$

$$\mathbf{B}_{\text{cont}} = \frac{\partial \mathbf{f}}{\partial \mathbf{u}} = \begin{bmatrix}
\mathbf{0}_{3\times 4} \\
\mathbf{0}_{3\times 4} \\
\begin{bmatrix}
k_\phi/\tau_\phi & 0 & 0 & 0 \\
0 & k_\theta/\tau_\theta & 0 & 0 \\
0 & 0 & k_{vz}/\tau_{vz} & 0
\end{bmatrix} \\
\begin{bmatrix}
0 & 0 & 0 & 1
\end{bmatrix}
\end{bmatrix}$$

Only the attitude and vertical velocity derivatives depend directly on control:
- $\partial\dot{\phi}/\partial\phi_c = k_\phi/\tau_\phi$
- $\partial\dot{\theta}/\partial\theta_c = k_\theta/\tau_\theta$
- $\partial\dot{v}_z/\partial v_{zc} = k_{vz}/\tau_{vz}$
- $\partial\dot{\psi}/\partial\dot{\psi}_c = 1$

## 8.5 Second-Order B-Matrix Correction

**Critical insight** from the implementation: The standard first-order $\mathbf{B}_k = \Delta t \cdot \mathbf{B}_{\text{cont}}$ fails to capture important coupling at hover.

### The Problem

At hover ($\theta = 0, v_x = 0$), the first-order model predicts:

$$v_x^{k+1} = v_x^k + \Delta t(g\theta^k - k_D v_x^k) = 0 + 0.06(0 - 0) = 0$$

This is **wrong**: a pitch command $\theta_c$ causes $\theta$ to rise during the step, which should produce velocity change. The first-order model misses this because it treats $\theta$ as constant during the step.

### The Solution

Use a second-order Taylor expansion for $\mathbf{B}$:

$$\boxed{\mathbf{B}_k = \Delta t \cdot \mathbf{B}_{\text{cont}} + \frac{\Delta t^2}{2} \cdot \mathbf{A}_{\text{cont}} \cdot \mathbf{B}_{\text{cont}}}$$

The $\frac{\Delta t^2}{2}\mathbf{A}_{\text{cont}}\mathbf{B}_{\text{cont}}$ term captures the **cascade coupling**:

$$\theta_c \xrightarrow{k_\theta/\tau_\theta} \dot{\theta} \xrightarrow{\Delta t} \theta \xrightarrow{g} \dot{v}_x \xrightarrow{\Delta t} v_x$$

Without this term (first-order $\mathbf{B}$):

```python
# First-order: B_k = dt * B_cont
# At hover: vx_{k+1} = vx_k + dt * (g * theta_k - kD * vx_k) + 0*theta_c
# Since theta_k = 0, vx_k = 0: vx_{k+1} = 0 ← WRONG
```

With second-order correction:

```python
# Second-order: B_k = dt*B_cont + dt²/2 * A_cont * B_cont
# Additional term: dt²/2 * d(dvx)/dtheta * d(dtheta)/dtheta_c * theta_c
# = 0.06²/2 * g * (k_theta/tau_theta) * theta_c
# = 0.0018 * 9.81 * 5.0 * theta_c ≈ 0.088 * theta_c ← CORRECT
```

## 8.6 Affine Offset $\mathbf{C}_k$

The standard linearization $\mathbf{x}_{k+1} = \mathbf{A}_k\mathbf{x}_k + \mathbf{B}_k\mathbf{u}_k$ assumes the linearization point is an equilibrium. In general, it's not, so we need an affine offset:

$$\mathbf{C}_k = \mathbf{x}_{k+1}^{\text{true}} - \mathbf{A}_k\bar{\mathbf{x}}_k - \mathbf{B}_k\bar{\mathbf{u}}_k$$

where $\mathbf{x}_{k+1}^{\text{true}} = \text{RK4}(\bar{\mathbf{x}}_k, \bar{\mathbf{u}}_k, \Delta t)$ is the actual discrete rollout.

**Why this is important**: At the linearization point, the linear model should **exactly** match the true model:

$$\mathbf{A}_k\bar{\mathbf{x}}_k + \mathbf{B}_k\bar{\mathbf{u}}_k + \mathbf{C}_k = \mathbf{x}_{k+1}^{\text{true}}$$

Without $\mathbf{C}_k$, there can be 7 mm/step error that accumulates over the horizon.

## 8.7 Implementation

```python
def linearize(self, x_bar, u_bar, dt):
 """Linearize around (x_bar, u_bar)."""
 # Continuous Jacobians
 A_cont = jacobian_state(x_bar, u_bar)
 B_cont = jacobian_control(x_bar, u_bar)
 
 # Discrete-time matrices
 A_k = np.eye(9) + dt * A_cont # First-order A
 B_k = dt * B_cont + 0.5 * dt**2 * (A_cont @ B_cont) # Second-order B
 
 # Affine offset for exact match at expansion point
 x_next = discrete_step(x_bar, u_bar, dt) # RK4
 C_k = x_next - A_k @ x_bar - B_k @ u_bar
 
 return A_k, B_k, C_k
```

## 8.8 Numerical Jacobian (Finite Differences)

For the 9D state, an analytical Jacobian can be derived but is error-prone. The implementation uses central finite differences:

```python
def jacobian(f, x0, eps=1e-6):
 n_out = len(f(x0))
 n_in = len(x0)
 J = np.zeros((n_out, n_in))
 for i in range(n_in):
 dx = np.zeros(n_in)
 dx[i] = eps
 J[:, i] = (f(x0 + dx) - f(x0 - dx)) / (2 * eps)
 return J
```

This computes all 9×9 = 81 entries in a loop but is:
- **Simple** to implement and verify
- **Accurate** to O(eps²) (central difference)
- **Fast enough** for 9D state (81 function evaluations ≈ microseconds)

## 8.9 Verification

The linearization is verified by checking:
1. $\mathbf{A}_k\bar{\mathbf{x}}_k + \mathbf{B}_k\bar{\mathbf{u}}_k + \mathbf{C}_k = \mathbf{x}_{k+1}^{\text{true}}$ (within machine precision at linearization point)
2. For small perturbations $\delta\mathbf{x}$, the linear prediction matches RK4 to O(δx²)
3. The second-order B correction significantly improves accuracy at hover

## 8.10 The iMPC Loop

The linearization is embedded in an iterative scheme:

```
Algorithm: Iterative MPC
─────────────────────────
1. Start with guess trajectory (x_guess, u_guess)
 - Either warm-start from previous solution
 - Or straight line toward goal

2. For iter = 1 to max_iter:
 a. Linearize at (x_guess, u_guess) → A_k, B_k, C_k
 b. Set up chance constraint params
 c. Solve QP → (x_new, u_new)
 d. If max|x_new - x_guess| < tol: CONVERGED
 e. Update guess: (x_guess, u_guess) = (x_new, u_new)

3. Return converged trajectory
```

Convergence typically occurs in 2–3 iterations. The tolerance is 0.01 (max state element change).

## 8.11 Prerequisites and Related Chapters

> [!info] Prerequisites
- Calculus (partial derivatives, Taylor series)
- [[02_Quadrotor_Dynamics|Ch.2: Quadrotor Dynamics]] — The function being linearized

> [!info] Used In
- [[09_Discretization|Ch.9: Discretization]] — Converting continuous to discrete
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — Dynamics constraint in MPC
- [[14_Covariance_Propagation|Ch.14: Covariance Propagation]] — State transition matrix
