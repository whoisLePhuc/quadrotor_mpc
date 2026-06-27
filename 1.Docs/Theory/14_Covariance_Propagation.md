---
title: 'Chương 14: Covariance Propagation'
chapter: 14
tags:
- ccmpc
- uncertainty
- ekf
- covariance
phase: control-theory
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 14
- Covariance Propagation
- Ch.14
---

## 14.1 Introduction

To evaluate chance constraints at each step of the prediction horizon, we need the robot's **position uncertainty covariance** $\boldsymbol{\Sigma}^k$ at each future step $k$.

**Covariance propagation** is the process of computing how uncertainty evolves over time under the nonlinear dynamics:

$$\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) + \boldsymbol{\omega}_k$$

where $\boldsymbol{\omega}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$ is process noise.

## 14.2 Approaches to Uncertainty Propagation

There are three main approaches (from least to most expensive):

| Method | Accuracy | Computation | Used Here? |
|--------|----------|-------------|------------|
| **EKF linearization** | Good for short horizons | Very fast | ✅ Yes |
| Unscented Transform (UKF) | Better for strong nonlinearities | Moderate | No |
| Polynomial Chaos Expansion | Excellent for long horizons | Expensive | No |

The EKF approach is chosen because:
1. The prediction horizon is short (1–2 seconds)
2. Real-time performance is critical (~16 Hz control loop)
3. For the quadrotor dynamics over short horizons, linearization error is small

## 14.3 EKF-Type Propagation (Eq. 19)

The state covariance evolves according to the linearized dynamics:

$$\boxed{\boldsymbol{\Gamma}^{k+1} = \mathbf{F}^k \boldsymbol{\Gamma}^k \mathbf{F}^{k\;T} + \mathbf{W}^k}$$

where:

- $\boldsymbol{\Gamma}^k \in \mathbb{R}^{9 \times 9}$: Full state covariance at step $k$
- $\mathbf{F}^k = \frac{\partial \mathbf{f}}{\partial \mathbf{x}}\big|_{\hat{\mathbf{x}}^k, \mathbf{u}^k}$: State transition Jacobian (9×9)
- $\mathbf{W}^k$: Process noise covariance (9×9), typically $\mathbf{W} \cdot \Delta t$

### Initial Covariance

$$\boldsymbol{\Gamma}^0 = \text{diag}\left([\sigma_{p0}^2]_{1\times 3}, [\sigma_{v0}^2]_{1\times 3}, [\sigma_{a0}^2]_{1\times 3}\right)$$

Typical values from VIO:
- $\sigma_{p0} = 0.05$ m (position uncertainty)
- $\sigma_{v0} = 0.1$ m/s (velocity uncertainty)
- $\sigma_{a0} = 0.03$ rad (attitude uncertainty)

### Process Noise

$$\mathbf{W} = \text{diag}\left([\sigma_{p,w}^2]_{1\times 3}, [\sigma_{v,w}^2]_{1\times 3}, [\sigma_{a,w}^2]_{1\times 3}\right)$$

Typical values:
- $\sigma_{p,w} = 0.01$ m (per-step position noise)
- $\sigma_{v,w} = 0.1$ m/s (per-step velocity noise)
- $\sigma_{a,w} = 0.02$ rad (per-step attitude noise)

### Position Covariance Extraction

The position covariance (3×3 block) used in chance constraints is:

$$\boldsymbol{\Sigma}^k = \boldsymbol{\Gamma}^k_{[0:3, 0:3]}$$

(the top-left 3×3 submatrix).

## 14.4 State Transition Jacobian

$\mathbf{F}^k = \frac{\partial \mathbf{f}}{\partial \mathbf{x}}$ at the linearization point $(\hat{\mathbf{x}}^k, \mathbf{u}^k)$.

For the quadrotor dynamics, this is a 9×9 matrix computed via finite differences:

```python
def jacobian_state(f, x, u, eps=1e-6):
 """Central finite difference Jacobian."""
 n = len(x)
 J = np.zeros((n, n))
 for i in range(n):
 dx = np.zeros(n)
 dx[i] = eps
 J[:, i] = (f(x + dx, u) - f(x - dx, u)) / (2 * eps)
 return J
```

The analytical form has significant structure:

$$\mathbf{F} = \begin{bmatrix}
\mathbf{0}_{3\times 3} & \mathbf{I}_{3\times 3} & \mathbf{0}_{3\times 3} \\
\mathbf{0}_{3\times 3} & -k_D\mathbf{I}_{3\times 3} & \frac{\partial \dot{\mathbf{v}}}{\partial \boldsymbol{\eta}} \\
\mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \frac{\partial \dot{\boldsymbol{\eta}}}{\partial \boldsymbol{\eta}}
\end{bmatrix}$$

where:
- Position row: $\dot{\mathbf{p}} = \mathbf{v}$ → Jacobian is $[\mathbf{0}, \mathbf{I}, \mathbf{0}]$
- Velocity row: $\dot{\mathbf{v}}$ depends on attitude through $\tan\phi$, $\tan\theta$
- Attitude row: $\dot{\phi} = -\phi/\tau_\phi$, etc. → diagonal Jacobian

### Key Coupling: $\frac{\partial \dot{\mathbf{v}}}{\partial \boldsymbol{\eta}}$

This is the critical coupling that makes uncertainty grow. At hover ($\phi = \theta = 0, \psi = 0$):

$$\frac{\partial \dot{v}_x}{\partial \theta} = g, \quad \frac{\partial \dot{v}_x}{\partial \phi} = 0$$
$$\frac{\partial \dot{v}_y}{\partial \theta} = 0, \quad \frac{\partial \dot{v}_y}{\partial \phi} = -g$$

At non-zero attitude angles, the trigonometric derivatives become more complex:

$$\frac{\partial \dot{v}_x}{\partial \theta} = g \cdot \frac{1}{\cos^2\theta}(\cos\psi - \tan\phi \sin\psi)$$
$$\frac{\partial \dot{v}_x}{\partial \phi} = g \cdot \frac{1}{\cos^2\phi}(\tan\theta \sin\psi + \sin\psi)$$

## 14.5 Why Last-Loop Trajectory?

**Key design choice**: The covariance propagation uses the **last-loop trajectory** as the linearization point, NOT the current optimization variables.

**Reason**: If $\mathbf{F}^k$ depends on the current optimization variables $\hat{\mathbf{x}}^k$ and $\mathbf{u}^k$, then:
- $\mathbf{F}^k = \mathbf{F}^k(\hat{\mathbf{x}}^k, \mathbf{u}^k)$
- $\boldsymbol{\Gamma}^{k+1}$ depends on $\mathbf{F}^k$
- The chance constraint depends on $\boldsymbol{\Gamma}^{k+1}$
- This introduces $N(n_x^2 + n_x)$ additional variables in the optimization

With last-loop propagation, $\mathbf{F}^k$ is fixed before solving the QP, keeping the problem size manageable.

**Justification**: In receding-horizon MPC, the trajectory changes only slightly between consecutive solves (especially near convergence), so the linearization error is small.

## 14.6 Implementation

```python
class UncertaintyPropagator:
 def propagate(self, Gamma_0, x_guess, u_guess, dynamics, dt):
 """Propagate covariance over prediction horizon.
 
 Args:
 Gamma_0: Initial state covariance (9×9)
 x_guess: Last-loop state trajectory (9, N+1)
 u_guess: Last-loop control sequence (4, N)
 dynamics: Quadrotor dynamics model
 dt: Time step
 
 Returns:
 List of covariances [Gamma_0, ..., Gamma_N]
 """
 N = x_guess.shape[1] - 1
 Gamma_list = [Gamma_0.copy()]
 
 for k in range(N):
 Gamma_k = Gamma_list[-1]
 A_cont = dynamics.jacobian_state(x_guess[:, k], u_guess[:, k])
 F_k = np.eye(9) + dt * A_cont # Discrete-time Jacobian
 Gamma_k1 = F_k @ Gamma_k @ F_k.T + self.W * dt
 Gamma_list.append(Gamma_k1)
 
 return Gamma_list
```

## 14.7 Verification

The following properties are verified numerically:

1. **Monotonic growth**: $\text{tr}(\boldsymbol{\Gamma}^{k+1}) > \text{tr}(\boldsymbol{\Gamma}^k)$ for all $k$ (uncertainty increases with time)
2. **Positive definiteness**: All $\boldsymbol{\Gamma}^k$ have positive eigenvalues
3. **Position submatrix valid**: $\boldsymbol{\Sigma}^k = \boldsymbol{\Gamma}^k_{[0:3,0:3]}$ is positive definite

```python
# Verification: trace grows monotonically
traces = [np.trace(Gamma_0)]
for k in range(N):
 Gamma = F @ Gamma @ F.T + W
 traces.append(np.trace(Gamma))
assert all(traces[i] > traces[i-1] for i in range(1, len(traces)))
```

## 14.8 Obstacle Uncertainty Propagation

Obstacles use a simpler constant-velocity model (Eq. 6):

$$\begin{aligned}
\hat{\mathbf{p}}_o^{k+1} &= \hat{\mathbf{p}}_o^k + \hat{\mathbf{v}}_o^k \Delta t \\
\hat{\mathbf{v}}_o^{k+1} &= \hat{\mathbf{v}}_o^k \\
\boldsymbol{\Sigma}_o^{k+1} &= \boldsymbol{\Sigma}_o^k + \boldsymbol{\Sigma}_{o,v} \Delta t^2
\end{aligned}$$

Key properties:
- Position propagates linearly (constant velocity)
- Velocity remains constant (no acceleration model)
- Position uncertainty grows quadratically with time: $\boldsymbol{\Sigma}_o^k = \boldsymbol{\Sigma}_o^0 + k \cdot \boldsymbol{\Sigma}_{o,v} \Delta t^2$

**Note on velocity uncertainty**: If obstacle velocity estimation is very noisy (large $\boldsymbol{\Sigma}_{o,v}$), the predicted covariance grows rapidly. In practice, $\boldsymbol{\Sigma}_{o,v}$ is bounded to prevent excessive growth.

## 14.9 VIO Drift Model

For realistic simulation, a time-correlated drift model is used:

$$\mathbf{b}_{t+1} = \mathbf{b}_t + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, \mathbf{Q}_{\text{drift}} \cdot \Delta t)$$

where:
- $\mathbf{b}_t \in \mathbb{R}^9$: Bias on state estimate
- $\mathbf{Q}_{\text{drift}}$: Per-second drift noise variances

The VIO estimate is then: $\hat{\mathbf{x}}_{\text{VIO}} = \mathbf{x}_{\text{true}} + \mathbf{b} + \boldsymbol{\nu}$

where $\boldsymbol{\nu} \sim \mathcal{N}(0, \mathbf{R}_{\text{VIO}})$ is instantaneous measurement noise.

## 14.10 Prerequisites and Related Chapters

> [!info] Prerequisites
- [[02_Quadrotor_Dynamics|Ch.2: Quadrotor Dynamics]] (Quadrotor Dynamics)
- [[08_Linearization|Ch.8: Linearization]] — For computing $\mathbf{F}^k$
- Probability theory (covariance matrices)

> [!info] Used In
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — $\boldsymbol{\Sigma}^k$ in chance constraints
- [[13_Chance_Constraints|Ch.13: Chance Constraints]] — $\boldsymbol{\Sigma}^k$ determines constraint tightness

> [!info] See Also
- [[10_State_Space_Model|Ch.10: State-Space Model]] — Stochastic system formulation
