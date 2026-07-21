---
title: 'Chương 9: Discretization'
chapter: 9
tags:
- quadrotor
- control
- discretization
- rk4
- numerical
phase: foundations
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 9
- Discretization
- Ch.9
---

## 9.1 Introduction

The quadrotor dynamics are naturally described in **continuous time**:

$$\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$$

However, both the MPC optimization and the real-world control loop operate in **discrete time** at fixed intervals $\Delta t$:

$$\mathbf{x}_{k+1} = \mathbf{f}_d(\mathbf{x}_k, \mathbf{u}_k)$$

**Discretization** converts the continuous ODE to a discrete-time difference equation. The choice of discretization method affects accuracy, stability, and computation time.

## 9.2 Methods

### Forward Euler (1st Order)

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \Delta t \cdot \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k)$$

- **Pros**: Simplest, fastest
- **Cons**: 1st-order accuracy O(Δt), can be unstable for stiff systems
- **Usage**: Not used for dynamics (insufficient accuracy), used for Jacobian discretization

### Runge-Kutta 4th Order (RK4)

$$\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) \\
\mathbf{k}_2 &= \mathbf{f}(\mathbf{x}_k + \tfrac{\Delta t}{2}\mathbf{k}_1, \mathbf{u}_k) \\
\mathbf{k}_3 &= \mathbf{f}(\mathbf{x}_k + \tfrac{\Delta t}{2}\mathbf{k}_2, \mathbf{u}_k) \\
\mathbf{k}_4 &= \mathbf{f}(\mathbf{x}_k + \Delta t\mathbf{k}_3, \mathbf{u}_k) \\
\mathbf{x}_{k+1} &= \mathbf{x}_k + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
\end{aligned}$$

- **Pros**: Fourth-order global accuracy $O(\Delta t^4)$ and local truncation error $O(\Delta t^5)$ for sufficiently smooth dynamics
- **Cons**: 4 function evaluations per step
- **Usage**: Primary discretization in CC-MPC implementation

### Why RK4?

Error-order notation does not directly produce a position error in metres; the omitted constants and trajectory derivatives matter. RK4 should therefore be selected from convergence tests against a smaller-step reference integration, not from multiplying powers of $\Delta t$ by the number of horizon steps.

## 9.3 Discrete-Time State-Space Model

After discretization, the dynamics become:

$$\boxed{\mathbf{x}_{k+1} = \mathbf{f}_d(\mathbf{x}_k, \mathbf{u}_k) + \boldsymbol{\omega}_k}$$

where the process noise $\boldsymbol{\omega}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$ accounts for discretization error and unmodeled effects.

## 9.4 Linearized Discrete-Time Model

For the iMPC, we need the linearized discrete-time model:

$$\mathbf{x}_{k+1} \approx \mathbf{A}_k\mathbf{x}_k + \mathbf{B}_k\mathbf{u}_k + \mathbf{C}_k$$

### First-Order Discretization of Jacobians

Given the continuous Jacobians $\mathbf{A}_{\text{cont}} = \frac{\partial\mathbf{f}}{\partial\mathbf{x}}$ and $\mathbf{B}_{\text{cont}} = \frac{\partial\mathbf{f}}{\partial\mathbf{u}}$:

**Standard first-order**:
$$\begin{aligned}
\mathbf{A}_k &= \mathbf{I} + \Delta t \cdot \mathbf{A}_{\text{cont}} \\
\mathbf{B}_k &= \Delta t \cdot \mathbf{B}_{\text{cont}}
\end{aligned}$$

**Optional second-order frozen-Jacobian approximation used by this project implementation**:
$$\mathbf{B}_k = \Delta t \cdot \mathbf{B}_{\text{cont}} + \frac{\Delta t^2}{2} \cdot \mathbf{A}_{\text{cont}} \cdot \mathbf{B}_{\text{cont}}$$

The second-order term captures **cascade coupling** — the effect of controls acting through the state during the timestep. See [[08_Linearization|Ch.8: Linearization]] for detailed derivation.

## 9.5 Process Noise Discretization

If $\mathbf{W}$ denotes a continuous-time white-noise spectral density entering additively through $\mathbf{G}=\mathbf{I}$, a first-order approximation is:

$$\mathbf{Q}_k = \mathbf{W} \cdot \Delta t$$

In general:

$$\boxed{\mathbf{Q}_d = \int_0^{\Delta t} e^{\mathbf{A}\tau}\mathbf{G}\mathbf{Q}_c\mathbf{G}^T e^{\mathbf{A}^T\tau}\,d\tau}$$

If $\mathbf{W}$ is already a per-step discrete covariance, it must be added directly and must not be multiplied by $\Delta t$ again. The source papers denote discrete process covariance by $\mathbf{Q}^k$ or $\mathbf{W}^k$ without prescribing the project-specific scaling rule.

## 9.6 Implementation

```python
def discrete_step(x, u, dt, g=9.81, kD=0.5, ...):
 """RK4 integration of quadrotor dynamics."""
 def f(xk):
 return continuous_dynamics(xk, u, g=g, kD=kD, ...)
 
 k1 = f(x)
 k2 = f(x + 0.5 * dt * k1)
 k3 = f(x + 0.5 * dt * k2)
 k4 = f(x + dt * k3)
 
 return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
```

## 9.7 Discretization of Covariance Propagation

The EKF uncertainty propagation (Eq. 19) discretizes the state Jacobian:

$$\mathbf{F}_k = \mathbf{I} + \Delta t \cdot \mathbf{A}_{\text{cont}}$$

Then:
$$\boldsymbol{\Gamma}_{k+1} = \mathbf{F}_k \boldsymbol{\Gamma}_k \mathbf{F}_k^T + \mathbf{W}\Delta t$$

> [!note] 
> This is a first-order approximation. When RK4 is already available, differentiating the RK4 discrete map or integrating the variational equations gives a more consistent transition Jacobian and should be used when covariance accuracy is important.

## 9.8 Timestep Selection

The timestep $\Delta t$ is a critical design parameter:

| $\Delta t$ (s) | Horizon Steps ($N$) for a project-specific 1.8 s horizon | Computation | Control Resolution |
|----------------|-------------------------------|-------------|------------------|
| 0.10 | 18 | Fastest | Low (coarse control) |
| **0.06** | **30** | **Balanced** | **Good** |
| 0.03 | 60 | Slow | High (fine control) |
| 0.01 | 180 | Too slow | Overkill |

$\Delta t = 0.06$ s and $N=30$ are project implementation choices. The source papers use 0.05 s with a 1.0 s horizon (2019) and 0.06 s with a 1.5 s horizon (2020). The project setting may be retained if justified by measured solve time and closed-loop tests because:
1. It matches the control loop frequency (~16.7 Hz)
2. It provides sufficient resolution for obstacle avoidance
3. Its measured solve time on the actual implementation and hardware can be checked against the 60 ms deadline; the papers' runtimes are not CLARABEL benchmarks

## 9.9 Interpolation for Control

Between MPC solves (every $\Delta t = 0.06$ s), the control command may need to be applied at a higher rate (e.g., 100 Hz for the low-level controller). Linear interpolation or zero-order hold can be used.

## 9.10 Prerequisites and Related Chapters

> [!info] Prerequisites
- [[02_Quadrotor_Dynamics|Ch.2: Quadrotor Dynamics]] — Continuous ODE
- [[08_Linearization|Ch.8: Linearization]] — Jacobian discretization

> [!info] Used In
- [[10_State_Space_Model|Ch.10: State-Space Model]] — Discrete-time representation
- [[11_MPC|Ch.11: Model Predictive Control]] — Prediction model
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — Dynamics constraint
- [[14_Covariance_Propagation|Ch.14: Covariance Propagation]] — Discrete propagation
