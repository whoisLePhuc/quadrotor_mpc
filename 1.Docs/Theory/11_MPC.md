---
title: 'Chương 11: Model Predictive Control'
chapter: 11
tags:
- mpc
- control
- optimization
- receding-horizon
phase: control-theory
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 11
- Model Predictive Control
- Ch.11
---

## 11.1 Introduction

Model Predictive Control (MPC) is an optimization-based control strategy that:
1. Uses a **model** of the system to predict future states
2. Solves an **optimization** problem over a finite horizon
3. Applies only the **first** control input (receding horizon)
4. Repeats at each time step with new measurements

**Why MPC for quadrotors?**
- Handles constraints (actuator limits, obstacle avoidance) naturally
- Anticipates future obstacles (look-ahead capability)
- Can incorporate nonlinear dynamics
- Provides optimal trajectories (vs. reactive controllers)

## 11.2 Basic Formulation

### System Model

Discrete-time nonlinear dynamics:

$$\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k)$$

where:
- $\mathbf{x}_k \in \mathbb{R}^{n_x}$: State at step $k$
- $\mathbf{u}_k \in \mathbb{R}^{n_u}$: Control input at step $k$
- $\mathbf{f}: \mathbb{R}^{n_x} \times \mathbb{R}^{n_u} \to \mathbb{R}^{n_x}$: Dynamics

### Receding Horizon Optimization

At each time step $t$, solve:

$$\begin{aligned}
\min_{\mathbf{u}^{0:N-1}} \quad & J^N(\mathbf{x}^N) + \sum_{k=0}^{N-1} J^k(\mathbf{x}^k, \mathbf{u}^k) \\
\text{s.t.} \quad & \mathbf{x}^{k+1} = \mathbf{f}(\mathbf{x}^k, \mathbf{u}^k), \quad k = 0, \ldots, N-1 \\
& \mathbf{x}^0 = \mathbf{x}(t) \quad \text{(current state)} \\
& \mathbf{u}^k \in \mathcal{U}, \quad \mathbf{x}^k \in \mathcal{X}
\end{aligned}$$

Then apply $\mathbf{u}^*(0)$ to the system, wait one time step, and repeat.

## 11.3 Cost Function Components

### Terminal Cost $J^N$

Guarantees stability by penalizing the final state:

$$J^N(\mathbf{x}^N) = \|\mathbf{x}^N - \mathbf{x}_{\text{ref}}\|_{\mathbf{P}}^2$$

where $\mathbf{P} \succeq 0$ is the terminal cost matrix (solution of the discrete algebraic Riccati equation for linear systems).

In the CC-MPC implementation, the terminal cost is on **position only**:

$$J^N(\hat{\mathbf{x}}^N) = \|\hat{\mathbf{p}}^N - \mathbf{p}_g\|_{\mathbf{Q}_g}^2$$

### Stage Cost $J^k$

Encodes the control objectives at each step:

$$J^k(\mathbf{x}^k, \mathbf{u}^k) = \|\mathbf{x}^k - \mathbf{x}_{\text{ref}}^k\|_{\mathbf{Q}}^2 + \|\mathbf{u}^k\|_{\mathbf{R}}^2$$

where:
- $\mathbf{Q} \succeq 0$: State penalty
- $\mathbf{R} \succ 0$: Control effort penalty

## 11.4 Prediction Horizon

The choice of horizon $N$ involves trade-offs:

| Property | Short Horizon | Long Horizon |
|----------|---------------|--------------|
| Computation | Fast | Slow |
| Obstacle anticipation | Poor | Good |
| Stability | May need terminal cost | Easier to guarantee |
| Deadlock avoidance | Prone to | Less prone |

**CC-MPC defaults**: $N = 30$, $\Delta t = 0.06$ s → $T = 1.8$ s horizon.

## 11.5 Linear vs. Nonlinear MPC

### Nonlinear MPC (NMPC)

- Uses the full nonlinear model $\mathbf{f}(\mathbf{x}, \mathbf{u})$
- Nonlinear program (NLP) — harder to solve
- More accurate for aggressive maneuvers
- Used by: ACADO, CasADi, Forces Pro

### Linear Time-Varying MPC (LTV-MPC / iMPC)

- Linearizes dynamics around a nominal trajectory
- Quadratic program (QP) — fast and reliable
- Requires iteration (iMPC) for nonlinear systems
- Used by: CC-MPC implementation (CVXPY + CLARABEL)

The CC-MPC uses **iMPC**: iterate between solving a QP and re-linearizing until convergence.

## 11.6 iMPC Algorithm

```
Algorithm: Iterative Model Predictive Control
──────────────────────────────────────────────
Input: Current state x_0, goal p_g, obstacles
Output: Optimal control sequence u*

1. Initialize guess trajectory:
 (x_guess, u_guess) ← warm-start or straight line

2. For iter = 1 to max_iters:
 a. For k = 0 to N-1:
 Linearize dynamics at (x_guess[k], u_guess[k]):
 A_k, B_k, C_k ← linearize(f, x_guess[k], u_guess[k], dt)
 
 b. Build QP with linearized dynamics
 c. Solve QP → (x_new, u_new)
 d. If converged (max|x_new - x_guess| < tol): break
 e. Update guess: (x_guess, u_guess) ← (x_new, u_new)

3. Return (x_new, u_new)
```

**Convergence**: Typically 2–3 iterations. Max: 5 iterations.

## 11.7 Constraints in MPC

### Hard Constraints
- Must be satisfied exactly
- Example: actuator limits $\mathbf{u}^k \in [\mathbf{u}_{\text{min}}, \mathbf{u}_{\text{max}}]$
- Risk: Can cause infeasibility

### Soft Constraints
- Allowed to be violated with penalty
- Example: obstacle avoidance with slack variable $s \geq 0$
- Form: $g(\mathbf{x}) \leq s$, with $s$ heavily penalized in cost
- Benefits: Always feasible, prioritizes critical constraints

### CC-MPC uses soft chance constraints:
$$\mathbf{a}^T\hat{\mathbf{p}} - 1 + s \geq \text{rhs}, \quad s \geq 0, \quad \text{cost} += 1000 \cdot s$$

## 11.8 Stability and Feasibility

### Recursive Feasibility
If the optimization at step $t$ is feasible, the shifted solution (dropping first step) should be feasible at step $t+1$. This is **not guaranteed** in CC-MPC due to:
- Moving obstacles
- Changing chance constraint parameters
- Nonlinear dynamics

**Fallback**: PID controller when QP is infeasible.

### Stability
For linear MPC with terminal cost, stability can be proven via Lyapunov arguments. For nonlinear iMPC with soft constraints, stability is empirical — in practice, the receding horizon strategy works well when:
- Horizon is long enough (1–2 s)
- Re-planning frequency is high (≥ 15 Hz)
- Fallback controller exists

## 11.9 Implementation Considerations

### Timestep Selection

$$\Delta t = 0.06 \text{ s} \quad \text{(≈ 16.7 Hz control)}$$

- Must be fast enough to react to moving obstacles
- Must be slow enough for solver to complete
- Based on Bebop 2 command interface (ROS at 15 Hz)

### Warm-Starting

The previous solution provides an excellent initial guess:
- Reduced iterations: 5 → 2–3
- Better convergence: Avoids local minima
- Implementation: Shift trajectory by 1, duplicate last step

### Solver Integration

CVXPY provides a clean interface:
```python
problem = opt.Problem(opt.Minimize(cost), constraints)
problem.solve(solver=opt.CLARABEL, warm_start=False)
```

Note: warm_start is set to False because DPP parameters change between iterations; the warm-start is done manually via guess trajectory.

## 11.10 Prerequisites and Related Chapters

> [!info] Prerequisites
- [[02_Quadrotor_Dynamics|Ch.2: Quadrotor Dynamics]] — The system model
- [[08_Linearization|Ch.8: Linearization]] — iMPC relies on linearization
- [[09_Discretization|Ch.9: Discretization]] — Converting continuous to discrete
- [[10_State_Space_Model|Ch.10: State-Space Model]] — System representation
- Control theory (cost functions, constraints)

> [!info] Used In
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — Extended with chance constraints
- [[16_Optimization|Ch.16: Optimization Formulation]] — QP formulation

> [!info] See Also
- [[17_Solver|Ch.17: Solver & Real-Time]] — Solver details
