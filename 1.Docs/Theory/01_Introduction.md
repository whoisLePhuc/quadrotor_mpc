---
title: 'Chương 1: Introduction'
chapter: 1
tags:
- quadrotor
- introduction
- overview
phase: foundations
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 1
- Introduction
- Ch.1
---

## 1.1 Problem Statement

Autonomous navigation of Micro Aerial Vehicles (MAVs) in cluttered, dynamic environments presents a fundamental challenge in robotics. The MAV must:

1. **Navigate to a goal**: Plan and execute trajectories from start to goal position
2. **Avoid obstacles**: Detect and avoid both static and moving obstacles (other robots, humans)
3. **Handle uncertainty**: Account for sensing noise, state estimation errors, and motion disturbances
4. **Run in real-time**: Solve the planning problem online while sensing and state-estimation processes run at their own rates

The core difficulty is that **uncertainty is inevitable** in real-world operation:
- **State estimation uncertainty**: Visual-inertial odometry (VIO) accumulates drift; GPS has noise
- **Obstacle sensing uncertainty**: Depth cameras have quadratic range error; occlusions occur
- **Motion disturbances**: Wind gusts, ground effects, unmodeled dynamics

A purely deterministic planner that ignores these uncertainties will fail when the actual state deviates from the planned trajectory. This is demonstrated experimentally in Zhu & Alonso-Mora (2019): a deterministic MPC approach succeeded in only 64% of trials under moderate noise (0.05 m position error), dropping to 36% under high noise (0.09 m).

## 1.2 The Chance-Constrained Approach

> [!tip] Key insight 
> Instead of requiring *guaranteed* collision avoidance (which is impossible with unbounded Gaussian noise), we require *probabilistic* safety:

> The probability of collision with each modeled obstacle at each prediction step is upper-bounded by a user-specified threshold $\delta$, under the Gaussian, independence, prediction, and linearization assumptions of the papers.

This is formalized as a **chance constraint**:

$$\mathbb{P}(\mathbf{x}_k \notin \mathcal{C}_k) \geq 1 - \delta$$

where $\mathcal{C}_k$ is the collision region at time step $k$.

For $\delta = 0.03$, this means: *"I am 97% confident that I will not collide at this step."*

### Why Not Deterministic?

Deterministic approaches typically use **bounding volumes** (e.g., 3-$\sigma$ confidence ellipsoids) to inflate obstacles and robots. While fast, this:
- **Over-estimates** collision probability significantly
- Leads to **infeasible solutions** in cluttered environments
- Is overly **conservative** — avoids safe configurations

The chance-constrained approach constructs a tractable **upper bound** on collision probability by enclosing the nonlinear collision region with a linear half-space. It does not evaluate the exact Gaussian probability integral over the ellipsoid.

### Comparison of Methods

| Method | Collision Prob. | Computation (ms) | Feasible? |
|--------|----------------|-------------------|-----------|
| Numerical integration (truth) | 0.011 | 258.7 | Yes |
| Bounding volume (3-σ) | 1.0 | 0.011 | No |
| Center point PDF | 3.6×10⁻¹⁸ | 0.016 | Yes (unsafe) |
| Cube approximation | 0.100 | 0.044 | No |
| **Our method (CC-MPC)** | **0.017** | **0.011** | **Yes** |

*Table from Zhu & Alonso-Mora (2019), Table I*

## 1.3 System Architecture

The complete system (from Lin, Zhu & Alonso-Mora, 2020) consists of three components:

```
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ State Estimation │ │ Obstacle Sensing │ │ CC-MPC Planner │
│ (UKF or VIO)     │ │ (Depth → U-depth │ │ (online NMPC)   │
│                  │ │ → Box → Ellipsoid)│ │                 │
└────────┬──────────┘ └────────┬──────────┘ └────────┬──────────┘
 │ │ │
 ▼ ▼ ▼
 x̂₀, Γ₀ p̂ₒ, v̂ₒ, Σₒ u* (control command)
 │ │ │
 └────────────────────────┼────────────────────────┘
 │
 ▼
 ┌──────────────────┐
 │ Quadrotor │
 │ (Parrot Bebop 2) │
 └──────────────────┘
```

**State Estimation**: The 2019 work uses a UKF with motion-capture-derived measurements, while the 2020 system uses S-MSCKF visual-inertial odometry at 15 Hz. In both cases the planner receives $\hat{\mathbf{x}}_0$ and $\boldsymbol{\Gamma}_0$.

**Obstacle Sensing**: Depth images (60 Hz) → U-depth maps → box detection → ellipsoidal model with position $\hat{\mathbf{p}}_o$, velocity $\hat{\mathbf{v}}_o$, size $(a,b,c)$, and uncertainty $\boldsymbol{\Sigma}_o$.

**CC-MPC Planner**: The source papers solve a chance-constrained nonlinear MPC problem. The QP/iMPC formulation described later in this document set is a separate implementation choice and must not be presented as the solver formulation used in the papers.

## 1.4 Key Mathematical Concepts Used

This knowledge base covers the following theoretical foundations:

1. **Rigid body kinematics**: Rotation matrices, Euler angles, quaternions ([[03_Coordinate_Frames|Ch.3]]–6)
2. **Newton-Euler dynamics**: Forces and torques on a quadrotor ([[07_Newton_Euler|Ch.7]])
3. **Linearization**: Taylor expansion of nonlinear dynamics ([[08_Linearization|Ch.8]])
4. **Discretization**: Continuous-to-discrete time conversion ([[09_Discretization|Ch.9]])
5. **State-space representation**: $\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k)$ form ([[10_State_Space_Model|Ch.10]])
6. **Optimal control**: Cost functions, constraints, receding horizon ([[11_MPC|Ch.11]], 16)
7. **Probabilistic constraints**: Gaussian chance constraints ([[13_Chance_Constraints|Ch.13]])
8. **Uncertainty propagation**: EKF-style covariance evolution ([[14_Covariance_Propagation|Ch.14]])
9. **Convex optimization**: Quadratic programming (QP) ([[17_Solver|Ch.17]])

## 1.5 Notation Conventions

Throughout this knowledge base:

- **Bold lowercase** $\mathbf{x}$: vectors
- **Bold uppercase** $\mathbf{M}$: matrices
- **Hat** $\hat{\mathbf{x}}$: mean/estimate of random variable
- **Superscript** $\mathbf{x}^k$: value at time step $k$
- **Subscript** $\mathbf{x}_i$: belonging to robot/obstacle $i$
- **Calligraphic** $\mathcal{C}$: sets/regions
- $\|\mathbf{x}\|$: Euclidean norm
- $\|\mathbf{x}\|_{\mathbf{Q}} := \mathbf{x}^T\mathbf{Q}\mathbf{x}$: the papers' notation for a weighted squared norm (not the conventional square-root norm)
- $\mathbb{P}[\cdot]$: probability of event
- $p[\cdot]$: probability density function

## 1.6 Prerequisites

- Linear algebra (matrix multiplication, eigenvalues, Cholesky decomposition)
- Calculus (partial derivatives, Taylor series)
- Probability theory (Gaussian distributions, covariance)
- Basic control theory (state-space models, feedback)

## 1.7 Related Chapters

- **Next**: [[02_Quadrotor_Dynamics|Ch.2: Quadrotor Dynamics]]
- **Reference**: [[19_Glossary|Ch.19: Glossary]]
- **Reference**: [[20_Reference_Formula|Ch.20: Reference Formulas]]
