---
title: 'Chương 18: Implementation Notes'
chapter: 18
tags:
- ccmpc
- implementation
- code
- python
phase: implementation
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 18
- Implementation Notes
- Ch.18
---

## 18.1 Overview

This chapter bridges theory and code, documenting the `quadrotor_ccmpc` Python implementation. It serves as a companion to the source code at `2.Code/quadrotor_ccmpc/`.

## 18.2 Package Structure

```
quadrotor_ccmpc/
├── ccmpc/
│ ├── __init__.py
│ ├── ccmpc.py # Main CC-MPC controller
│ ├── _problem.py # CVXPY QP construction
│ ├── dynamics.py # Quadrotor dynamics (RK4, Jacobians)
│ ├── uncertainty.py # Covariance propagation (Eq 19)
│ ├── obstacle.py # EllipsoidalObstacle, ObstacleManager
│ ├── sensor.py # VIO, depth camera simulation
│ ├── mixer.py # Control allocation
│ ├── mujoco_dynamics.py # MuJoCo simulation interface
│ └── utils.py # Math utilities (erfinv, Omega, quaternion)
├── config/
│ ├── mpc.yaml # Main CC-MPC configuration
│ ├── simulation.yaml # Simulation scenario config
│ └── simulation_corridor.yaml
├── models/
│ └── quadrotor.xml # MuJoCo model
├── scripts/
│ └── benchmark.py # Performance benchmarking
├── tests/
│ ├── test_dynamics.py
│ ├── test_ccmpc.py
│ ├── test_uncertainty.py
│ ├── test_obstacle.py
│ ├── test_utils.py
│ └── test_mixer.py
├── sim_demo_mujoco.py # MuJoCo simulation demo
└── sim_demo_nosim.py # Headless simulation demo
```

## 18.3 Core Class: CCMPC

The central controller class in `ccmpc.py`:

```python
class CCMPC:
 def __init__(self, config, horizon_time=None, timestep=None, ...):
 # Load configuration
 # Create QuadrotorDynamics, UncertaintyPropagator
 # Build CVXPY problem ONCE (DPP-compliant)
 self._build_problem()
 
 def solve(self, initial_state, goal, obstacle_manager, Gamma_0, ...):
 # 1. Build warm-start guess
 # 2. Propagate uncertainty (Eq 19)
 # 3. iMPC loop:
 # a. Linearize dynamics
 # b. Set chance constraint params
 # c. Set logistic cost gradient
 # d. Solve QP
 # e. Check convergence
 # 4. Store for warm-start
 # 5. Return trajectory, controls
```

### Key Design Decisions

1. **Problem built once**: The CVXPY `Problem` is constructed in `__init__` and reused — only parameter values change between solves
2. **DPP compliance**: All constraints use `Parameter @ Variable` form (no Parameter×Parameter)
3. **Two warm-start layers**: The shifted trajectory initializes the next linearization, while CVXPY/CLARABEL may independently reuse numerical solver state
4. **Optional soft chance constraints**: Slack can relax collision inequalities but does not guarantee overall feasibility and removes the original hard probability guarantee whenever nonzero

## 18.4 Dynamics Implementation (`dynamics.py`)

### Continuous Dynamics

The `continuous_dynamics(x, u)` function computes $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$:

```python
def continuous_dynamics(x, u, g=9.81, kD=0.5, ...):
 # Position derivatives
 vx, vy, vz = x[3:6]
 dx, dy, dz = vx, vy, vz
 
 # Velocity derivatives
 dvx = g * (tan(theta)*cos(psi) + tan(phi)*sin(psi)) - kD * vx
 dvy = g * (tan(theta)*sin(psi) - tan(phi)*cos(psi)) - kD * vy
 dvz = (k_vz * vz_c - vz) / tau_vz
 
 # Attitude derivatives (first-order lags)
 dphi = (k_phi * phi_c - phi) / tau_phi
 dtheta = (k_theta * theta_c - theta) / tau_theta
 dpsi = psi_dot_c
 
 return [dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi]
```

### Dynamics Fidelity

The equations above reproduce the Appendix model of Zhu & Alonso-Mora (2019). A normalized thrust-vector “tilt factor” is a different physical model and must be implemented as a separately named model variant with its own Jacobians and validation. Under the adopted FLU convention, the roll contribution is $+g\tan\phi\sin\psi$ in $\dot v_x$ and $-g\tan\phi\cos\psi$ in $\dot v_y$.

### Discrete Integration (RK4)

```python
def discrete_step(x, u, dt, **params):
 k1 = f(x)
 k2 = f(x + 0.5*dt*k1)
 k3 = f(x + 0.5*dt*k2)
 k4 = f(x + dt*k3)
 return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
```

RK4 provides 4th-order accuracy vs. Euler's 1st-order.

### Linearization

```python
def linearize(self, x_bar, u_bar, dt):
 A_cont = self.jacobian_state(x_bar, u_bar)
 B_cont = self.jacobian_control(x_bar, u_bar)
 
 A_k = np.eye(9) + dt * A_cont # First-order A
 B_k = dt * B_cont + 0.5*dt**2 * (A_cont @ B_cont) # Second-order B
 
 x_next = discrete_step(x_bar, u_bar, dt) # True rollout
 C_k = x_next - A_k @ x_bar - B_k @ u_bar # Affine offset
 
 return A_k, B_k, C_k
```

**Critique of the second-order B correction**: The analytical Jacobian for control `jacobian_control()` is a simple sparse matrix because controls enter linearly in the attitude and vertical velocity equations. The `A_cont @ B_cont` term captures the cascade: $\theta_c \to \dot{\theta} \to \theta \to \dot{v}_x \to v_x$.

## 18.5 Uncertainty Propagation (`uncertainty.py`)

```python
class UncertaintyPropagator:
 def propagate(self, Gamma_0, x_guess, u_guess, dynamics, dt):
 Gamma_list = [Gamma_0.copy()]
 for k in range(N):
 A_cont = dynamics.jacobian_state(x_guess[:, k], u_guess[:, k])
 F_k = np.eye(9) + dt * A_cont
 W_d = self.discrete_process_covariance(A_cont, dt)
 Gamma_k1 = F_k @ Gamma_list[-1] @ F_k.T + W_d
 Gamma_list.append(Gamma_k1)
 return Gamma_list
```

`W_d` must be an explicitly discrete covariance. If the configuration stores a continuous-time spectral density `W_c`, compute a discrete approximation such as `W_c * dt` only under the corresponding additive-white-noise assumptions; do not multiply an already discrete per-step covariance by `dt` again.

## 18.6 Obstacle Model (`obstacle.py`)

```python
class EllipsoidalObstacle:
 def __init__(self, position, size, yaw=0.0, velocity=None, ...):
 self.p_hat = np.array(position)
 self.v_hat = np.array(velocity) if velocity else np.zeros(3)
 self.axes = box_to_ellipsoid_axes(size) # Eq (7)
 self.R_OW = yaw_to_world_to_obstacle_rotation(yaw)
 self.Sigma = np.diag([pos_uncertainty**2] * 3)
 self.Sigma_v = np.diag([vel_uncertainty**2] * 3)
 
 def predict(self, dt):
 self.p_hat += self.v_hat * dt # Eq (6)
 self.Sigma += self.Sigma_v * dt**2 # Eq (6)
 
 def get_omega(self, mav_radius):
 inv_sq = 1.0 / (self.axes + mav_radius)**2
 return self.R_OW.T @ np.diag(inv_sq) @ self.R_OW
 
 def get_collision_transform(self, mav_radius):
     # L @ L.T = Omega; the required transform U satisfies U.T @ U = Omega
     L = np.linalg.cholesky(self.get_omega(mav_radius))
     return L.T
```

## 18.7 Math Utilities (`utils.py`)

### Inverse Error Function

```python
def erfinv(y, tol=1e-12, max_iter=50):
 # Winitzki 2008 rational approximation as initial guess
 # Newton iteration for refinement
```

### Chance Constraint RHS

```python
def chance_constraint_parameters(U, p_mav_guess, p_obs,
                                 Sigma_mav, Sigma_obs, delta):
 d = p_mav_guess - p_obs
 d_tilde = U @ d
 n_o = d_tilde / np.linalg.norm(d_tilde)
 Sigma_combined = Sigma_mav + Sigma_obs
 inner_cov = U @ Sigma_combined @ U.T
 sigma_scaled = sqrt(2.0 * n_o @ inner_cov @ n_o)
 margin = erfinv(1.0 - 2.0 * delta) * sigma_scaled
 a_vec = U.T @ n_o
 return a_vec, margin + a_vec @ p_obs
```

**Cholesky convention**: For lower `L` with `L @ L.T = Omega`, the unit-sphere transform is `U = L.T`, so the transformed covariance is `U @ Sigma @ U.T = L.T @ Sigma @ L`. Using `L @ Sigma @ L.T` with the lower factor does not preserve $d^T\Omega d$ for a rotated obstacle.

## 18.8 CVXPY Problem Construction (`_problem.py`)

The QP is built once with DPP-compliant structure:

```python
def build_qp_problem(mpc):
 N = mpc.control_horizon
 cost = 0.0
 constraints = []
 
 # Dynamics constraints
 for k in range(N):
 constraints.append(
 states[:, k+1] == A_params[k] @ states[:, k] 
 + B_params[k] @ controls[:, k]
 + C_params[k]
 )
 
 # Chance constraints
 for k in range(N):
 for i_obs in range(max_obs):
 a_k_i = opt.Parameter(3) # U.T @ n
 rhs_k_i = opt.Parameter() # RHS + a^T @ p_obs
 slack_k_i = opt.Variable(nonneg=True)
 
 constraints.append(
 a_k_i @ states[:3, k+1] - 1.0 + slack_k_i >= rhs_k_i
 )
 cost += slack_penalty * slack_k_i
 
 # Costs, bounds, etc.
 ...
 
 mpc._problem = opt.Problem(opt.Minimize(cost), constraints)
```

## 18.9 Configuration (`mpc.yaml`)

The YAML configuration centralizes all parameters:

```yaml
model:
 quadrotor:
 g: 9.81
 kD: 0.5
 k_phi: 1.0
 tau_phi: 0.2
 # ...

controller:
 prediction:
 horizon_time: 1.8
 timestep: 0.06
 max_iter: 5
 tolerance: 0.01
 solver: CLARABEL
 
 obstacle:
 delta: 0.03
 mav_radius: 0.4
 slack_penalty: 1000.0
 max_obstacles: 2
 
 limits:
 max_roll: 0.35
 max_pitch: 0.35
 max_speed: 8.0
 # ...

 uncertainty:
 process_noise_pos: 0.01
 process_noise_vel: 0.1
 init_pos_noise: 0.05
 # ...
```

## 18.10 Key Implementation Insights

### 1. Optional Second-order B-Matrix
The $\frac{\Delta t^2}{2}\mathbf{A}_{\text{cont}}\mathbf{B}_{\text{cont}}$ term captures the leading within-step attitude-command coupling. It is a project approximation, not a formula from the papers; differentiating the RK4 discrete map is more internally consistent. Claims about oscillation must be supported by an ablation benchmark.

### 2. Affine Offset C_k
Use $\mathbf{C}_k=\mathbf{f}_d(\bar{\mathbf{x}},\bar{\mathbf{u}})-\mathbf{A}_k\bar{\mathbf{x}}-\mathbf{B}_k\bar{\mathbf{u}}$ so the affine model matches the discrete rollout at the linearization point. Any numerical error-per-step claim must come from a reproducible test.

### 3. Ellipsoid Transform Convention
The transform in $\|\mathbf{U}d\|^2=d^T\boldsymbol{\Omega}d$ must satisfy $\mathbf{U}^T\mathbf{U}=\boldsymbol{\Omega}$. NumPy returns lower `L` with `L @ L.T = Omega`, hence `U = L.T`. Use `U @ Sigma @ U.T` and compute the normal from `U @ d`.

### 4. Warm-Start Alignment
The first state of the shifted warm-start must be overwritten with the current measurement: `x_guess[:, 0] = initial_state`. This ensures the initial condition constraint is satisfied.

### 5. DPP Parameter Workaround
To avoid `Parameter @ Parameter` (forbidden in DPP), the chance constraint's RHS includes the term `a_vec @ p_obs` pre-computed as a scalar.

## 18.11 Testing

The test suite must be rerun after the frame, sign, and Cholesky corrections. A historical “13/13 passing” result is insufficient because tests can encode the same incorrect convention as the implementation.

Run tests:
```bash
cd 2.Code/quadrotor_ccmpc
python -m pytest tests/ -v
python ../verify_formulas.py
```

Required regression cases include non-axis-aligned ellipsoids, $\|Ud\|^2=d^T\Omega d$, Monte Carlo collision-bound checks, roll/pitch sign tests, covariance frame transforms, and nonzero-slack reporting.

## 18.12 Prerequisites and Related Chapters

> [!info] Prerequisites

> [!info] See Also
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — Algorithm description
- [[16_Optimization|Ch.16: Optimization Formulation]] — QP formulation
- [[17_Solver|Ch.17: Solver & Real-Time]] — Solver details
- [[20_Reference_Formula|Ch.20: Reference Formulas]] — Complete formula index
