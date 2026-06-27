# quadrotor_mpc

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Status](https://img.shields.io/badge/status-in%20development-yellow)]()

**Chance-Constrained Model Predictive Control for quadrotor obstacle avoidance in dynamic environments.**

> From theory to implementation — probabilistically safe quadrotor navigation among moving obstacles under state estimation and sensing uncertainty.

---

## Overview

Autonomous quadrotors in cluttered, dynamic environments face three challenges:

1. **Uncertainty** — localization noise, sensing error, motion disturbances
2. **Moving obstacles** — humans, other robots with unpredictable intentions
3. **Real-time constraints** — planning must complete in milliseconds

A purely deterministic planner fails under uncertainty. Experimental results show deterministic MPC succeeding in only 64% of trials at moderate noise, dropping to 36% at high noise. This repository implements a **Chance-Constrained NMPC** that:

- **Models uncertainty explicitly** — Gaussian distributions propagated via EKF
- **Guarantees probabilistic safety** — collision probability ≤ δ per time step
- **Runs in real time** — ~14 ms per MPC solve
- **Handles 3D ellipsoidal obstacles** — from vision (depth camera) or simulation

Based on:
- Zhu & Alonso-Mora (2019) — *Chance-Constrained Collision Avoidance for MAVs* (IEEE RA-L)
- Lin, Zhu & Alonso-Mora (2020) — *Robust Vision-based Obstacle Avoidance for MAVs* (IEEE ICRA)

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌─────────────────┐
│ State Estim. │    │Obstacle Det. │    │   CC-MPC Core   │
│  (VIO / UKF) │    │(Depth → Box  │    │  (QP Solver)    │
│              │    │ → Ellipsoid) │    │                 │
└──────┬───────┘    └──────┬───────┘    └────────┬────────┘
       │                   │                     │
       ▼                   ▼                     ▼
   x̂₀, Γ₀            p̂ₒ, v̂ₒ, Σₒ              u* (command)
       │                    │                     │
       └────────────────────┼─────────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │  Quadrotor     │
                   │ (Sim / Real)   │
                   └────────────────┘
```

---

## Quick Start

```bash
git clone https://github.com/whoisLePhuc/quadrotor_mpc.git
cd quadrotor_mpc
pip install -e .
python sim_demo.py
```

> ⚠️ Code is under active development. See [Roadmap](#roadmap) below.

---

## Documentation

Complete theory reference in [`1.Docs/Theory/`](1.Docs/Theory/):

| Phase | Chapters | Topics |
|-------|----------|--------|
| Foundations (Ch.1–10) | 10 chapters | Quadrotor dynamics, coordinate frames, rotation, linearization, discretization, state-space |
| Control Theory (Ch.11–14) | 4 chapters | MPC, chance constraints (Lemmas 1–2), covariance propagation |
| CC-MPC Core (Ch.12,15) | 2 chapters | Full CC-MPC formulation, obstacle avoidance pipeline |
| Implementation (Ch.16–18) | 3 chapters | Optimization (QP), solver, code architecture |
| Reference (Ch.19–20) | 2 chapters | Glossary, complete formula index |

Start with [`1.Docs/Theory/README.md`](1.Docs/Theory/README.md) — Map of Content with knowledge graph, reading roadmap, and Dataview queries.

---

## Core Theory

### Chance Constraint → Deterministic Reformulation

A probabilistic collision constraint $\mathbb{P}(\text{collision}) \leq \delta$ is transformed into a deterministic inequality on the state mean and covariance:

$$\mathbf{n}_o^T\boldsymbol{\Omega}^{1/2}(\hat{\mathbf{p}} - \hat{\mathbf{p}}_o) - 1 \geq \text{erf}^{-1}(1-2\delta)\sqrt{2\mathbf{n}_o^T\boldsymbol{\Omega}^{1/2}(\boldsymbol{\Sigma} + \boldsymbol{\Sigma}_o)\boldsymbol{\Omega}^{1/2\;T}\mathbf{n}_o}$$

where $\boldsymbol{\Omega}$ is the ellipsoidal collision matrix, $\mathbf{n}_o$ is the unit normal from obstacle to robot, and $\delta = 0.03$ (97% confidence per step).

### Key Formulas

| # | Formula | Source |
|---|---------|--------|
| Stochastic dynamics | $\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) + \boldsymbol{\omega}_k,\ \boldsymbol{\omega}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)$ | Ch.2, 10 |
| Covariance propagation | $\boldsymbol{\Gamma}^{k+1} = \mathbf{F}^k \boldsymbol{\Gamma}^k \mathbf{F}^{k\;T} + \mathbf{W}\Delta t$ | Ch.14 |
| Quadrotor velocity | $\dot{\mathbf{v}} = \mathbf{R}_Z(\psi)[g\tan\theta,\ -g\tan\phi]^T - k_D\mathbf{v}$ | Ch.2 |
| Box → ellipsoid | $(a,b,c) = \frac{\sqrt{3}}{2}(l,w,h)$ | Ch.15 |
| Collision matrix | $\boldsymbol{\Omega}_{io} = \mathbf{R}_o^T \text{diag}(\frac{1}{(a+r)^2}, \frac{1}{(b+r)^2}, \frac{1}{(c+r)^2})\mathbf{R}_o$ | Ch.15 |

---

## Repository Structure

```
quadrotor_mpc/
├── 1.Docs/                  # Theory + papers
│   ├── Theory/              # 21-chapter knowledge base (Obsidian-ready)
│   └── Paper/               # Original paper PDFs
├── 2.Code/                  # Implementation
│   ├── quadrotor_mpc/       # Python package
│   │   ├── dynamics.py      # Quadrotor model (RK4, Jacobians)
│   │   ├── ccmpc.py         # CC-MPC controller
│   │   ├── obstacle.py      # Ellipsoidal obstacle model
│   │   ├── uncertainty.py   # EKF covariance propagation
│   │   ├── utils.py         # Math utilities (erfinv, Omega, quaternion)
│   │   └── ...
│   ├── config/              # YAML configuration files
│   └── tests/               # Unit + formula verification tests
├── 3.Notebooks/             # Jupyter notebooks
└── README.md
```

---

## Roadmap

- [x] Complete theory documentation (21 chapters)
- [x] Formula verification suite (13/13 tests pass)
- [x] Quadrotor dynamics model with linearization
- [x] CC-MPC controller (CVXPY + CLARABEL)
- [x] Obstacle model (ellipsoidal, Kalman tracking)
- [x] Uncertainty propagation (EKF)
- [x] Multi-robot coordination (DC strategy)
- [x] MuJoCo simulation environment
- [ ] Vision-based obstacle detection pipeline
- [ ] ROS2 integration
- [ ] Hardware deployment (Bebop 2 / custom quadrotor)
- [ ] GPU-accelerated solver (cuRobo / acados)

---

## Performance

From the original experiments (Zhu & Alonso-Mora, 2019):

| Scenario | CC-MPC solve | Framework total |
|----------|-------------|-----------------|
| 2 quadrotors | 14.3 ms | 71.3 ms |
| 6 quadrotors (DC) | 16.2 ms | — |
| 16 quadrotors (DC) | 24.7 ms | — |

| Metric | CC-MPC (δ=0.03) | Deterministic MPC | Bounding Volume |
|--------|-----------------|-------------------|-----------------|
| Safety (success rate) | 100% | 64% | 100% |
| Trajectory efficiency | 7.08 m | 6.74 m | 7.09 m |
| Min. separation | 0.81 m | 0.64 m | 0.87 m |

CC-MPC achieves the same safety as bounding volumes with **shorter, more efficient trajectories**.

---

## Citation

```bibtex
@article{zhu2019chance,
  title   = {Chance-Constrained Collision Avoidance for MAVs in Dynamic Environments},
  author  = {Zhu, Hai and Alonso-Mora, Javier},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {4},
  number  = {2},
  pages   = {776--783},
  year    = {2019}
}

@inproceedings{lin2020robust,
  title     = {Robust Vision-based Obstacle Avoidance for MAVs in Dynamic Environments},
  author    = {Lin, Jiahao and Zhu, Hai and Alonso-Mora, Javier},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages     = {2682--2688},
  year      = {2020}
}
```

---

## License

MIT
