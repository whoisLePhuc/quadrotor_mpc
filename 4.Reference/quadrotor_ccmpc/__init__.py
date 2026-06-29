"""
Quadrotor Chance-Constrained MPC — obstacle avoidance for micro aerial vehicles.

Based on: Lin, Zhu, Alonso-Mora — ICRA 2020
  "Robust Vision-based Obstacle Avoidance for Micro Aerial Vehicles in Dynamic Environments"

Quick start:
    from ccmpc import CCMPC, ObstacleManager

    mpc = CCMPC("config/mpc.yaml")
    obstacles = ObstacleManager.from_config("config/simulation.yaml")
    trajectory, controls = mpc.solve(initial_state, goal, obstacle_manager=obstacles)
"""
