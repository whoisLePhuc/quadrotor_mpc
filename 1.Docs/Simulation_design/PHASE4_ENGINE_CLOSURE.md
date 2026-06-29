# Phase 4 Engine Interface Closure

## Scope

This closure refactor finalizes the public physics-engine boundary before Phase 6 logging.

## Canonical contracts

| Engine | Public input | Public output | Internal details |
|---|---|---|---|
| ODEPhysicsEngine | `ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]` | `State9` | reduced-order ODE dynamics |
| MuJoCoPhysicsEngine | `ActuatorCommand4 = [T1, T2, T3, T4]` | `State9` | MuJoCo `MjModel`, `MjData`, `qpos`, `qvel`, quaternion |

## Boundary decisions

- MuJoCo does **not** accept `ControlCommand4` at the engine boundary.
- `ControlCommand4 -> ActuatorCommand4` mixing stays outside `MuJoCoPhysicsEngine`.
- MuJoCo `qpos/qvel` and quaternion state are private backend details.
- Runtime/controller/logging should consume only `StepResult`, `EngineMetadata`, and canonical arrays.

## Files updated/added

- `2.Code/simulation/engines/base.py`
- `2.Code/simulation/engines/factory.py`
- `2.Code/simulation/engines/__init__.py`
- `2.Code/simulation/engines/mujoco_engine.py`
- `2.Code/simulation/engines/adapters/mujoco_state_adapter.py`
- `2.Code/simulation/engines/adapters/mujoco_actuator_adapter.py`
- `2.Code/tests/unit/test_mujoco_metadata_policy.py`
- `2.Code/tests/unit/test_mujoco_state_adapter.py`
- `2.Code/tests/unit/test_mujoco_actuator_adapter.py`
- `2.Code/tests/interface/test_mujoco_engine_interface.py`

## Validation commands

```bash
cd 2.Code
pytest tests/unit/test_engine_base.py -v
pytest tests/unit/test_engine_factory.py -v
pytest tests/unit/test_ode_engine.py -v
pytest tests/unit/test_mujoco_metadata_policy.py -v
pytest tests/unit/test_mujoco_state_adapter.py -v
pytest tests/unit/test_mujoco_actuator_adapter.py -v
pytest tests/interface/test_mujoco_engine_interface.py -v
pytest tests/unit -v
pytest tests/interface -v
```

If MuJoCo is not installed, `test_mujoco_engine_interface.py` should skip.
