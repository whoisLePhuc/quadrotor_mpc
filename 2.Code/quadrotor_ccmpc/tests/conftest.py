"""Shared fixtures for CC-MPC test suite."""

import pathlib
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures: default state / control / config paths
# ---------------------------------------------------------------------------

PKG_ROOT = pathlib.Path(__file__).parent.parent
CONFIG_DIR = PKG_ROOT / "config"


@pytest.fixture
def hover_state() -> np.ndarray:
    """9D state at hover: [0,0,1, 0,0,0, 0,0,0]."""
    x = np.zeros(9)
    x[2] = 1.0  # z=1m (above ground)
    return x


@pytest.fixture
def max_pitch_cmd() -> np.ndarray:
    """4D control: max pitch forward, no roll/climb/yaw."""
    return np.array([0.0, 0.35, 0.0, 0.0])


@pytest.fixture
def max_roll_cmd() -> np.ndarray:
    """4D control: max roll right, no pitch/climb/yaw."""
    return np.array([0.35, 0.0, 0.0, 0.0])


@pytest.fixture
def zero_cmd() -> np.ndarray:
    """Zero control (hover)."""
    return np.zeros(4)


@pytest.fixture
def moving_state() -> np.ndarray:
    """9D state moving forward at 3 m/s, slight pitch."""
    x = np.zeros(9)
    x[0:3] = [2.0, 1.0, 1.5]   # position
    x[3:6] = [3.0, 0.0, 0.0]    # velocity
    x[6:9] = [0.0, 0.15, 0.5]   # attitude: slight pitch, some yaw
    return x


@pytest.fixture
def mpc_config() -> pathlib.Path:
    """Path to default MPC config."""
    return CONFIG_DIR / "mpc.yaml"


@pytest.fixture
def sim_config() -> pathlib.Path:
    """Path to default simulation config."""
    return CONFIG_DIR / "simulation.yaml"
