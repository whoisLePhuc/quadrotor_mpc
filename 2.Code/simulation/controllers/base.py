"""Base controller abstraction for the quadrotor simulation.

This module defines only the public controller protocol/base class.

Controller data contracts such as ControllerInput, ControllerOutput,
ControllerDiagnostics, and ControllerMetadata live in
``simulation.controllers.metadata``.

For backward compatibility during the refactor, this module re-exports the
metadata/data-contract names that used to be defined here.  Existing imports
such as ``from simulation.controllers.base import ControllerInput`` should keep
working while new code may import those contracts directly from
``simulation.controllers.metadata``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from simulation.controllers.metadata import (
    DEFAULT_CCMPC_METADATA,
    DEFAULT_EMERGENCY_STOP_METADATA,
    ControllerConfigurationError,
    ControllerDiagnostics,
    ControllerError,
    ControllerInput,
    ControllerInputError,
    ControllerMetadata,
    ControllerOutput,
    ControllerOutputError,
    ControllerSolveError,
    ControllerStatus,
    ControllerType,
    ObstaclePrediction,
    first_control_from_trajectory,
    make_fallback_output,
    make_success_output,
    validate_controller_time,
)


@runtime_checkable
class ControllerProtocol(Protocol):
    """Structural protocol for controllers."""

    def reset(self) -> None:
        """Reset controller internal state."""

    def compute_command(self, input_data: ControllerInput) -> ControllerOutput:
        """Compute one high-level ControlCommand4."""

    def get_metadata(self) -> ControllerMetadata:
        """Return static controller metadata."""

    def close(self) -> None:
        """Release controller resources."""


class Controller(ABC):
    """Abstract base class for all controllers."""

    @abstractmethod
    def reset(self) -> None:
        """Reset controller internal state."""

    @abstractmethod
    def compute_command(self, input_data: ControllerInput) -> ControllerOutput:
        """Compute one high-level ControlCommand4."""

    @abstractmethod
    def get_metadata(self) -> ControllerMetadata:
        """Return static controller metadata."""

    def close(self) -> None:
        """Release controller resources.

        Stateless controllers may keep the default no-op implementation.
        """


__all__ = [
    "DEFAULT_CCMPC_METADATA",
    "DEFAULT_EMERGENCY_STOP_METADATA",
    "Controller",
    "ControllerConfigurationError",
    "ControllerDiagnostics",
    "ControllerError",
    "ControllerInput",
    "ControllerInputError",
    "ControllerMetadata",
    "ControllerOutput",
    "ControllerOutputError",
    "ControllerProtocol",
    "ControllerSolveError",
    "ControllerStatus",
    "ControllerType",
    "ObstaclePrediction",
    "first_control_from_trajectory",
    "make_fallback_output",
    "make_success_output",
    "validate_controller_time",
]
