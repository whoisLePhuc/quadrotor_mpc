"""Thread/process-safe command vocabulary for the interactive MuJoCo runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, SimpleQueue
from typing import Any


class CommandName(str, Enum):
    TOGGLE_PAUSE = "toggle_pause"
    STEP = "step"
    RESET = "reset"
    STOP = "stop"
    SNAPSHOT = "snapshot"
    TOGGLE_TRAIL = "toggle_trail"
    TOGGLE_PREDICTION = "toggle_prediction"
    TOGGLE_SAFETY = "toggle_safety"
    TOGGLE_CAMERA = "toggle_camera"


@dataclass(frozen=True, slots=True)
class RuntimeCommand:
    name: CommandName
    source: str = "runtime"
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_message(cls, message: Any, *, source: str = "panel") -> "RuntimeCommand":
        if isinstance(message, RuntimeCommand):
            return message
        if isinstance(message, str):
            return cls(CommandName(message), source=source)
        if isinstance(message, dict):
            return cls(
                CommandName(message["name"]),
                source=str(message.get("source", source)),
                payload=dict(message.get("payload", {})),
            )
        raise ValueError(f"unsupported runtime command: {message!r}")

    def as_message(self) -> dict[str, Any]:
        return {"name": self.name.value, "source": self.source, "payload": self.payload}


class LocalCommandQueue:
    """Small non-blocking queue used by the native viewer key callback."""

    def __init__(self) -> None:
        self._queue: SimpleQueue[RuntimeCommand] = SimpleQueue()

    def put(self, command: RuntimeCommand) -> None:
        self._queue.put(command)

    def drain(self) -> list[RuntimeCommand]:
        commands: list[RuntimeCommand] = []
        while True:
            try:
                commands.append(self._queue.get_nowait())
            except Empty:
                return commands
