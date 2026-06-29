"""Controller package exports for the CC-MPC core.

Import policy
-------------
This package is being migrated incrementally.  Some controller modules may not
exist yet, or may exist without their final public class.  Package import should
not break unit tests for already-migrated modules.

Examples that should work during staged refactor:
    from ccmpc.controllers.fallback_controller import FallbackController
    from ccmpc.controllers.ccmpc_controller import CCMPCController

The optional imports below intentionally catch ImportError, not only
ModuleNotFoundError, because a staged module can exist but not yet export the
final symbol.
"""

from __future__ import annotations

from typing import Any


try:
    from ccmpc.controllers.ccmpc_controller import (
        CCMPC,
        CCMPCController,
        CCMPCSolveInfo,
    )
except ImportError:  # pragma: no cover - allowed during staged migration.
    CCMPC = None  # type: ignore[assignment]
    CCMPCController = None  # type: ignore[assignment]
    CCMPCSolveInfo = None  # type: ignore[assignment]


try:
    from ccmpc.controllers.fallback_controller import (
        FallbackConfig,
        FallbackController,
        FallbackMode,
        FallbackResult,
        FallbackStatus,
    )
except ImportError:  # pragma: no cover - allowed during staged migration.
    FallbackConfig = None  # type: ignore[assignment]
    FallbackController = None  # type: ignore[assignment]
    FallbackMode = None  # type: ignore[assignment]
    FallbackResult = None  # type: ignore[assignment]
    FallbackStatus = None  # type: ignore[assignment]


try:
    from ccmpc.controllers.solver_adapter import SolverAdapter
except ImportError:  # pragma: no cover - solver adapter not migrated yet.
    SolverAdapter = None  # type: ignore[assignment]


__all__ = [
    "CCMPC",
    "CCMPCController",
    "CCMPCSolveInfo",
    "FallbackConfig",
    "FallbackController",
    "FallbackMode",
    "FallbackResult",
    "FallbackStatus",
    "SolverAdapter",
]
