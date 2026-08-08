"""Validation mode canonicalization for the native Monte Carlo protocol.

The legacy ``joint`` validation mode is an alias for the canonical
``joint_uniform`` mode.  Canonicalization happens once at the
CLI/config/application boundary; internal code only ever sees canonical
modes.  ``joint_adaptive`` is deliberately rejected until the adaptive
allocator actually exists.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

CANONICAL_MODES = (
    "deterministic",
    "individual",
    "joint_uniform",
)

MODE_ALIASES = {
    "joint": "joint_uniform",
}

_DEPRECATION_MESSAGE = (
    "validation mode 'joint' is deprecated; the canonical mode is now "
    "'joint_uniform'.  'joint' will keep working as an alias."
)


@dataclass(frozen=True, slots=True)
class CanonicalMode:
    """Result of canonicalizing one requested validation mode."""

    requested_mode: str
    canonical_mode: str
    legacy_alias_used: bool


def canonicalize_mode(raw_mode: str) -> CanonicalMode:
    """Canonicalize a requested mode, rejecting unknown and future modes."""
    requested = str(raw_mode).strip().lower()
    if not requested:
        raise ValueError("validation mode must not be empty")
    canonical = MODE_ALIASES.get(requested, requested)
    if canonical not in CANONICAL_MODES:
        raise ValueError(
            f"unsupported native Monte Carlo mode: {raw_mode!r}; "
            f"supported modes are {', '.join(CANONICAL_MODES)}"
        )
    return CanonicalMode(
        requested_mode=requested,
        canonical_mode=canonical,
        legacy_alias_used=requested != canonical,
    )


def canonicalize_modes(
    modes: list[str] | tuple[str, ...],
    *,
    warn: bool = True,
) -> tuple[CanonicalMode, ...]:
    """Canonicalize a mode list, emitting one deprecation warning per alias."""
    results = tuple(canonicalize_mode(mode) for mode in modes)
    aliases = [result for result in results if result.legacy_alias_used]
    if warn and aliases:
        warnings.warn(
            _DEPRECATION_MESSAGE,
            DeprecationWarning,
            stacklevel=2,
        )
    return results


def canonical_mode_names(modes: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return canonical mode names for a requested mode list."""
    return tuple(result.canonical_mode for result in canonicalize_modes(modes))
