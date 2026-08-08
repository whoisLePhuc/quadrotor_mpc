"""Immutable chance-profile snapshot with provenance.

The distinction this module enforces:

- an *enforced* chance profile is a frozen snapshot of the exact risk and
  tightening values that were loaded into the NMPC solver parameters/TVPs for
  one solve attempt;
- a *post-solve diagnostic* profile is an optional re-evaluation built on the
  optimized trajectory after the solve and must never be reported as enforced.

A profile is only called enforced when the system has evidence it was applied
to the solver before the solve attempt started.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

ENFORCED_PHASE = "ENFORCED"
POST_SOLVE_DIAGNOSTIC_PHASE = "POST_SOLVE_DIAGNOSTIC"
CHANCE_PROFILE_SCHEMA_VERSION = 1


class ChanceProfileShapeError(ValueError):
    """Raised when profile/trajectory shapes cannot be loaded into the solver."""


@dataclass(frozen=True, slots=True)
class ChanceProfileProvenance:
    """Identity and source metadata for one chance profile snapshot."""

    control_tick: int
    solve_attempt_id: str
    profile_id: str
    phase: str
    trajectory_source: str
    allocation_method: str
    risk_semantics: str
    horizon_nodes: int
    obstacle_count: int
    config_hash: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "control_tick": self.control_tick,
            "solve_attempt_id": self.solve_attempt_id,
            "profile_id": self.profile_id,
            "phase": self.phase,
            "trajectory_source": self.trajectory_source,
            "allocation_method": self.allocation_method,
            "risk_semantics": self.risk_semantics,
            "horizon_nodes": self.horizon_nodes,
            "obstacle_count": self.obstacle_count,
            "config_hash": self.config_hash,
        }


def new_profile_id(payload: dict[str, Any]) -> str:
    """Deterministic digest over a canonical profile payload.

    Python's builtin ``hash()`` is process-instance specific and must not be
    used for artifact identity.
    """
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def new_solve_attempt_id() -> str:
    """Return a fresh, process-stable solve attempt identifier."""
    return uuid.uuid4().hex[:16]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _readonly_2d(array: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(array, dtype=float)
    if result.ndim != 2:
        raise ChanceProfileShapeError(f"{label} must be 2-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ChanceProfile:
    """Immutable snapshot of chance-constraint parameters for one solve."""

    provenance: ChanceProfileProvenance
    node_indices: tuple[int, ...]
    obstacle_ids: tuple[str, ...]
    epsilon: np.ndarray
    projected_sigma: np.ndarray
    tightening_margin: np.ndarray
    tightened_radius: np.ndarray
    active_mask: np.ndarray
    joint_budget: float | None
    allocated_budget: float | None
    remaining_budget: float | None = None

    def __post_init__(self) -> None:
        steps = len(self.node_indices)
        obstacles = len(self.obstacle_ids)
        expected = (steps, obstacles)
        epsilon = _readonly_2d(self.epsilon, "ChanceProfile.epsilon")
        sigma = _readonly_2d(self.projected_sigma, "ChanceProfile.projected_sigma")
        margin = _readonly_2d(self.tightening_margin, "ChanceProfile.tightening_margin")
        radius = _readonly_2d(self.tightened_radius, "ChanceProfile.tightened_radius")
        active = np.asarray(self.active_mask, dtype=bool)
        if active.shape != expected:
            raise ChanceProfileShapeError(
                f"active_mask must have shape {expected}, got {active.shape}"
            )
        for label, array in (
            ("epsilon", epsilon),
            ("projected_sigma", sigma),
            ("tightening_margin", margin),
            ("tightened_radius", radius),
        ):
            if array.shape != expected:
                raise ChanceProfileShapeError(
                    f"{label} must have shape {expected}, got {array.shape}"
                )
        if np.any(epsilon[active] <= 0.0) or np.any(epsilon[active] >= 1.0):
            raise ValueError("active epsilon values must lie in (0, 1)")
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "projected_sigma", sigma)
        object.__setattr__(self, "tightening_margin", margin)
        object.__setattr__(self, "tightened_radius", radius)
        copied_active = active.copy()
        copied_active.setflags(write=False)
        object.__setattr__(self, "active_mask", copied_active)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_mapping(),
            "node_indices": list(self.node_indices),
            "obstacle_ids": list(self.obstacle_ids),
            "epsilon": self.epsilon.tolist(),
            "projected_sigma": self.projected_sigma.tolist(),
            "tightening_margin": self.tightening_margin.tolist(),
            "tightened_radius": self.tightened_radius.tolist(),
            "active_mask": self.active_mask.tolist(),
            "joint_budget": self.joint_budget,
            "allocated_budget": self.allocated_budget,
            "remaining_budget": self.remaining_budget,
        }

    def canonical_payload(self) -> dict[str, Any]:
        """Payload used for deterministic identity.

        Phase and trajectory source are part of the identity so an enforced
        profile and a post-solve diagnostic never collide on the same id.
        """
        payload = self.to_mapping()
        payload["phase"] = self.provenance.phase
        payload["trajectory_source"] = self.provenance.trajectory_source
        payload["solve_attempt_id"] = self.provenance.solve_attempt_id
        payload.pop("provenance", None)
        return payload


def build_chance_profile_snapshot(
    *,
    control_tick: int,
    solve_attempt_id: str,
    phase: str,
    trajectory_source: str,
    allocation_method: str,
    risk_semantics: str,
    node_indices: tuple[int, ...],
    obstacle_ids: tuple[str, ...],
    epsilon: np.ndarray,
    projected_sigma: np.ndarray,
    tightening_margin: np.ndarray,
    tightened_radius: np.ndarray,
    active_mask: np.ndarray,
    joint_budget: float | None,
    allocated_budget: float | None,
    remaining_budget: float | None,
    config_hash: str | None = None,
) -> ChanceProfile:
    """Build a frozen :class:`ChanceProfile` with derived identity.

    The profile id is a deterministic digest of the canonical payload so the
    same values always produce the same id across processes and ticks.
    """
    provenance = ChanceProfileProvenance(
        control_tick=control_tick,
        solve_attempt_id=solve_attempt_id,
        profile_id="",
        phase=phase,
        trajectory_source=trajectory_source,
        allocation_method=allocation_method,
        risk_semantics=risk_semantics,
        horizon_nodes=len(node_indices),
        obstacle_count=len(obstacle_ids),
        config_hash=config_hash,
    )
    profile = ChanceProfile(
        provenance=provenance,
        node_indices=node_indices,
        obstacle_ids=obstacle_ids,
        epsilon=epsilon,
        projected_sigma=projected_sigma,
        tightening_margin=tightening_margin,
        tightened_radius=tightened_radius,
        active_mask=active_mask,
        joint_budget=joint_budget,
        allocated_budget=allocated_budget,
        remaining_budget=remaining_budget,
    )
    payload = profile.canonical_payload()
    profile_id = new_profile_id(payload)
    final_provenance = ChanceProfileProvenance(
        control_tick=control_tick,
        solve_attempt_id=solve_attempt_id,
        profile_id=profile_id,
        phase=phase,
        trajectory_source=trajectory_source,
        allocation_method=allocation_method,
        risk_semantics=risk_semantics,
        horizon_nodes=len(node_indices),
        obstacle_count=len(obstacle_ids),
        config_hash=config_hash,
    )
    return ChanceProfile(
        provenance=final_provenance,
        node_indices=node_indices,
        obstacle_ids=obstacle_ids,
        epsilon=profile.epsilon,
        projected_sigma=profile.projected_sigma,
        tightening_margin=profile.tightening_margin,
        tightened_radius=profile.tightened_radius,
        active_mask=profile.active_mask,
        joint_budget=joint_budget,
        allocated_budget=allocated_budget,
        remaining_budget=remaining_budget,
    )
