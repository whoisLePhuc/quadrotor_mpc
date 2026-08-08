"""Pure presentation model for the native CC-MPC desktop interface.

The Qt process must not reinterpret safety semantics.  This module converts
validated runtime telemetry and the active controller policy into immutable
view data that can be unit-tested without PySide6 or a display server.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

OK = "ok"
INFO = "info"
WARNING = "warning"
DANGER = "danger"
MUTED = "muted"


def _finite_nonnegative(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and >= 0")
    return number


@dataclass(frozen=True, slots=True)
class PanelRuntimeContext:
    """Controller policy required to interpret one telemetry sample."""

    scenario_name: str
    mpc_period_ms: float
    estimation_enabled: bool
    chance_constraints_enabled: bool
    covariance_propagation_enabled: bool
    supervisor_enabled: bool
    solve_deadline_ms: float
    guarantee_slack_tolerance_m: float
    maximum_acceptable_slack_m: float
    maximum_solver_residual: float
    configured_risk_semantics: str
    configured_risk_allocation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_name", str(self.scenario_name))
        for field in (
            "mpc_period_ms",
            "solve_deadline_ms",
            "guarantee_slack_tolerance_m",
            "maximum_acceptable_slack_m",
            "maximum_solver_residual",
        ):
            object.__setattr__(
                self,
                field,
                _finite_nonnegative(getattr(self, field), f"panel context {field}"),
            )
        if self.mpc_period_ms <= 0.0:
            raise ValueError("panel context mpc_period_ms must be > 0")
        if self.guarantee_slack_tolerance_m > self.maximum_acceptable_slack_m:
            raise ValueError("panel guarantee slack tolerance must not exceed the acceptance limit")
        object.__setattr__(
            self,
            "configured_risk_semantics",
            str(self.configured_risk_semantics),
        )
        object.__setattr__(
            self,
            "configured_risk_allocation",
            str(self.configured_risk_allocation),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PanelRuntimeContext:
        return cls(
            scenario_name=raw.get("scenario_name", "native-mujoco"),
            mpc_period_ms=raw.get("mpc_period_ms", 50.0),
            estimation_enabled=bool(raw.get("estimation_enabled", False)),
            chance_constraints_enabled=bool(raw.get("chance_constraints_enabled", False)),
            covariance_propagation_enabled=bool(raw.get("covariance_propagation_enabled", False)),
            supervisor_enabled=bool(raw.get("supervisor_enabled", False)),
            solve_deadline_ms=raw.get("solve_deadline_ms", 0.0),
            guarantee_slack_tolerance_m=raw.get(
                "guarantee_slack_tolerance_m",
                0.0,
            ),
            maximum_acceptable_slack_m=raw.get(
                "maximum_acceptable_slack_m",
                0.0,
            ),
            maximum_solver_residual=raw.get("maximum_solver_residual", 0.0),
            configured_risk_semantics=raw.get(
                "configured_risk_semantics",
                "disabled",
            ),
            configured_risk_allocation=raw.get(
                "configured_risk_allocation",
                "none",
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @property
    def mode_label(self) -> str:
        controller = "CC-MPC" if self.chance_constraints_enabled else "deterministic NMPC"
        estimator = "belief/ESEKF" if self.estimation_enabled else "ground truth"
        supervisor = "supervised" if self.supervisor_enabled else "unsupervised"
        return f"{controller} · {estimator} · {supervisor}"


@dataclass(frozen=True, slots=True)
class StatusCard:
    key: str
    title: str
    value: str
    detail: str
    tone: str


@dataclass(frozen=True, slots=True)
class PanelViewState:
    """All safety-critical labels rendered by the desktop panel."""

    runtime_state: str
    runtime_tone: str
    banner: str
    cards: tuple[StatusCard, ...]
    fallback_level: int
    deadline_missed: bool
    solution_accepted: bool
    safety_assurance_status: str
    horizon_assurance_eligible: bool
    risk_budget_status: str
    collided: bool
    completed: bool
    time_s: float

    def card(self, key: str) -> StatusCard:
        for card in self.cards:
            if card.key == key:
                return card
        raise KeyError(key)


@dataclass(frozen=True, slots=True)
class PanelAlert:
    key: str
    time_s: float
    tone: str
    message: str


def _optional_float(sample: Mapping[str, Any], key: str) -> float | None:
    value = sample.get(key)
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _runtime_card(sample: Mapping[str, Any]) -> tuple[str, str, StatusCard]:
    collided = bool(sample.get("collided", False))
    completed = bool(sample.get("completed", False))
    paused = bool(sample.get("paused", False))
    if collided:
        state, tone = "COLLISION", DANGER
    elif completed:
        state, tone = "COMPLETED", INFO
    elif paused:
        state, tone = "PAUSED", WARNING
    else:
        state, tone = "RUNNING", OK
    detail = (
        f"t={float(sample.get('time_s', 0.0)):.2f}s · "
        f"goal={float(sample.get('goal_distance_m', 0.0)):.3f}m · "
        f"clearance={float(sample.get('min_clearance_m', 0.0)):.3f}m"
    )
    return state, tone, StatusCard("runtime", "Episode", state, detail, tone)


def _controller_card(sample: Mapping[str, Any]) -> StatusCard:
    fallback_active = bool(sample.get("fallback_active", False))
    accepted = bool(sample.get("solution_accepted", True))
    source = str(sample.get("command_source", "PRIMARY_NMPC"))
    status = str(sample.get("primary_solver_status", sample.get("solver_status", "")))
    iterations = int(sample.get("primary_solver_iterations", 0))
    raw_primal = sample.get("primary_solver_primal_residual")
    raw_dual = sample.get("primary_solver_dual_residual")
    available_residuals = [
        value
        for value in (raw_primal, raw_dual)
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    residual_text = f"{max(available_residuals):.2e}" if available_residuals else "n/a"
    if fallback_active:
        level = int(sample.get("fallback_level", 0))
        value = f"FALLBACK L{level}"
        tone = DANGER if level >= 3 else WARNING
    elif accepted:
        value, tone = "PRIMARY NMPC", OK
    else:
        value, tone = "REJECTED", DANGER
    detail = f"{source} · {status or 'unknown'} · iter={iterations} · res={residual_text}"
    return StatusCard("controller", "Applied control", value, detail, tone)


def _assurance_card(sample: Mapping[str, Any], context: PanelRuntimeContext) -> StatusCard:
    status = str(sample.get("horizon_assurance_status", ""))
    reason = str(sample.get("horizon_assurance_reason", ""))
    eligible = bool(sample.get("horizon_assurance_eligible", False))
    if not context.chance_constraints_enabled:
        return StatusCard(
            "assurance",
            "Safety assurance",
            "DETERMINISTIC",
            "Chance constraints are disabled",
            MUTED,
        )
    if eligible:
        return StatusCard(
            "assurance",
            "Safety assurance",
            "HORIZON ELIGIBLE",
            "Joint chance constraints satisfied on the current prediction horizon",
            OK,
        )
    if status == "NOT_GUARANTEED_POSITIVE_SLACK" or reason == "positive_slack":
        return StatusCard(
            "assurance",
            "Safety assurance",
            "NOT GUARANTEED",
            "A degraded primary solution uses positive slack",
            WARNING,
        )
    if status == "NOT_GUARANTEED_FALLBACK_ACTIVE" or reason == "fallback_active":
        return StatusCard(
            "assurance",
            "Safety assurance",
            "NOT GUARANTEED",
            "Fallback control is active",
            DANGER,
        )
    return StatusCard(
        "assurance",
        "Safety assurance",
        _assurance_short_label(status, reason),
        "No joint-horizon assurance on the current prediction horizon",
        MUTED,
    )


def _assurance_short_label(status: str, reason: str) -> str:
    if status:
        return status.replace("NOT_GUARANTEED_", "NOT GUARANTEED ").replace("_", " ")
    return reason or "UNAVAILABLE"


def _risk_card(sample: Mapping[str, Any], context: PanelRuntimeContext) -> StatusCard:
    semantics = str(sample.get("risk_semantics", context.configured_risk_semantics))
    allocation = str(sample.get("risk_allocation_method", context.configured_risk_allocation))
    status = str(sample.get("risk_budget_status", ""))
    total = _optional_float(sample, "risk_budget_total")
    allocated = float(sample.get("risk_budget_allocated", 0.0))
    remaining = _optional_float(sample, "risk_budget_remaining")
    count = int(sample.get("risk_constraint_count", 0))
    if not context.chance_constraints_enabled or semantics == "disabled":
        return StatusCard("risk", "Risk budget", "DISABLED", "No chance constraints", MUTED)
    if total is None:
        return StatusCard(
            "risk",
            "Risk budget",
            "INDIVIDUAL",
            f"ε per constraint · {allocation} · {count} constraints",
            INFO,
        )
    tone = OK if status == "BUDGET_OK" and allocated <= total + 1e-12 else DANGER
    remaining_text = "—" if remaining is None else f"{remaining:.3e}"
    return StatusCard(
        "risk",
        "Joint risk budget",
        status or "UNKNOWN",
        (
            f"allocated={allocated:.6f}/{total:.6f} · "
            f"remaining={remaining_text} · {count} constraints"
        ),
        tone,
    )


def _slack_card(sample: Mapping[str, Any], context: PanelRuntimeContext) -> StatusCard:
    slack = _optional_float(sample, "maximum_slack_m")
    residual = _optional_float(sample, "minimum_chance_residual_m")
    if not context.chance_constraints_enabled or slack is None:
        return StatusCard("slack", "Chance constraint", "DISABLED", "No slack", MUTED)
    if slack <= context.guarantee_slack_tolerance_m:
        value, tone = "HARD-SAFE", OK
    elif slack <= context.maximum_acceptable_slack_m:
        value, tone = "DEGRADED", WARNING
    else:
        value, tone = "REJECT LIMIT", DANGER
    residual_text = "—" if residual is None else f"{residual:.4f}m"
    return StatusCard(
        "slack",
        "Chance constraint",
        value,
        (
            f"max slack={slack:.6f}m · min residual={residual_text} · "
            f"accept≤{context.maximum_acceptable_slack_m:.3f}m"
        ),
        tone,
    )


def _deadline_card(sample: Mapping[str, Any], context: PanelRuntimeContext) -> StatusCard:
    solve_ms = float(sample.get("solver_time_ms", 0.0))
    missed = bool(sample.get("deadline_missed", False))
    if not context.supervisor_enabled or context.solve_deadline_ms <= 0.0:
        tone = WARNING if solve_ms > context.mpc_period_ms else INFO
        return StatusCard(
            "deadline",
            "Solver timing",
            "MONITOR ONLY",
            f"{solve_ms:.1f}ms · MPC period={context.mpc_period_ms:.1f}ms",
            tone,
        )
    value, tone = ("DEADLINE MISSED", DANGER) if missed else ("ON TIME", OK)
    return StatusCard(
        "deadline",
        "Solver timing",
        value,
        (
            f"{solve_ms:.1f}ms / deadline={context.solve_deadline_ms:.1f}ms · "
            f"MPC period={context.mpc_period_ms:.1f}ms"
        ),
        tone,
    )


def build_panel_view(
    sample: Mapping[str, Any],
    context: PanelRuntimeContext,
) -> PanelViewState:
    """Convert one native telemetry sample into UI-ready safety semantics."""

    runtime_state, runtime_tone, runtime_card = _runtime_card(sample)
    cards = (
        runtime_card,
        _controller_card(sample),
        _assurance_card(sample, context),
        _risk_card(sample, context),
        _slack_card(sample, context),
        _deadline_card(sample, context),
    )
    card_summary = (
        f"{cards[1].value} · {cards[2].value} · {cards[3].value} · "
        f"{cards[4].value} · {cards[5].value}"
    )
    return PanelViewState(
        runtime_state=runtime_state,
        runtime_tone=runtime_tone,
        banner=f"{runtime_state} · {card_summary}",
        cards=cards,
        fallback_level=int(sample.get("fallback_level", 0)),
        deadline_missed=bool(sample.get("deadline_missed", False)),
        solution_accepted=bool(sample.get("solution_accepted", True)),
        safety_assurance_status=str(sample.get("safety_assurance_status", "")),
        horizon_assurance_eligible=bool(sample.get("horizon_assurance_eligible", False)),
        risk_budget_status=str(sample.get("risk_budget_status", "")),
        collided=bool(sample.get("collided", False)),
        completed=bool(sample.get("completed", False)),
        time_s=float(sample.get("time_s", 0.0)),
    )


def panel_transition_alerts(
    previous: PanelViewState | None,
    current: PanelViewState,
) -> tuple[PanelAlert, ...]:
    """Return deduplicated alerts for operationally relevant state changes."""

    alerts: list[PanelAlert] = []
    if current.collided and (previous is None or not previous.collided):
        alerts.append(PanelAlert("collision", current.time_s, DANGER, "Collision detected"))
    previous_level = 0 if previous is None else previous.fallback_level
    if current.fallback_level != previous_level:
        if current.fallback_level > 0:
            alerts.append(
                PanelAlert(
                    "fallback_entered",
                    current.time_s,
                    DANGER if current.fallback_level >= 3 else WARNING,
                    f"Fallback entered at level {current.fallback_level}",
                )
            )
        else:
            alerts.append(
                PanelAlert("fallback_recovered", current.time_s, OK, "Primary NMPC recovered")
            )
    if current.deadline_missed and (previous is None or not previous.deadline_missed):
        alerts.append(
            PanelAlert("deadline_missed", current.time_s, DANGER, "Solver deadline missed")
        )
    if current.risk_budget_status == "BUDGET_EXCEEDED" and (
        previous is None or previous.risk_budget_status != current.risk_budget_status
    ):
        alerts.append(
            PanelAlert("risk_budget", current.time_s, DANGER, "Joint risk budget exceeded")
        )
    if previous is not None and (
        previous.horizon_assurance_eligible and not current.horizon_assurance_eligible
    ):
        alerts.append(
            PanelAlert(
                "assurance_lost",
                current.time_s,
                WARNING,
                f"Horizon eligibility lost: {current.safety_assurance_status or 'unavailable'}",
            )
        )
    if current.completed and (previous is None or not previous.completed):
        alerts.append(PanelAlert("completed", current.time_s, INFO, "Episode completed"))
    return tuple(alerts)
