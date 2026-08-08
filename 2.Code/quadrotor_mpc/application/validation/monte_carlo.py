"""Paired Monte Carlo validation for the native MuJoCo CC-MPC pipeline.

Stage 8 deliberately validates the same 13-state MuJoCo plant, ESEKF,
chance-constraint, risk-budget and safety-supervisor path used by the native
desktop application.  It does not reuse the lightweight 9-state ODE benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import yaml

from quadrotor_mpc.interfaces.desktop.viewer import NativeMuJoCoConfig

SUPPORTED_MODES = ("deterministic", "individual", "joint")
_SENSOR_STD_FIELDS = (
    "position_std_m",
    "velocity_std_mps",
    "attitude_std_rad",
    "angular_rate_std_radps",
    "obstacle_position_std_m",
    "position_bias_rw_std_m_sqrt_s",
    "velocity_bias_rw_std_mps_sqrt_s",
    "attitude_bias_rw_std_rad_sqrt_s",
    "angular_rate_bias_rw_std_radps_sqrt_s",
    "obstacle_bias_rw_std_m_sqrt_s",
)
_VEHICLE_FILTER_STD_FIELDS = (
    "acceleration_process_std_mps2",
    "angular_acceleration_process_std_radps2",
    "initial_position_std_m",
    "initial_velocity_std_mps",
    "initial_attitude_std_rad",
    "initial_angular_rate_std_radps",
)
_OBSTACLE_FILTER_STD_FIELDS = (
    "acceleration_process_std_mps2",
    "initial_position_std_m",
    "initial_velocity_std_mps",
)
_PROPAGATION_STD_FIELDS = (
    "acceleration_process_std_mps2",
    "angular_acceleration_process_std_radps2",
    "obstacle_acceleration_process_std_mps2",
)


def _finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and > 0")
    return number


def _probability(value: Any, label: str) -> float:
    number = float(value)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{label} must be in (0, 1)")
    return number


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class NoiseLevel:
    """Named multiplier applied to covariance, not directly to standard deviation."""

    label: str
    covariance_scale: float

    def __post_init__(self) -> None:
        label = str(self.label).strip()
        if not label or any(character in label for character in "/\\"):
            raise ValueError("noise-level label must be non-empty and path-safe")
        object.__setattr__(self, "label", label)
        object.__setattr__(
            self,
            "covariance_scale",
            _finite_positive(self.covariance_scale, f"noise_levels.{label}"),
        )


@dataclass(frozen=True, slots=True)
class NativeMonteCarloProtocol:
    """Validated Stage 8 experiment and claim-gate configuration."""

    name: str
    base_config_path: Path
    output_dir: Path
    modes: tuple[str, ...]
    noise_levels: tuple[NoiseLevel, ...]
    trials: int
    first_seed: int
    confidence_level: float
    minimum_trials_for_claim: int
    empirical_collision_rate_limit: float
    require_zero_positive_slack: bool
    require_zero_fallback: bool
    require_zero_budget_failures: bool
    timing_percentile: float
    protocol_type: str = "algorithmic_comparison"
    control_period_ms: float = 50.0
    deadline_clock: str = "solver_only"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if self.trials < 1:
            raise ValueError("trials must be >= 1")
        if self.minimum_trials_for_claim < 1:
            raise ValueError("minimum_trials_for_claim must be >= 1")
        if self.first_seed < 0:
            raise ValueError("first_seed must be >= 0")
        _probability(self.confidence_level, "statistics.confidence_level")
        _probability(
            self.empirical_collision_rate_limit,
            "validation.empirical_collision_rate_limit",
        )
        _probability(self.timing_percentile, "validation.timing_percentile")
        if not any(
            math.isclose(self.timing_percentile, supported)
            for supported in (0.95, 0.99)
        ):
            raise ValueError("validation.timing_percentile must be 0.95 or 0.99")
        if not self.modes:
            raise ValueError("modes must contain at least one controller")
        unsupported = set(self.modes) - set(SUPPORTED_MODES)
        if unsupported:
            raise ValueError(f"unsupported native Monte Carlo modes: {sorted(unsupported)}")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must not contain duplicates")
        if not self.noise_levels:
            raise ValueError("noise_levels must contain at least one level")
        labels = [level.label for level in self.noise_levels]
        if len(set(labels)) != len(labels):
            raise ValueError("noise-level labels must be unique")
        self._validate_protocol()

    def _validate_protocol(self) -> None:
        from quadrotor_mpc.application.validation.protocol import (
            DeadlineClock,
            MonteCarloProtocol,
            validate_control_period,
            validate_protocol_policy,
        )

        protocol = MonteCarloProtocol(self.protocol_type)
        derived_policy = deadline_policy_from_protocol(protocol)
        validate_protocol_policy(protocol, derived_policy)
        validate_control_period(self.control_period_ms)
        if self.deadline_clock not in {clock.value for clock in DeadlineClock}:
            raise ValueError(
                f"validation.monte_carlo.deadline_clock must be one of "
                f"{[clock.value for clock in DeadlineClock]}"
            )

    @property
    def protocol(self):
        from quadrotor_mpc.application.validation.protocol import (
            MonteCarloProtocol,
        )

        return MonteCarloProtocol(self.protocol_type)

    @property
    def deadline_policy(self) -> str:
        return deadline_policy_from_protocol(self.protocol).value

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.first_seed, self.first_seed + self.trials))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_config": str(self.base_config_path),
            "output_dir": str(self.output_dir),
            "modes": list(self.modes),
            "noise_levels": [
                {
                    "label": level.label,
                    "covariance_scale": level.covariance_scale,
                }
                for level in self.noise_levels
            ],
            "trials": self.trials,
            "first_seed": self.first_seed,
            "statistics": {
                "confidence_level": self.confidence_level,
                "minimum_trials_for_claim": self.minimum_trials_for_claim,
            },
            "validation": {
                "empirical_collision_rate_limit": (
                    self.empirical_collision_rate_limit
                ),
                "require_zero_positive_slack": self.require_zero_positive_slack,
                "require_zero_fallback": self.require_zero_fallback,
                "require_zero_budget_failures": self.require_zero_budget_failures,
                "timing_percentile": self.timing_percentile,
            },
            "monte_carlo": {
                "protocol": self.protocol_type,
                "control_period_ms": self.control_period_ms,
                "deadline_clock": self.deadline_clock,
            },
        }


def load_native_monte_carlo_protocol(
    path: str | Path,
) -> NativeMonteCarloProtocol:
    """Load a Stage 8 YAML protocol and resolve its paths relative to the file."""
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "native Monte Carlo protocol")
    statistics = _mapping(root.get("statistics", {}), "statistics")
    validation = _mapping(root.get("validation", {}), "validation")
    raw_levels = root.get("noise_levels", [])
    if not isinstance(raw_levels, Sequence) or isinstance(raw_levels, (str, bytes)):
        raise TypeError("noise_levels must be a list")
    levels = tuple(
        NoiseLevel(
            label=str(_mapping(item, f"noise_levels[{index}]")["label"]),
            covariance_scale=float(
                _mapping(item, f"noise_levels[{index}]")["covariance_scale"]
            ),
        )
        for index, item in enumerate(raw_levels)
    )
    raw_modes = root.get("modes", SUPPORTED_MODES)
    if not isinstance(raw_modes, Sequence) or isinstance(raw_modes, (str, bytes)):
        raise TypeError("modes must be a list")

    def resolve_input(value: Any) -> Path:
        candidate = Path(str(value)).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (source.parent / candidate).resolve()

    def resolve_output(value: Any) -> Path:
        candidate = Path(str(value)).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (Path.cwd() / candidate).resolve()

    return NativeMonteCarloProtocol(
        name=str(root.get("name", "native-monte-carlo")),
        base_config_path=resolve_input(
            root.get("base_config", "mujoco_native_ccmpc.yaml")
        ),
        output_dir=resolve_output(
            root.get("output_dir", "outputs/native_monte_carlo")
        ),
        modes=tuple(str(mode).lower() for mode in raw_modes),
        noise_levels=levels,
        trials=int(root.get("trials", 50)),
        first_seed=int(root.get("first_seed", 1000)),
        confidence_level=float(statistics.get("confidence_level", 0.95)),
        minimum_trials_for_claim=int(
            statistics.get("minimum_trials_for_claim", 30)
        ),
        empirical_collision_rate_limit=float(
            validation.get("empirical_collision_rate_limit", 0.10)
        ),
        require_zero_positive_slack=bool(
            validation.get("require_zero_positive_slack", True)
        ),
        require_zero_fallback=bool(
            validation.get("require_zero_fallback", True)
        ),
        require_zero_budget_failures=bool(
            validation.get("require_zero_budget_failures", True)
        ),
        timing_percentile=float(validation.get("timing_percentile", 0.99)),
        protocol_type=str(
            _mapping(root.get("monte_carlo", {}), "monte_carlo").get(
                "protocol", "algorithmic_comparison"
            )
        ),
        control_period_ms=float(
            _mapping(root.get("monte_carlo", {}), "monte_carlo").get(
                "control_period_ms", 50.0
            )
        ),
        deadline_clock=str(
            _mapping(root.get("monte_carlo", {}), "monte_carlo").get(
                "deadline_clock", "solver_only"
            )
        ),
    )


def deadline_policy_from_protocol(protocol):
    from quadrotor_mpc.application.validation.protocol import (
        DeadlinePolicy,
    )

    if protocol.value == "algorithmic_comparison":
        return DeadlinePolicy.RECORD_ONLY
    if protocol.value == "realtime_qualification":
        return DeadlinePolicy.REJECT_TO_FALLBACK
    raise ValueError(f"Unsupported protocol: {protocol}")


def _scale_fields(
    mapping: dict[str, Any],
    fields: Iterable[str],
    standard_deviation_scale: float,
) -> None:
    for field in fields:
        if field in mapping:
            mapping[field] = float(mapping[field]) * standard_deviation_scale


def effective_native_config(
    base: NativeMuJoCoConfig,
    *,
    mode: str,
    covariance_scale: float,
    seed: int,
    protocol_type: str = "algorithmic_comparison",
) -> NativeMuJoCoConfig:
    """Build one paired-trial config without mutating the source configuration."""
    normalized_mode = str(mode).lower()
    if normalized_mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported native Monte Carlo mode: {mode}")
    covariance_scale = _finite_positive(covariance_scale, "covariance_scale")
    mapping = base.to_mapping()
    mapping["name"] = f"{base.name}-mc-{normalized_mode}-cov{covariance_scale:g}-seed{seed}"
    mapping["estimation"]["seed"] = int(seed)

    standard_deviation_scale = math.sqrt(covariance_scale)
    _scale_fields(
        mapping["estimation"]["sensor"],
        _SENSOR_STD_FIELDS,
        standard_deviation_scale,
    )
    _scale_fields(
        mapping["estimation"]["vehicle_filter"],
        _VEHICLE_FILTER_STD_FIELDS,
        standard_deviation_scale,
    )
    _scale_fields(
        mapping["estimation"]["obstacle_filter"],
        _OBSTACLE_FILTER_STD_FIELDS,
        standard_deviation_scale,
    )
    propagation = mapping["controller"]["covariance_propagation"]
    _scale_fields(
        propagation,
        _PROPAGATION_STD_FIELDS,
        standard_deviation_scale,
    )

    chance = mapping["controller"]["chance_constraints"]
    # Deadline handling is protocol-derived: algorithmic comparison records
    # deadline facts but applies late-but-valid primary commands so allocator
    # quality is not hidden by deadline rejection; realtime qualification
    # rejects late commands to fallback as the strict policy requires.
    reject_on_deadline = (
        protocol_type == "realtime_qualification"
    )
    mapping["controller"]["safety_fallback"]["reject_on_deadline_miss"] = (
        reject_on_deadline
    )
    if normalized_mode == "deterministic":
        chance["enabled"] = False
        propagation["enabled"] = False
    else:
        chance["enabled"] = True
        propagation["enabled"] = True
        chance["risk_budget"]["semantics"] = (
            "individual" if normalized_mode == "individual" else "joint"
        )
    return NativeMuJoCoConfig.from_mapping(mapping)


def _build_controller(config: NativeMuJoCoConfig):
    from quadrotor_mpc.control.nmpc.chance_constrained import (
        SphericalChanceConstrainedNMPCController,
    )
    from quadrotor_mpc.control.nmpc.deterministic import DeterministicNMPCController
    from quadrotor_mpc.control.nmpc.safety import SafeFallbackController

    controller_type = (
        SphericalChanceConstrainedNMPCController
        if config.chance_constraints.enabled
        else DeterministicNMPCController
    )
    controller = controller_type(
        bounds=config.bounds,
        obstacle_specs=config.obstacles,
        margin=config.safety_margin,
        horizon_steps=config.horizon_steps,
        timestep_s=config.mpc_timestep_s,
        max_iter=config.max_solver_iterations,
        covariance_options=config.covariance_propagation,
        chance_options=config.chance_constraints,
    )
    if config.safety_fallback.enabled:
        controller = SafeFallbackController(
            controller,
            options=config.safety_fallback,
            bounds=config.bounds,
        )
    return controller


def _percentile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if finite.size else 0.0


def _line_tracking_rmse(
    positions: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
) -> float:
    direction = goal - start
    denominator = float(direction @ direction)
    if denominator < 1e-12:
        errors = np.linalg.norm(positions - start, axis=1)
    else:
        alpha = np.clip((positions - start) @ direction / denominator, 0.0, 1.0)
        projection = start + alpha[:, None] * direction
        errors = np.linalg.norm(positions - projection, axis=1)
    return float(np.sqrt(np.mean(errors**2))) if errors.size else 0.0


@dataclass(frozen=True, slots=True)
class NativeTrialResult:
    """One scalar row in the Stage 8 paired Monte Carlo table."""

    noise_label: str
    covariance_scale: float
    mode: str
    seed: int
    expected_steps: int
    completed_steps: int
    completed: bool
    termination_reason: str
    success: bool
    collision: bool
    numerical_failure: bool
    final_error_m: float
    min_clearance_m: float
    path_length_m: float
    tracking_rmse_m: float
    control_effort: float
    estimation_position_rmse_m: float
    mean_solver_ms: float
    p95_solver_ms: float
    p99_solver_ms: float
    max_solver_ms: float
    primary_success_rate: float
    accepted_solution_rate: float
    deadline_miss_rate: float
    positive_slack_ticks: int
    positive_slack_rate: float
    max_slack_m: float
    chance_violation_rate: float
    fallback_ticks: int
    fallback_rate: float
    guarantee_eligible_ticks: int
    guarantee_eligible_rate: float
    guarantee_eligible_episode: bool
    horizon_eligible_tick_count: int
    horizon_eligible_tick_rate: float
    horizon_ineligible_reason_counts: dict[str, int]
    episode_all_ticks_horizon_eligible: bool
    episode_any_fallback: bool
    episode_any_positive_slack: bool
    episode_any_deadline_miss: bool
    budget_failure_ticks: int
    maximum_budget_error: float
    residual_available_count: int
    residual_unavailable_count: int
    residual_invalid_count: int
    residual_available_rate: float
    residual_invalid_rate: float
    residual_gate_pass_rate: float
    residual_gate_fail_rate: float
    residual_gate_unknown_rate: float
    primal_residual_p50: float | None
    primal_residual_p95: float | None
    primal_residual_p99: float | None
    dual_residual_p50: float | None
    dual_residual_p95: float | None
    dual_residual_p99: float | None
    enforced_profile_count: int
    missing_enforced_profile_count: int
    post_solve_diagnostic_profile_count: int
    protocol: str = "algorithmic_comparison"
    total_ticks: int = 0
    deadline_miss_count: int = 0
    primary_applied_on_time_count: int = 0
    primary_applied_late_count: int = 0
    fallback_deadline_count: int = 0
    fallback_primary_invalid_count: int = 0
    primary_application_rate: float = 0.0
    fallback_application_rate: float = 0.0
    max_deadline_overrun_ms: float = 0.0

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def summarize_native_trial(
    result: Mapping[str, Any],
    config: NativeMuJoCoConfig,
    *,
    mode: str,
    noise_label: str,
    covariance_scale: float,
    seed: int,
    protocol: str = "algorithmic_comparison",
) -> NativeTrialResult:
    """Reduce native time-series output to a transparent trial-level record."""
    positions = np.asarray(result.get("pos", []), dtype=float)
    controls = np.asarray(result.get("u", []), dtype=float)
    clearances = np.asarray(result.get("clearance", []), dtype=float)
    solver_times = np.asarray(result.get("solver_time_ms", []), dtype=float)
    expected_steps = int(config.duration_s / config.mpc_timestep_s)
    completed_steps = len(positions)
    goal = np.array(
        [
            config.goal_position["x"],
            config.goal_position["y"],
            config.goal_position["z"],
        ],
        dtype=float,
    )
    start = np.array(
        [config.start["x"], config.start["y"], config.start["z"]],
        dtype=float,
    )
    finite_position = bool(positions.size and np.all(np.isfinite(positions)))
    finite_control = bool(not controls.size or np.all(np.isfinite(controls)))
    numerical_failure = not (finite_position and finite_control)
    final_error = (
        float(np.linalg.norm(positions[-1] - goal))
        if finite_position
        else float("inf")
    )
    finite_clearances = clearances[np.isfinite(clearances)]
    min_clearance = (
        float(np.min(finite_clearances)) if finite_clearances.size else float("inf")
    )
    path_length = (
        float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        if len(positions) > 1 and finite_position
        else 0.0
    )
    tracking_rmse = (
        _line_tracking_rmse(positions, start, goal) if finite_position else float("inf")
    )
    control_effort = (
        float(np.sum(np.sum(controls**2, axis=1)) * config.mpc_timestep_s)
        if finite_control and controls.size
        else float("inf")
    )

    estimates = np.asarray(result.get("estimated_state", []), dtype=float)
    estimation_rmse = 0.0
    if (
        finite_position
        and estimates.ndim == 2
        and len(estimates) == len(positions)
        and estimates.shape[1] >= 3
        and np.all(np.isfinite(estimates[:, :3]))
    ):
        estimation_rmse = float(
            np.sqrt(np.mean(np.sum((estimates[:, :3] - positions) ** 2, axis=1)))
        )

    primary_success = np.asarray(
        result.get("primary_solver_success", []),
        dtype=bool,
    )
    accepted = np.asarray(result.get("solution_accepted", []), dtype=bool)
    deadline_missed = np.asarray(result.get("deadline_missed", []), dtype=bool)
    fallback = np.asarray(result.get("fallback_active", []), dtype=bool)
    assurance = np.asarray(result.get("safety_assurance_status", []), dtype=str)
    guarantee_eligible = (assurance == "GUARANTEE_ELIGIBLE") | (
        assurance == "HORIZON_GUARANTEE_ELIGIBLE"
    )
    horizon_eligible = np.asarray(
        result.get("horizon_assurance_eligible", []),
        dtype=bool,
    )
    horizon_reasons = np.asarray(
        result.get("horizon_assurance_reason", []),
        dtype=str,
    )
    horizon_failed_checks = np.asarray(
        result.get("horizon_assurance_failed_checks", []),
        dtype=object,
    )

    def reason_counts() -> dict[str, int]:
        counts: dict[str, int] = {}
        for reason, checks in zip(horizon_reasons, horizon_failed_checks):
            if not reason or reason == "eligible":
                continue
            if isinstance(checks, (list, tuple)) and checks:
                for check in checks:
                    counts[str(check)] = counts.get(str(check), 0) + 1
            else:
                counts[str(reason)] = counts.get(str(reason), 0) + 1
        return counts

    slack = np.asarray(result.get("slack_horizon", []), dtype=float)
    if config.chance_constraints.enabled and slack.ndim >= 2 and len(slack):
        per_tick_slack = np.max(slack.reshape(len(slack), -1), axis=1)
    else:
        per_tick_slack = np.zeros(completed_steps, dtype=float)
    positive_slack = (
        per_tick_slack > config.chance_constraints.slack_tolerance_m
        if config.chance_constraints.enabled
        else np.zeros_like(per_tick_slack, dtype=bool)
    )

    residuals = np.asarray(result.get("chance_residual_horizon", []), dtype=float)
    finite_residuals = residuals[np.isfinite(residuals)]
    chance_violation_rate = (
        float(np.mean(finite_residuals < -config.safety_fallback.constraint_tolerance_m))
        if config.chance_constraints.enabled and finite_residuals.size
        else 0.0
    )

    budget_status = np.asarray(result.get("risk_budget_status", []), dtype=str)
    budget_failures = budget_status == "BUDGET_EXCEEDED"
    budget_total = np.asarray(result.get("risk_budget_total", []), dtype=float)
    budget_allocated = np.asarray(
        result.get("risk_budget_allocated", []),
        dtype=float,
    )
    finite_budget = np.isfinite(budget_total) & np.isfinite(budget_allocated)
    maximum_budget_error = (
        float(np.max(np.abs(budget_total[finite_budget] - budget_allocated[finite_budget])))
        if np.any(finite_budget)
        else 0.0
    )
    completed = (
        completed_steps == expected_steps
        and str(result.get("termination_reason", "")) == "completed"
    )
    collision = bool(result.get("collided", False))
    success = bool(
        completed
        and not collision
        and not numerical_failure
        and final_error <= config.goal_tolerance_m
    )
    guarantee_episode = bool(
        config.chance_constraints.enabled
        and completed_steps > 0
        and len(guarantee_eligible) == completed_steps
        and np.all(guarantee_eligible)
    )
    episode_all_ticks_horizon_eligible = bool(
        config.chance_constraints.enabled
        and completed_steps > 0
        and len(horizon_eligible) == completed_steps
        and np.all(horizon_eligible)
    )

    def rate(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size else 0.0

    dispositions = np.asarray(
        result.get("primary_disposition", []),
        dtype=str,
    )
    deadline_overruns = np.asarray(
        result.get("deadline_overrun_ms", []),
        dtype=float,
    )
    total_ticks = max(0, int(len(dispositions)))
    primary_applied_on_time_count = int(
        np.sum(dispositions == "PRIMARY_APPLIED_ON_TIME")
    )
    primary_applied_late_count = int(
        np.sum(dispositions == "PRIMARY_APPLIED_LATE")
    )
    fallback_deadline_count = int(
        np.sum(dispositions == "FALLBACK_APPLIED_DEADLINE_MISS")
    )
    fallback_primary_invalid_count = int(
        np.sum(dispositions == "FALLBACK_APPLIED_PRIMARY_INVALID")
    )
    primary_applied_total = (
        primary_applied_on_time_count + primary_applied_late_count
    )
    fallback_applied_total = (
        fallback_deadline_count + fallback_primary_invalid_count
    )
    deadline_miss_count = int(np.sum(deadline_missed)) if deadline_missed.size else 0
    max_deadline_overrun_ms = (
        float(np.max(deadline_overruns))
        if deadline_overruns.size
        else 0.0
    )

    profile_status = np.asarray(
        result.get("chance_profile_application_status", []),
        dtype=str,
    )
    enforced_profile_count = int(
        np.sum(profile_status == "APPLIED")
    )
    missing_enforced_profile_count = int(
        np.sum(profile_status == "NOT_APPLICABLE_DETERMINISTIC")
    )
    diagnostic_profiles = result.get(
        "post_solve_diagnostic_profile", []
    )
    post_solve_diagnostic_profile_count = (
        len(diagnostic_profiles)
        if isinstance(diagnostic_profiles, (list, tuple))
        else 0
    )

    primal_status = np.asarray(
        result.get("primary_solver_primal_residual_status", []),
        dtype=str,
    )
    dual_status = np.asarray(
        result.get("primary_solver_dual_residual_status", []),
        dtype=str,
    )
    gate_status = np.asarray(
        result.get("primary_solver_residual_gate_status", []),
        dtype=str,
    )
    raw_primal = np.asarray(
        result.get("primary_solver_primal_residual", []),
        dtype=object,
    )
    available_primal = np.asarray(
        [
            float(value)
            for value, status in zip(raw_primal, primal_status)
            if status == "AVAILABLE" and isinstance(value, (int, float))
        ],
        dtype=float,
    )
    raw_dual = np.asarray(
        result.get("primary_solver_dual_residual", []),
        dtype=object,
    )
    available_dual = np.asarray(
        [
            float(value)
            for value, status in zip(raw_dual, dual_status)
            if status == "AVAILABLE" and isinstance(value, (int, float))
        ],
        dtype=float,
    )
    residual_available_count = int(np.sum(primal_status == "AVAILABLE"))
    residual_unavailable_count = int(np.sum(primal_status == "UNAVAILABLE"))
    residual_invalid_count = int(np.sum(primal_status == "INVALID"))
    tick_count = len(primal_status) if len(primal_status) else 1
    gate_pass_rate = float(np.mean(gate_status == "PASS")) if gate_status.size else 0.0
    gate_fail_rate = float(
        np.mean((gate_status == "FAIL_THRESHOLD") | (gate_status == "FAIL_INVALID"))
    ) if gate_status.size else 0.0
    gate_unknown_rate = (
        float(np.mean(gate_status == "UNKNOWN_UNAVAILABLE"))
        if gate_status.size
        else 0.0
    )

    def available_percentile(values: np.ndarray, q: float) -> float | None:
        if not values.size:
            return None
        return float(np.percentile(values, q))

    return NativeTrialResult(
        noise_label=noise_label,
        covariance_scale=float(covariance_scale),
        mode=mode,
        seed=int(seed),
        expected_steps=expected_steps,
        completed_steps=completed_steps,
        completed=completed,
        termination_reason=str(result.get("termination_reason", "unknown")),
        success=success,
        collision=collision,
        numerical_failure=numerical_failure,
        final_error_m=final_error,
        min_clearance_m=min_clearance,
        path_length_m=path_length,
        tracking_rmse_m=tracking_rmse,
        control_effort=control_effort,
        estimation_position_rmse_m=estimation_rmse,
        mean_solver_ms=float(np.mean(solver_times)) if solver_times.size else 0.0,
        p95_solver_ms=_percentile(solver_times, 0.95),
        p99_solver_ms=_percentile(solver_times, 0.99),
        max_solver_ms=float(np.max(solver_times)) if solver_times.size else 0.0,
        primary_success_rate=rate(primary_success),
        accepted_solution_rate=rate(accepted),
        deadline_miss_rate=rate(deadline_missed),
        positive_slack_ticks=int(np.sum(positive_slack)),
        positive_slack_rate=rate(positive_slack),
        max_slack_m=float(np.max(per_tick_slack)) if per_tick_slack.size else 0.0,
        chance_violation_rate=chance_violation_rate,
        fallback_ticks=int(np.sum(fallback)),
        fallback_rate=rate(fallback),
        guarantee_eligible_ticks=int(np.sum(guarantee_eligible)),
        guarantee_eligible_rate=rate(guarantee_eligible),
        guarantee_eligible_episode=guarantee_episode,
        horizon_eligible_tick_count=int(np.sum(horizon_eligible)),
        horizon_eligible_tick_rate=rate(horizon_eligible),
        horizon_ineligible_reason_counts=reason_counts(),
        episode_all_ticks_horizon_eligible=episode_all_ticks_horizon_eligible,
        episode_any_fallback=bool(np.any(fallback)),
        episode_any_positive_slack=bool(np.any(positive_slack)),
        episode_any_deadline_miss=bool(np.any(deadline_missed)),
        budget_failure_ticks=int(np.sum(budget_failures)),
        maximum_budget_error=maximum_budget_error,
        residual_available_count=residual_available_count,
        residual_unavailable_count=residual_unavailable_count,
        residual_invalid_count=residual_invalid_count,
        residual_available_rate=residual_available_count / tick_count,
        residual_invalid_rate=residual_invalid_count / tick_count,
        residual_gate_pass_rate=gate_pass_rate,
        residual_gate_fail_rate=gate_fail_rate,
        residual_gate_unknown_rate=gate_unknown_rate,
        primal_residual_p50=available_percentile(available_primal, 50),
        primal_residual_p95=available_percentile(available_primal, 95),
        primal_residual_p99=available_percentile(available_primal, 99),
        dual_residual_p50=available_percentile(available_dual, 50),
        dual_residual_p95=available_percentile(available_dual, 95),
        dual_residual_p99=available_percentile(available_dual, 99),
        enforced_profile_count=enforced_profile_count,
        missing_enforced_profile_count=missing_enforced_profile_count,
        post_solve_diagnostic_profile_count=post_solve_diagnostic_profile_count,
        protocol=protocol,
        total_ticks=total_ticks,
        deadline_miss_count=deadline_miss_count,
        primary_applied_on_time_count=primary_applied_on_time_count,
        primary_applied_late_count=primary_applied_late_count,
        fallback_deadline_count=fallback_deadline_count,
        fallback_primary_invalid_count=fallback_primary_invalid_count,
        primary_application_rate=(
            primary_applied_total / total_ticks if total_ticks else 0.0
        ),
        fallback_application_rate=(
            fallback_applied_total / total_ticks if total_ticks else 0.0
        ),
        max_deadline_overrun_ms=max_deadline_overrun_ms,
    )


class NativeMonteCarloRunner:
    """Reuse one compiled controller per mode/noise combination."""

    def __init__(
        self,
        base_config: NativeMuJoCoConfig,
        *,
        protocol_type: str = "algorithmic_comparison",
    ):
        self.base_config = base_config
        self.protocol_type = str(protocol_type)
        self._controllers: dict[tuple[str, float], Any] = {}

    def run_trial(
        self,
        *,
        mode: str,
        noise_level: NoiseLevel,
        seed: int,
    ) -> NativeTrialResult:
        from quadrotor_mpc.application.native.runtime import run_coupled_simulation

        config = effective_native_config(
            self.base_config,
            mode=mode,
            covariance_scale=noise_level.covariance_scale,
            seed=seed,
            protocol_type=self.protocol_type,
        )
        key = (mode, noise_level.covariance_scale)
        if key not in self._controllers:
            self._controllers[key] = _build_controller(config)
        result = run_coupled_simulation(
            x0_vals=config.start,
            goal_pos=config.goal_position,
            goal_euler=config.goal_euler,
            bounds=config.bounds,
            obstacles=[dict(item) for item in config.obstacles],
            margin=config.safety_margin,
            sim_seconds=config.duration_s,
            mpc_dt=config.mpc_timestep_s,
            n_horizon=config.horizon_steps,
            max_iter=config.max_solver_iterations,
            mj_dt=config.mujoco_timestep_s,
            runtime=None,
            stop_on_goal=config.stop_on_goal,
            goal_tolerance=config.goal_tolerance_m,
            stop_on_collision=config.stop_on_collision,
            controller=self._controllers[key],
            estimation_options=config.estimation,
            covariance_options=config.covariance_propagation,
            chance_options=config.chance_constraints,
            safety_fallback_options=config.safety_fallback,
            protocol_type=self.protocol_type,
        )
        return summarize_native_trial(
            result,
            config,
            mode=mode,
            noise_label=noise_level.label,
            covariance_scale=noise_level.covariance_scale,
            seed=seed,
            protocol=self.protocol_type,
        )


def run_native_trial_batch(
    base_config_mapping: Mapping[str, Any],
    *,
    mode: str,
    noise_label: str,
    covariance_scale: float,
    seeds: Sequence[int],
    protocol_type: str = "algorithmic_comparison",
) -> list[dict[str, Any]]:
    """Process-safe batch entry point; one compiled controller serves all seeds."""
    import warnings

    warnings.filterwarnings("ignore", message="The ONNX feature is not available.*")
    warnings.filterwarnings("ignore", message="The opcua feature is not available.*")
    warnings.filterwarnings(
        "ignore",
        message="The approximateMPC feature requires PyTorch.*",
    )
    runner = NativeMonteCarloRunner(
        NativeMuJoCoConfig.from_mapping(base_config_mapping),
        protocol_type=protocol_type,
    )
    level = NoiseLevel(noise_label, covariance_scale)
    return [
        runner.run_trial(mode=mode, noise_level=level, seed=int(seed)).to_mapping()
        for seed in seeds
    ]


def wilson_interval(
    events: int,
    total: int,
    confidence_level: float = 0.95,
) -> list[float]:
    """Wilson score interval for a Bernoulli episode rate."""
    if total <= 0:
        return [0.0, 1.0]
    confidence_level = _probability(confidence_level, "confidence_level")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = events / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def _numeric_summary(values: Iterable[float]) -> dict[str, float] | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return None
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _rate_summary(
    items: Sequence[NativeTrialResult],
    predicate,
    confidence_level: float,
) -> dict[str, Any]:
    events = sum(bool(predicate(item)) for item in items)
    return {
        "events": events,
        "trials": len(items),
        "rate": events / len(items),
        "ci": wilson_interval(events, len(items), confidence_level),
    }


def _group_gates(
    items: Sequence[NativeTrialResult],
    summary: Mapping[str, Any],
    protocol: NativeMonteCarloProtocol,
    *,
    controller_period_ms: float,
) -> dict[str, Any]:
    mode = items[0].mode
    enough_trials = len(items) >= protocol.minimum_trials_for_claim
    execution_ok = all(item.completed and not item.numerical_failure for item in items)
    budget_ok = all(item.budget_failure_ticks == 0 for item in items)
    collision_upper = float(summary["collision_rate"]["ci"][1])
    empirical_collision_ok = (
        enough_trials
        and collision_upper <= protocol.empirical_collision_rate_limit
    )
    positive_slack_episodes = sum(item.positive_slack_ticks > 0 for item in items)
    fallback_episodes = sum(item.fallback_ticks > 0 for item in items)
    timing_field = (
        "p95_solver_ms"
        if math.isclose(protocol.timing_percentile, 0.95)
        else "p99_solver_ms"
    )
    timing_summary = summary[timing_field]
    timing_value = (
        float(timing_summary["p95"]) if timing_summary is not None else float("inf")
    )
    timing_ok = timing_value <= controller_period_ms
    # A real-time qualification claim requires the realtime protocol, an
    # end-to-end deadline clock, and the timing gate.  Algorithmic runs are
    # never allowed to produce a real-time pass.
    realtime_claim_eligible = bool(
        protocol.protocol_type == "realtime_qualification"
        and protocol.deadline_clock == "end_to_end_controller"
    )

    claim_blockers: list[str] = []
    if not enough_trials:
        claim_blockers.append("INSUFFICIENT_TRIALS")
    if not execution_ok:
        claim_blockers.append("NUMERICAL_OR_INCOMPLETE")
    if enough_trials and not empirical_collision_ok:
        claim_blockers.append("COLLISION_CONFIDENCE_BOUND")
    if (
        mode != "deterministic"
        and protocol.require_zero_budget_failures
        and not budget_ok
    ):
        claim_blockers.append("RISK_BUDGET_FAILURE")
    if (
        mode != "deterministic"
        and protocol.require_zero_positive_slack
        and positive_slack_episodes
    ):
        claim_blockers.append("POSITIVE_SLACK")
    if (
        mode != "deterministic"
        and protocol.require_zero_fallback
        and fallback_episodes
    ):
        claim_blockers.append("FALLBACK")

    if mode == "deterministic":
        claim_status = "NOT_APPLICABLE_DETERMINISTIC"
    elif claim_blockers:
        claim_status = f"BLOCKED_{claim_blockers[0]}"
    else:
        claim_status = "EMPIRICALLY_SUPPORTED_NOT_PROVEN"

    if not execution_ok or (
        mode != "deterministic"
        and protocol.require_zero_budget_failures
        and not budget_ok
    ):
        validation_status = "FAIL"
    elif (
        not enough_trials
        or not empirical_collision_ok
        or not timing_ok
        or claim_status.startswith("BLOCKED_")
        or not realtime_claim_eligible
    ):
        validation_status = "VALIDATED_WITH_LIMITATIONS"
    else:
        validation_status = "PASS"

    return {
        "sample_size": "PASS" if enough_trials else "INSUFFICIENT",
        "execution_integrity": "PASS" if execution_ok else "FAIL",
        "risk_accounting": (
            "NOT_APPLICABLE"
            if mode == "deterministic"
            else ("PASS" if budget_ok else "FAIL")
        ),
        "empirical_collision_bound": (
            "PASS"
            if empirical_collision_ok
            else ("INSUFFICIENT" if not enough_trials else "FAIL")
        ),
        "real_time_p99": "PASS" if timing_ok else "FAIL",
        "realtime_claim_eligible": realtime_claim_eligible,
        "claim_status": claim_status,
        "claim_blockers": claim_blockers if mode != "deterministic" else [],
        "validation_status": validation_status,
        "details": {
            "collision_ci_upper": collision_upper,
            "collision_rate_limit": protocol.empirical_collision_rate_limit,
            "positive_slack_episodes": positive_slack_episodes,
            "fallback_episodes": fallback_episodes,
            "timing_percentile": protocol.timing_percentile,
            "timing_across_trials_p95_ms": timing_value,
            "controller_period_ms": controller_period_ms,
        },
    }


def aggregate_native_trials(
    trials: Sequence[NativeTrialResult],
    protocol: NativeMonteCarloProtocol,
    *,
    controller_period_ms: float,
) -> dict[str, Any]:
    """Aggregate by noise and controller with paired deltas and claim gates."""
    if not trials:
        raise ValueError("at least one native Monte Carlo trial is required")
    active_protocols = {
        trial.protocol for trial in trials if trial.protocol != "UNKNOWN_LEGACY"
    }
    if len(active_protocols) > 1:
        raise ValueError(
            "cannot aggregate trials from multiple validation protocols: "
            f"{sorted(active_protocols)}; run separate campaigns per protocol"
        )
    legacy_trials = [trial for trial in trials if trial.protocol == "UNKNOWN_LEGACY"]
    if legacy_trials:
        raise ValueError(
            "cannot aggregate UNKNOWN_LEGACY trials into a new protocol summary"
        )
    numeric_fields = (
        "final_error_m",
        "min_clearance_m",
        "path_length_m",
        "tracking_rmse_m",
        "control_effort",
        "estimation_position_rmse_m",
        "mean_solver_ms",
        "p95_solver_ms",
        "p99_solver_ms",
        "max_solver_ms",
        "primary_success_rate",
        "accepted_solution_rate",
        "deadline_miss_rate",
        "positive_slack_rate",
        "max_slack_m",
        "chance_violation_rate",
        "fallback_rate",
        "guarantee_eligible_rate",
        "horizon_eligible_tick_rate",
        "maximum_budget_error",
        "primary_application_rate",
        "fallback_application_rate",
    )
    output: dict[str, Any] = {
        "schema_version": 2,
        "protocol": (
            active_protocols.pop() if active_protocols else "UNKNOWN_LEGACY"
        ),
        "deadline_policy": (
            deadline_policy_from_protocol(protocol.protocol).value
        ),
        "deadline_clock": protocol.deadline_clock,
        "control_period_ms": controller_period_ms,
        "confidence_level": protocol.confidence_level,
        "minimum_trials_for_claim": protocol.minimum_trials_for_claim,
        "noise_levels": {},
    }
    for level in protocol.noise_levels:
        level_items = [
            trial for trial in trials if trial.noise_label == level.label
        ]
        level_output: dict[str, Any] = {
            "covariance_scale": level.covariance_scale,
            "controllers": {},
            "paired_comparisons": {},
        }
        for mode in protocol.modes:
            items = [trial for trial in level_items if trial.mode == mode]
            if not items:
                continue
            summary: dict[str, Any] = {
                "trials": len(items),
                "success_rate": _rate_summary(
                    items,
                    lambda item: item.success,
                    protocol.confidence_level,
                ),
                "collision_rate": _rate_summary(
                    items,
                    lambda item: item.collision,
                    protocol.confidence_level,
                ),
                "incomplete_rate": _rate_summary(
                    items,
                    lambda item: not item.completed,
                    protocol.confidence_level,
                ),
                "numerical_failure_rate": _rate_summary(
                    items,
                    lambda item: item.numerical_failure,
                    protocol.confidence_level,
                ),
                "positive_slack_episode_rate": _rate_summary(
                    items,
                    lambda item: item.positive_slack_ticks > 0,
                    protocol.confidence_level,
                ),
                "fallback_episode_rate": _rate_summary(
                    items,
                    lambda item: item.fallback_ticks > 0,
                    protocol.confidence_level,
                ),
                "deadline_miss_episode_rate": _rate_summary(
                    items,
                    lambda item: item.deadline_miss_count > 0,
                    protocol.confidence_level,
                ),
                "primary_application_episode_rate": _rate_summary(
                    items,
                    lambda item: item.primary_application_rate >= 1.0,
                    protocol.confidence_level,
                ),
                "guarantee_eligible_episode_rate": _rate_summary(
                    items,
                    lambda item: item.guarantee_eligible_episode,
                    protocol.confidence_level,
                ),
                "horizon_eligible_episode_rate": _rate_summary(
                    items,
                    lambda item: item.episode_all_ticks_horizon_eligible,
                    protocol.confidence_level,
                ),
                "budget_failure_episode_rate": _rate_summary(
                    items,
                    lambda item: item.budget_failure_ticks > 0,
                    protocol.confidence_level,
                ),
            }
            for name in numeric_fields:
                summary[name] = _numeric_summary(
                    getattr(item, name) for item in items
                )
            summary["gates"] = _group_gates(
                items,
                summary,
                protocol,
                controller_period_ms=controller_period_ms,
            )
            level_output["controllers"][mode] = summary

        deterministic = {
            item.seed: item for item in level_items if item.mode == "deterministic"
        }
        for mode in ("individual", "joint"):
            compared = {item.seed: item for item in level_items if item.mode == mode}
            common = sorted(set(deterministic) & set(compared))
            if not common:
                continue
            level_output["paired_comparisons"][f"{mode}_minus_deterministic"] = {
                "paired_seeds": common,
                "metrics": {
                    name: _numeric_summary(
                        getattr(compared[seed], name)
                        - getattr(deterministic[seed], name)
                        for seed in common
                    )
                    for name in (
                        "final_error_m",
                        "min_clearance_m",
                        "path_length_m",
                        "tracking_rmse_m",
                        "control_effort",
                        "p99_solver_ms",
                        "positive_slack_rate",
                        "fallback_rate",
                    )
                },
                "collision_discordance": {
                    "deterministic_only": sum(
                        deterministic[seed].collision and not compared[seed].collision
                        for seed in common
                    ),
                    f"{mode}_only": sum(
                        compared[seed].collision and not deterministic[seed].collision
                        for seed in common
                    ),
                },
            }
        output["noise_levels"][level.label] = level_output

    groups = [
        summary
        for level in output["noise_levels"].values()
        for summary in level["controllers"].values()
    ]
    chance_groups = [
        summary
        for level in output["noise_levels"].values()
        for mode, summary in level["controllers"].items()
        if mode != "deterministic"
    ]
    execution_ok = all(
        summary["gates"]["execution_integrity"] == "PASS" for summary in groups
    )
    enough_samples = all(
        summary["gates"]["sample_size"] == "PASS" for summary in groups
    )
    real_time_ok = all(
        summary["gates"]["real_time_p99"] == "PASS" for summary in groups
    )
    claims_supported = bool(chance_groups) and all(
        summary["gates"]["claim_status"]
        == "EMPIRICALLY_SUPPORTED_NOT_PROVEN"
        for summary in chance_groups
    )
    if not execution_ok:
        stage_status = "FAIL"
    elif not enough_samples:
        stage_status = "INCOMPLETE"
    elif not real_time_ok or not claims_supported:
        stage_status = "VALIDATED_WITH_LIMITATIONS"
    else:
        stage_status = "PASS"
    output["overall"] = {
        "stage_status": stage_status,
        "execution_integrity": "PASS" if execution_ok else "FAIL",
        "sample_size": "PASS" if enough_samples else "INSUFFICIENT",
        "real_time": "PASS" if real_time_ok else "FAIL",
        "probabilistic_claim": (
            "EMPIRICALLY_SUPPORTED_NOT_PROVEN"
            if claims_supported
            else "BLOCKED"
        ),
        "interpretation": (
            "A finite Monte Carlo result is empirical evidence only. The configured "
            "joint risk is per receding prediction horizon and is not an episode-wide "
            "probability guarantee."
        ),
    }
    return output


def _git_command(root: Path, *arguments: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _hash_validation_source(repository_root: Path) -> tuple[str, int]:
    process = _git_command(
        repository_root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    if process is None or process.returncode != 0:
        return "", 0
    selected: list[Path] = []
    for raw_path in process.stdout.split("\0"):
        if not raw_path:
            continue
        relative = Path(raw_path)
        if relative.parts[:2] == ("2.Code", "validation"):
            continue
        if relative.parts[:2] == ("2.Code", "outputs"):
            continue
        if (
            relative == Path("README.md")
            or relative == Path("LICENSE")
            or relative.parts[:2] == (".github", "workflows")
            or (
                relative.parts
                and relative.parts[0] == "2.Code"
                and relative.suffix
                in {".py", ".toml", ".yaml", ".yml", ".txt", ".lock", ".xml", ".obj"}
            )
        ):
            selected.append(relative)

    digest = hashlib.sha256()
    count = 0
    for relative in sorted(selected, key=lambda item: item.as_posix()):
        source = repository_root / relative
        if not source.is_file():
            continue
        encoded_path = relative.as_posix().encode()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        payload = source.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
    return digest.hexdigest(), count


def _hash_loose_validation_source(code_root: Path) -> tuple[str, int]:
    """Hash a source archive that has no Git identity without trusting it."""
    source_root = code_root.resolve()
    for candidate in (source_root, *source_root.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "quadrotor_mpc").is_dir()
        ):
            source_root = candidate
            break
    repository_root = (
        source_root.parent
        if (source_root.parent / "README.md").is_file()
        else source_root
    )
    selected: list[Path] = []
    suffixes = {".py", ".toml", ".yaml", ".yml", ".txt", ".lock", ".xml", ".obj"}
    excluded_parts = {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".venv",
        "build",
        "dist",
        "outputs",
        "validation",
    }
    for source in repository_root.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(repository_root)
        if excluded_parts.intersection(relative.parts):
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if (
            relative in {Path("README.md"), Path("LICENSE")}
            or relative.parts[:2] == (".github", "workflows")
            or source.suffix in suffixes
        ):
            selected.append(relative)

    digest = hashlib.sha256()
    count = 0
    for relative in sorted(selected, key=lambda item: item.as_posix()):
        encoded_path = relative.as_posix().encode()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        payload = (repository_root / relative).read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
    return digest.hexdigest(), count


def _installed_distribution_provenance() -> dict[str, Any]:
    """Fingerprint an installed wheel and reject unverifiable/editable installs."""
    try:
        distribution = metadata.distribution("quadrotor-mpc-sim")
    except metadata.PackageNotFoundError:
        return {
            "status": "UNVERIFIED_DISTRIBUTION",
            "git_commit": None,
            "git_branch": None,
            "git_clean": None,
            "source_snapshot_sha256": None,
            "source_file_count": 0,
            "distribution_version": None,
        }

    editable = False
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            editable = bool(json.loads(direct_url).get("dir_info", {}).get("editable"))
        except json.JSONDecodeError:
            editable = True

    selected: list[tuple[str, Path]] = []
    active_module = Path(__file__).resolve()
    module_belongs_to_distribution = False
    for package_path in distribution.files or ():
        logical_path = Path(str(package_path))
        if "__pycache__" in logical_path.parts or logical_path.suffix == ".pyc":
            continue
        if ".dist-info" in logical_path.as_posix():
            continue
        installed_path = Path(distribution.locate_file(package_path))
        if installed_path.is_file():
            selected.append((logical_path.as_posix(), installed_path))
            module_belongs_to_distribution = bool(
                module_belongs_to_distribution
                or installed_path.resolve() == active_module
            )

    if not module_belongs_to_distribution:
        snapshot_hash, file_count = _hash_loose_validation_source(active_module.parent)
        return {
            "status": "UNVERIFIED_SOURCE_TREE",
            "git_commit": None,
            "git_branch": None,
            "git_clean": None,
            "source_snapshot_sha256": snapshot_hash or None,
            "source_file_count": file_count,
            "distribution_version": None,
        }

    digest = hashlib.sha256()
    count = 0
    for logical_name, installed_path in sorted(selected):
        encoded_path = logical_name.encode()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        payload = installed_path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1

    verified = bool(count)
    if editable:
        status = "EDITABLE_DISTRIBUTION"
    elif verified:
        status = "INSTALLED_DISTRIBUTION"
    else:
        status = "UNVERIFIED_DISTRIBUTION"
    return {
        "status": status,
        "git_commit": None,
        "git_branch": None,
        "git_clean": None,
        "source_snapshot_sha256": digest.hexdigest() if verified else None,
        "source_file_count": count,
        "distribution_version": distribution.version,
    }


def source_provenance(start: str | Path | None = None) -> dict[str, Any]:
    """Describe the exact validation source and whether it is a clean commit."""
    search_root = Path.cwd() if start is None else Path(start)
    top_level = _git_command(search_root, "rev-parse", "--show-toplevel")
    if top_level is None or top_level.returncode != 0:
        return _installed_distribution_provenance()

    repository_root = Path(top_level.stdout.strip()).resolve()
    commit_process = _git_command(repository_root, "rev-parse", "HEAD")
    branch_process = _git_command(repository_root, "branch", "--show-current")
    status_process = _git_command(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    commit = (
        commit_process.stdout.strip()
        if commit_process is not None and commit_process.returncode == 0
        else None
    )
    branch = (
        branch_process.stdout.strip()
        if branch_process is not None and branch_process.returncode == 0
        else None
    )
    clean = bool(
        status_process is not None
        and status_process.returncode == 0
        and not status_process.stdout.strip()
    )
    snapshot_hash, file_count = _hash_validation_source(repository_root)
    return {
        "status": "CLEAN_GIT_COMMIT" if clean else "DIRTY_GIT_SNAPSHOT",
        "git_commit": commit,
        "git_branch": branch or None,
        "git_clean": clean,
        "source_snapshot_sha256": snapshot_hash or None,
        "source_file_count": file_count,
        "distribution_version": None,
    }


def _versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("numpy", "scipy", "PyYAML", "mujoco", "casadi", "do-mpc"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return result


def protocol_fingerprint(
    protocol: NativeMonteCarloProtocol,
    base_config: NativeMuJoCoConfig,
) -> str:
    payload = {
        "protocol": protocol.to_mapping(),
        "base_config": base_config.to_mapping(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def create_validation_directory(
    protocol: NativeMonteCarloProtocol,
    base_config: NativeMuJoCoConfig,
    *,
    command: Sequence[str] | None = None,
    run_id: str | None = None,
    allow_dirty_source: bool = False,
) -> Path:
    """Create a self-describing run directory before expensive trials begin."""
    timestamp = datetime.now(timezone.utc)
    provenance = source_provenance()
    non_release_sources = {
        "DIRTY_GIT_SNAPSHOT",
        "EDITABLE_DISTRIBUTION",
        "UNVERIFIED_DISTRIBUTION",
        "UNVERIFIED_SOURCE_TREE",
    }
    if provenance["status"] in non_release_sources and not allow_dirty_source:
        raise RuntimeError(
            "release validation requires a clean Git source or a fingerprinted "
            "non-editable distribution; commit/install the intended changes or pass "
            "allow_dirty_source=True for a non-release smoke campaign"
        )
    if run_id is None:
        run_id = timestamp.strftime("%Y%m%dT%H%M%SZ") + f"-{protocol.name}"
    directory = protocol.output_dir / run_id
    suffix = 1
    while directory.exists():
        directory = protocol.output_dir / f"{run_id}-{suffix:02d}"
        suffix += 1
    directory.mkdir(parents=True)
    (directory / "protocol.yaml").write_text(
        yaml.safe_dump(protocol.to_mapping(), sort_keys=False),
        encoding="utf-8",
    )
    (directory / "base_config.yaml").write_text(
        yaml.safe_dump(base_config.to_mapping(), sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 2,
        "stage": 8,
        "run_id": directory.name,
        "created_at_utc": timestamp.isoformat(),
        "status": "RUNNING",
        "protocol_fingerprint": protocol_fingerprint(protocol, base_config),
        "git_commit": provenance["git_commit"],
        "source": provenance,
        "command": list(command if command is not None else sys.argv),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": _versions(),
        },
        "expected_trials": (
            protocol.trials * len(protocol.modes) * len(protocol.noise_levels)
        ),
    }
    (directory / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return directory


def append_trial_checkpoint(directory: Path, trial: NativeTrialResult) -> None:
    """Append one completed trial so an interrupted long run remains recoverable."""
    with (directory / "trials.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(trial.to_mapping(), separators=(",", ":")) + "\n")


def load_trial_checkpoint(directory: str | Path) -> list[NativeTrialResult]:
    path = Path(directory) / "trials.jsonl"
    if not path.exists():
        return []
    trials: list[NativeTrialResult] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            trials.append(NativeTrialResult(**json.loads(line)))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid trial checkpoint at {path}:{line_number}"
            ) from exc
    return trials


def _save_plot(
    trials: Sequence[NativeTrialResult],
    aggregate: Mapping[str, Any],
    path: Path,
    *,
    controller_period_ms: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [level for level in aggregate["noise_levels"]]
    modes = sorted({trial.mode for trial in trials}, key=SUPPORTED_MODES.index)
    colors = {
        "deterministic": "#64748b",
        "individual": "#f59e0b",
        "joint": "#2563eb",
    }
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    x = np.arange(len(labels), dtype=float)
    width = 0.22
    offsets = {
        mode: (index - (len(modes) - 1) / 2.0) * width
        for index, mode in enumerate(modes)
    }
    for mode in modes:
        rates = []
        lows = []
        highs = []
        success = []
        clearances: list[list[float]] = []
        errors: list[list[float]] = []
        slack_rates = []
        solver_p99 = []
        for label in labels:
            summary = aggregate["noise_levels"][label]["controllers"][mode]
            collision = summary["collision_rate"]
            rates.append(collision["rate"])
            lows.append(collision["ci"][0])
            highs.append(collision["ci"][1])
            success.append(summary["success_rate"]["rate"])
            subset = [
                trial
                for trial in trials
                if trial.noise_label == label and trial.mode == mode
            ]
            clearances.append([trial.min_clearance_m for trial in subset])
            errors.append([trial.final_error_m for trial in subset])
            slack_rates.append(
                summary["positive_slack_episode_rate"]["rate"]
            )
            solver_p99.append(summary["p99_solver_ms"]["median"])
        position = x + offsets[mode]
        asymmetric = np.array([np.array(rates) - lows, np.array(highs) - rates])
        axes[0, 0].errorbar(
            position,
            rates,
            yerr=asymmetric,
            fmt="o-",
            capsize=3,
            color=colors[mode],
            label=mode,
        )
        axes[0, 1].plot(position, success, "o-", color=colors[mode], label=mode)
        axes[1, 1].plot(
            position,
            slack_rates,
            "o-",
            color=colors[mode],
            label=mode,
        )
        axes[1, 2].plot(
            position,
            solver_p99,
            "o-",
            color=colors[mode],
            label=mode,
        )
        for label_index, samples in enumerate(clearances):
            axes[0, 2].boxplot(
                samples,
                positions=[x[label_index] + offsets[mode]],
                widths=width * 0.8,
                patch_artist=True,
                boxprops={"facecolor": colors[mode], "alpha": 0.45},
                medianprops={"color": "black"},
                manage_ticks=False,
            )
        for label_index, samples in enumerate(errors):
            axes[1, 0].boxplot(
                samples,
                positions=[x[label_index] + offsets[mode]],
                widths=width * 0.8,
                patch_artist=True,
                boxprops={"facecolor": colors[mode], "alpha": 0.45},
                medianprops={"color": "black"},
                manage_ticks=False,
            )

    titles = (
        "Collision rate with confidence interval",
        "Goal success rate",
        "Minimum clearance distribution",
        "Final goal error distribution",
        "Episodes with positive slack",
        "Per-trial solver p99 (median)",
    )
    ylabels = ("rate", "rate", "m", "m", "rate", "ms")
    for axis, title, ylabel in zip(axes.flat, titles, ylabels):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels)
        axis.grid(True, alpha=0.25)
    axes[1, 2].axhline(
        controller_period_ms,
        color="#dc2626",
        linestyle="--",
        label=f"controller period ({controller_period_ms:g} ms)",
    )
    axes[0, 0].legend()
    axes[1, 2].legend(fontsize=8)
    figure.suptitle(
        "Native MuJoCo paired Monte Carlo validation",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _format_rate(summary: Mapping[str, Any]) -> str:
    low, high = summary["ci"]
    return f"{summary['rate']:.3f} [{low:.3f}, {high:.3f}]"


def _save_markdown_report(
    aggregate: Mapping[str, Any],
    protocol: NativeMonteCarloProtocol,
    path: Path,
) -> None:
    lines = [
        "# Native Monte Carlo Validation — Stage 8",
        "",
        f"- Overall status: **{aggregate['overall']['stage_status']}**",
        f"- Confidence level: {protocol.confidence_level:.1%}",
        f"- Trials per controller/noise level: {protocol.trials}",
        f"- Paired seeds: {protocol.first_seed}–{protocol.first_seed + protocol.trials - 1}",
        "",
        "## Aggregate results",
        "",
        (
        "| Noise | Controller | Success (CI) | Collision (CI) | "
        "Min clearance mean | Max slack mean | Solver p99 median | Claim blockers |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for label, level in aggregate["noise_levels"].items():
        for mode, summary in level["controllers"].items():
            lines.append(
                f"| {label} ({level['covariance_scale']:g}Σ) | {mode} | "
                f"{_format_rate(summary['success_rate'])} | "
                f"{_format_rate(summary['collision_rate'])} | "
                f"{summary['min_clearance_m']['mean']:.6f} m | "
                f"{summary['max_slack_m']['mean']:.6f} m | "
                f"{summary['p99_solver_ms']['median']:.3f} ms | "
                f"{', '.join(summary['gates']['claim_blockers']) or summary['gates']['claim_status']} |"
            )
    lines.extend(
        [
            "",
            "## Gate interpretation",
            "",
            f"- Execution integrity: **{aggregate['overall']['execution_integrity']}**",
            f"- Required sample size: **{aggregate['overall']['sample_size']}**",
            f"- Real-time gate: **{aggregate['overall']['real_time']}**",
            f"- Probabilistic claim: **{aggregate['overall']['probabilistic_claim']}**",
            (
                "- Release provenance: "
                f"**{aggregate['overall'].get('release_provenance', 'NOT_RECORDED')}**"
            ),
            "",
            aggregate["overall"]["interpretation"],
            "",
            (
                "Positive slack means the hard chance constraint was relaxed. Fallback means "
                "the primary NMPC command was not applied. Either event blocks a probabilistic "
                "safety claim under this protocol, even when no geometric collision was observed."
            ),
            "",
            "## Reproduction",
            "",
            (
                "Use `quadrotor-mpc-monte-carlo "
                "--config config/native_monte_carlo.yaml`. "
                "The manifest records the source revision, environment, protocol fingerprint "
                "and expected trial count. `trials.jsonl` is append-only and supports resume."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_validation_artifacts(
    directory: str | Path,
    trials: Sequence[NativeTrialResult],
    protocol: NativeMonteCarloProtocol,
    base_config: NativeMuJoCoConfig,
) -> dict[str, Path]:
    """Write deterministic final artifacts and close the run manifest."""
    target = Path(directory)
    expected_keys = {
        (level.label, mode, seed)
        for level in protocol.noise_levels
        for mode in protocol.modes
        for seed in protocol.seeds
    }
    actual_keys = {
        (trial.noise_label, trial.mode, trial.seed)
        for trial in trials
    }
    if len(actual_keys) != len(trials):
        raise ValueError("native Monte Carlo trials contain duplicate keys")
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "native Monte Carlo trial matrix is incomplete or unexpected: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    ordered = sorted(
        trials,
        key=lambda item: (
            [level.label for level in protocol.noise_levels].index(item.noise_label),
            protocol.modes.index(item.mode),
            item.seed,
        ),
    )
    checkpoint_trials = load_trial_checkpoint(target)
    checkpoint_keys = {
        (trial.noise_label, trial.mode, trial.seed)
        for trial in checkpoint_trials
    }
    if len(checkpoint_keys) != len(checkpoint_trials):
        raise ValueError("native Monte Carlo checkpoint contains duplicate keys")
    unexpected_checkpoint = checkpoint_keys - expected_keys
    if unexpected_checkpoint:
        raise ValueError(
            "native Monte Carlo checkpoint contains unexpected keys: "
            f"{sorted(unexpected_checkpoint)[:5]}"
        )
    for trial in ordered:
        key = (trial.noise_label, trial.mode, trial.seed)
        if key not in checkpoint_keys:
            append_trial_checkpoint(target, trial)
            checkpoint_keys.add(key)
    if checkpoint_keys != expected_keys:
        raise ValueError("native Monte Carlo checkpoint could not be completed")

    rows = [trial.to_mapping() for trial in ordered]
    trials_csv = target / "trials.csv"
    with trials_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = aggregate_native_trials(
        ordered,
        protocol,
        controller_period_ms=base_config.mpc_timestep_s * 1000.0,
    )
    aggregate_path = target / "aggregate.json"
    report_path = target / "report.md"
    plot_path = target / "report.png"

    manifest_path = target / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    source = _mapping(manifest.get("source", {}), "manifest.source")
    provenance_ok = source.get("status") in {
        "CLEAN_GIT_COMMIT",
        "INSTALLED_DISTRIBUTION",
    }
    aggregate["overall"]["release_provenance"] = (
        "PASS" if provenance_ok else "FAIL_DIRTY_SOURCE"
    )
    aggregate["overall"]["release_eligible"] = bool(
        provenance_ok
        and aggregate["overall"]["execution_integrity"] == "PASS"
        and aggregate["overall"]["sample_size"] == "PASS"
    )
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _save_markdown_report(aggregate, protocol, report_path)
    _save_plot(
        ordered,
        aggregate,
        plot_path,
        controller_period_ms=base_config.mpc_timestep_s * 1000.0,
    )
    manifest.update(
        {
            "status": "COMPLETED",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_trials": len(ordered),
            "checkpoint_trials": len(checkpoint_keys),
            "overall_validation_status": aggregate["overall"]["stage_status"],
            "files": {
                "protocol": "protocol.yaml",
                "base_config": "base_config.yaml",
                "checkpoint": "trials.jsonl",
                "trials": trials_csv.name,
                "aggregate": aggregate_path.name,
                "report": report_path.name,
                "plot": plot_path.name,
            },
        }
    )
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "directory": target,
        "manifest": manifest_path,
        "trials": trials_csv,
        "aggregate": aggregate_path,
        "report": report_path,
        "plot": plot_path,
    }
