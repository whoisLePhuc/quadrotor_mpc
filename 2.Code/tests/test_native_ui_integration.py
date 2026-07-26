from __future__ import annotations

import unittest
from pathlib import Path

from quadrotor_mpc.interfaces.desktop.model import (
    DANGER,
    MUTED,
    OK,
    WARNING,
    PanelRuntimeContext,
    build_panel_view,
    panel_transition_alerts,
)
from quadrotor_mpc.interfaces.desktop.panel import DesktopPanelOptions
from quadrotor_mpc.interfaces.desktop.viewer import load_native_mujoco_config

CODE_ROOT = Path(__file__).resolve().parents[1]


def _ccmpc_context() -> PanelRuntimeContext:
    return PanelRuntimeContext(
        scenario_name="test-ccmpc",
        mpc_period_ms=50.0,
        estimation_enabled=True,
        chance_constraints_enabled=True,
        covariance_propagation_enabled=True,
        supervisor_enabled=True,
        solve_deadline_ms=50.0,
        guarantee_slack_tolerance_m=1e-6,
        maximum_acceptable_slack_m=0.08,
        maximum_solver_residual=1e-3,
        configured_risk_semantics="joint",
        configured_risk_allocation="uniform",
    )


def _safe_sample() -> dict[str, object]:
    return {
        "time_s": 1.5,
        "position": [0.2, 0.3, 1.2],
        "control": [0.01, 0.0, 0.0, 0.0],
        "goal_distance_m": 2.1,
        "min_clearance_m": 0.42,
        "solver_time_ms": 47.0,
        "collided": False,
        "paused": False,
        "completed": False,
        "solver_status": "SOLVED_SAFE",
        "primary_solver_status": "Solve_Succeeded",
        "primary_solver_success": True,
        "primary_solver_iterations": 12,
        "primary_solver_primal_residual": 2e-7,
        "primary_solver_dual_residual": 4e-7,
        "command_source": "PRIMARY_NMPC",
        "solution_accepted": True,
        "fallback_active": False,
        "fallback_level": 0,
        "fallback_reason": "",
        "deadline_missed": False,
        "safety_assurance_status": "GUARANTEE_ELIGIBLE",
        "risk_semantics": "joint",
        "risk_allocation_method": "uniform",
        "risk_budget_total": 0.10,
        "risk_budget_allocated": 0.10,
        "risk_budget_remaining": 0.0,
        "risk_constraint_count": 63,
        "risk_budget_status": "BUDGET_OK",
        "minimum_chance_residual_m": 0.012,
        "maximum_slack_m": 0.0,
    }


class PanelRuntimeContextTests(unittest.TestCase):
    def test_stage7_context_comes_from_effective_ccmpc_config(self):
        config = load_native_mujoco_config(
            CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml"
        )
        context = config.panel_runtime_context()
        self.assertTrue(context.estimation_enabled)
        self.assertTrue(context.chance_constraints_enabled)
        self.assertTrue(context.covariance_propagation_enabled)
        self.assertTrue(context.supervisor_enabled)
        self.assertAlmostEqual(context.mpc_period_ms, 50.0)
        self.assertAlmostEqual(context.solve_deadline_ms, 50.0)
        self.assertEqual(context.configured_risk_semantics, "joint")
        self.assertEqual(context.configured_risk_allocation, "uniform")
        self.assertIn("CC-MPC", context.mode_label)

    def test_context_mapping_round_trip_is_lossless(self):
        context = _ccmpc_context()
        self.assertEqual(
            PanelRuntimeContext.from_mapping(context.to_mapping()),
            context,
        )

    def test_panel_alert_limit_is_validated_and_round_trips_config(self):
        options = DesktopPanelOptions.from_mapping(
            {
                "enabled": True,
                "update_hz": 20.0,
                "history_seconds": 12.0,
                "maximum_alerts": 7,
            }
        )
        self.assertEqual(options.maximum_alerts, 7)
        with self.assertRaisesRegex(ValueError, "maximum_alerts"):
            DesktopPanelOptions.from_mapping({"maximum_alerts": 0})


class PanelViewProjectionTests(unittest.TestCase):
    def test_safe_joint_solution_is_rendered_as_guarantee_eligible(self):
        view = build_panel_view(_safe_sample(), _ccmpc_context())
        self.assertEqual(view.card("controller").value, "PRIMARY NMPC")
        self.assertEqual(view.card("controller").tone, OK)
        self.assertEqual(view.card("assurance").value, "GUARANTEE ELIGIBLE")
        self.assertEqual(view.card("assurance").tone, OK)
        self.assertEqual(view.card("risk").value, "BUDGET_OK")
        self.assertEqual(view.card("slack").value, "HARD-SAFE")
        self.assertEqual(view.card("deadline").value, "ON TIME")

    def test_positive_slack_is_visible_and_never_labeled_guaranteed(self):
        sample = _safe_sample()
        sample.update(
            {
                "solver_status": "SOLVED_WITH_SLACK",
                "maximum_slack_m": 0.03,
                "minimum_chance_residual_m": -0.03,
                "safety_assurance_status": "NOT_GUARANTEED_POSITIVE_SLACK",
            }
        )
        view = build_panel_view(sample, _ccmpc_context())
        self.assertEqual(view.card("slack").value, "DEGRADED")
        self.assertEqual(view.card("slack").tone, WARNING)
        self.assertEqual(view.card("assurance").value, "NOT GUARANTEED")
        self.assertEqual(view.card("assurance").tone, WARNING)
        self.assertNotIn("GUARANTEE ELIGIBLE", view.banner)

    def test_fallback_and_deadline_miss_are_high_visibility_failures(self):
        sample = _safe_sample()
        sample.update(
            {
                "solver_time_ms": 112.0,
                "solution_accepted": False,
                "command_source": "POSITION_HOLD_PD",
                "fallback_active": True,
                "fallback_level": 2,
                "fallback_reason": "DEADLINE_MISSED",
                "deadline_missed": True,
                "safety_assurance_status": "NOT_GUARANTEED_FALLBACK_ACTIVE",
            }
        )
        view = build_panel_view(sample, _ccmpc_context())
        self.assertEqual(view.card("controller").value, "FALLBACK L2")
        self.assertEqual(view.card("controller").tone, WARNING)
        self.assertEqual(view.card("assurance").tone, DANGER)
        self.assertEqual(view.card("deadline").value, "DEADLINE MISSED")
        self.assertEqual(view.card("deadline").tone, DANGER)

    def test_deterministic_mode_does_not_claim_probabilistic_assurance(self):
        context = PanelRuntimeContext(
            scenario_name="deterministic",
            mpc_period_ms=50.0,
            estimation_enabled=False,
            chance_constraints_enabled=False,
            covariance_propagation_enabled=False,
            supervisor_enabled=False,
            solve_deadline_ms=0.0,
            guarantee_slack_tolerance_m=0.0,
            maximum_acceptable_slack_m=0.0,
            maximum_solver_residual=0.0,
            configured_risk_semantics="disabled",
            configured_risk_allocation="none",
        )
        sample = _safe_sample()
        sample.update(
            {
                "risk_semantics": "disabled",
                "risk_budget_total": None,
                "risk_budget_status": "DISABLED",
                "maximum_slack_m": None,
                "minimum_chance_residual_m": None,
                "safety_assurance_status": "",
            }
        )
        view = build_panel_view(sample, context)
        self.assertEqual(view.card("assurance").value, "DETERMINISTIC")
        self.assertEqual(view.card("assurance").tone, MUTED)
        self.assertEqual(view.card("risk").value, "DISABLED")
        self.assertEqual(view.card("slack").value, "DISABLED")
        self.assertEqual(view.card("deadline").value, "MONITOR ONLY")

    def test_joint_budget_failure_is_rendered_as_danger(self):
        sample = _safe_sample()
        sample.update(
            {
                "risk_budget_allocated": 0.11,
                "risk_budget_remaining": 0.0,
                "risk_budget_status": "BUDGET_EXCEEDED",
            }
        )
        view = build_panel_view(sample, _ccmpc_context())
        self.assertEqual(view.card("risk").value, "BUDGET_EXCEEDED")
        self.assertEqual(view.card("risk").tone, DANGER)

    def test_transition_alerts_are_emitted_once_per_state_change(self):
        previous = build_panel_view(_safe_sample(), _ccmpc_context())
        sample = _safe_sample()
        sample.update(
            {
                "time_s": 1.55,
                "solution_accepted": False,
                "fallback_active": True,
                "fallback_level": 2,
                "deadline_missed": True,
                "safety_assurance_status": "NOT_GUARANTEED_FALLBACK_ACTIVE",
            }
        )
        current = build_panel_view(sample, _ccmpc_context())
        alerts = panel_transition_alerts(previous, current)
        self.assertEqual(
            {alert.key for alert in alerts},
            {"fallback_entered", "deadline_missed", "assurance_lost"},
        )
        repeated = panel_transition_alerts(current, current)
        self.assertEqual(repeated, ())


if __name__ == "__main__":
    unittest.main()
