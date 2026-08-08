"""Guard the package boundaries introduced by the layered refactor."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import tomllib

CODE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = CODE_ROOT / "quadrotor_mpc"


def local_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return {name for name in imports if name.startswith("quadrotor_mpc.")}


class LayerArchitectureTests(unittest.TestCase):
    def test_root_contains_no_application_modules(self):
        self.assertEqual(
            list(CODE_ROOT.glob("*.py")),
            [],
            "runtime modules belong under quadrotor_mpc/, not at the 2.Code root",
        )

    def test_console_scripts_target_packaged_cli_adapters(self):
        project = tomllib.loads(
            (CODE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        for target in project["project"]["scripts"].values():
            self.assertTrue(
                target.startswith("quadrotor_mpc.interfaces.cli."),
                target,
            )

    def test_inward_layers_do_not_depend_on_outer_interfaces(self):
        forbidden_by_layer = {
            "core": (
                "quadrotor_mpc.application",
                "quadrotor_mpc.control",
                "quadrotor_mpc.estimation",
                "quadrotor_mpc.infrastructure",
                "quadrotor_mpc.interfaces",
                "quadrotor_mpc.reporting",
            ),
            "control": (
                "quadrotor_mpc.application",
                "quadrotor_mpc.interfaces",
                "quadrotor_mpc.reporting",
            ),
            "estimation": (
                "quadrotor_mpc.application",
                "quadrotor_mpc.control",
                "quadrotor_mpc.infrastructure",
                "quadrotor_mpc.interfaces",
                "quadrotor_mpc.reporting",
            ),
            "infrastructure": (
                "quadrotor_mpc.application",
                "quadrotor_mpc.interfaces",
                "quadrotor_mpc.reporting",
            ),
        }
        violations: list[str] = []
        for layer, forbidden_prefixes in forbidden_by_layer.items():
            for path in (PACKAGE_ROOT / layer).rglob("*.py"):
                for imported in local_imports(path):
                    if imported.startswith(forbidden_prefixes):
                        violations.append(
                            f"{path.relative_to(CODE_ROOT)} imports {imported}"
                        )
        self.assertEqual(violations, [])


class PackageBoundaryTests(unittest.TestCase):
    """Section 14: both controller packages import cleanly with their roles."""

    def test_public_api_imports_for_both_controller_packages(self):
        from quadrotor_mpc.control.ccmpc import (
            CCMPC,
            EllipsoidalObstacle,
            ObstacleManager,
            QuadrotorDynamics,
            UncertaintyPropagator,
            VIODriftModel,
            chance_constraint_residual,
            collision_clearance,
            symmetric_matrix_sqrt,
        )
        from quadrotor_mpc.control.nmpc import (
            core,
            covariance,
            deterministic,
            risk_budget,
            safety,
        )

        self.assertTrue(callable(CCMPC))
        self.assertTrue(callable(collision_clearance))
        self.assertTrue(callable(chance_constraint_residual))
        self.assertTrue(hasattr(core, "DRONE_RADIUS"))
        self.assertTrue(hasattr(deterministic, "DeterministicNMPCController"))
        self.assertTrue(hasattr(safety, "SafeFallbackController"))
        self.assertTrue(hasattr(covariance, "HorizonCovariancePropagator"))
        self.assertTrue(hasattr(risk_budget, "allocate_risk_budget"))
        self.assertIsInstance(QuadrotorDynamics, type)
        self.assertIsInstance(EllipsoidalObstacle, type)
        self.assertIsInstance(ObstacleManager, type)
        self.assertIsInstance(UncertaintyPropagator, type)
        self.assertIsInstance(VIODriftModel, type)
        self.assertTrue(callable(symmetric_matrix_sqrt))

    def test_monte_carlo_factory_builds_from_canonical_nmpc(self):
        source = (
            PACKAGE_ROOT / "application" / "validation" / "monte_carlo.py"
        ).read_text(encoding="utf-8")
        factory = source.split("def _build_controller")[1].split("\n\n")[0]
        self.assertIn("quadrotor_mpc.control.nmpc.chance_constrained", factory)
        self.assertIn("quadrotor_mpc.control.nmpc.deterministic", factory)
        self.assertIn("quadrotor_mpc.control.nmpc.safety", factory)
        self.assertNotIn("control.ccmpc", factory)

    def test_adaptive_config_does_not_point_to_reference_package(self):
        for path in (CODE_ROOT / "config").glob("*.yaml"):
            if "adaptive" not in path.name and "adaptive" not in path.read_text(
                encoding="utf-8"
            ):
                continue
            self.fail(f"adaptive config must not exist yet: {path.name}")


if __name__ == "__main__":
    unittest.main()
