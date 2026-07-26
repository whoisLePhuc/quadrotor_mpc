"""Native spherical chance-constrained NMPC controller."""

from __future__ import annotations

from deterministic_nmpc_controller import DeterministicNMPCController
from native_chance_constraints import ChanceConstraintOptions


class SphericalChanceConstrainedNMPCController(DeterministicNMPCController):
    """Explicit controller type for the Stage 4 CC-NMPC workflow."""

    def __init__(
        self,
        *,
        chance_options: ChanceConstraintOptions,
        **kwargs,
    ):
        if not chance_options.enabled:
            raise ValueError(
                "SphericalChanceConstrainedNMPCController requires enabled chance constraints"
            )
        super().__init__(chance_options=chance_options, **kwargs)
