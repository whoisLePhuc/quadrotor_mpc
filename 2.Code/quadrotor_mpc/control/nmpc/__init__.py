"""Canonical native MuJoCo CC-NMPC pipeline (Priority 1 validation target).

This package implements the 13-state quaternion NMPC controller, horizon
covariance propagation, spherical chance-constraint tightening, uniform
joint-risk allocation and the safety supervisor used by the native MuJoCo
runtime and the paired Monte Carlo baseline.  It is the canonical pipeline
for the upcoming Adaptive Geometry-Aware Risk Allocation phase.

Concrete controllers live in dedicated modules so importing a small numerical
helper does not initialize do-mpc or CasADi.
"""
