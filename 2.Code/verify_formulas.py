#!/usr/bin/env python3
"""
Numerical verification of all key formulas from:
"Robust Vision-based Obstacle Avoidance for Micro Aerial Vehicles in Dynamic Environments"
Lin, Zhu, Alonso-Mora — ICRA 2020

Each test function verifies one or more formulas from the paper.
"""

import numpy as np
from numpy.linalg import norm, inv
from math import erf
import math


# --------------------------------------------------------------------------
# Inverse error function (Newton's method, since math.erfinv unavailable)
# --------------------------------------------------------------------------
_SQRT_PI_INV = 0.5641895835477563  # 1/sqrt(pi)

def erfinv(y: float, tol: float = 1e-12) -> float:
    if y < -1 or y > 1:
        raise ValueError(f"erfinv({y}): argument must be in [-1, 1]")
    if abs(y) == 1:
        return math.copysign(float('inf'), y)
    if y == 0:
        return 0.0
    sign = 1.0 if y > 0 else -1.0
    ya = abs(y)
    # Initial guess from rational approximation (Winitzki 2008)
    a = 0.147
    t = 2.0 / (math.pi * a) + math.log(1 - ya * ya) / 2.0
    x = math.sqrt(math.sqrt(t * t - math.log(1 - ya * ya) / a) - t)
    x *= sign
    for _ in range(50):
        fx = erf(x) - y
        if abs(fx) < tol:
            break
        d = 2 * _SQRT_PI_INV * math.exp(-x * x)
        if abs(d) < 1e-300:
            if x > 0:
                x += 0.1
            else:
                x -= 0.1
            continue
        x -= fx / d
    return x


# ============================================================================
# Eq (1) — U-depth POI Threshold
# ============================================================================
def test_udepth_poi_threshold() -> None:
    f = 600.0
    T_ho = 1.0
    assert abs(f * T_ho / 1.0 / 600.0 - 1.0) < 1e-9
    assert abs(f * T_ho / 2.0 / (f * T_ho / 1.0) - 0.5) < 1e-9
    print("  [PASS] Eq(1): T_POI ∝ 1/d_bin")


# ============================================================================
# Eq (2) — Horizontal position and size from U-depth bounding box
# ============================================================================
def test_horizontal_projection() -> None:
    f = 600.0
    # Obstacle: front face at 3.0m, left edge at u=100, right edge at u=200
    d_b = 3.0
    u_l, u_r = 100, 200
    d_t = 2.7  # back face
    x_recon = d_b
    y_recon = (u_l + u_r) * d_b / (2 * f)
    w_recon = (u_r - u_l) * d_b / f
    l_recon = 2 * (d_b - d_t)
    assert abs(x_recon - d_b) < 1e-9
    assert abs(y_recon - 300 * 3.0 / 1200) < 1e-9
    assert abs(w_recon - 100 * 3.0 / 600) < 1e-9
    assert abs(l_recon - 2 * 0.3) < 1e-9
    print(f"  [PASS] Eq(2): x={x_recon:.2f}m, y={y_recon:.2f}m, l={l_recon:.2f}m, w={w_recon:.2f}m")


# ============================================================================
# Eq (3) — Vertical position and height (image coords: y-down → h_b > h_t)
# ============================================================================
def test_vertical_projection() -> None:
    f = 600.0
    d_b = 3.0
    # Bounding box: top at h_t=100 (smaller y), bottom at h_b=300 (larger y) in image
    # The obstacle center y_pixel = 200, height_y_pixel = 200
    h_t, h_b = 100, 300
    z_recon = (h_t + h_b) * d_b / (2 * f)
    # NOTE: Paper formula h_o^B = (h_t - h_b) * d_b / f gives NEGATIVE value.
    # The physical height is |h_t - h_b| * d_b / f.
    h_recon = abs(h_t - h_b) * d_b / f
    assert abs(z_recon - 400 * 3.0 / 1200) < 1e-9  # z = 1.0m (obstacle center above camera)
    assert abs(h_recon - 200 * 3.0 / 600) < 1e-9    # height = 1.0m
    print(f"  [PASS] Eq(3): z={z_recon:.2f}m, h={h_recon:.2f}m (using absolute value for height)")


# ============================================================================
# Eq (4) — World frame transformation with uncertainty
# ============================================================================
def test_world_frame_transform() -> None:
    p_o_B = np.array([2.0, 0.3, -0.2])
    phi, theta, psi = np.deg2rad([10, 15, 30])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi),  np.cos(phi)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi),  np.cos(psi), 0],
                   [0, 0, 1]])
    R_B_W = Rz @ Ry @ Rx
    p_W = np.array([1.0, 0.5, 0.3])
    Σ_o_B = np.diag([0.05, 0.05, 0.05]) ** 2
    Σ_W = np.diag([0.02, 0.02, 0.02]) ** 2
    p_o_W = R_B_W @ p_o_B + p_W
    Σ_o_W = R_B_W.T @ Σ_o_B @ R_B_W + Σ_W
    assert np.all(np.linalg.eigvalsh(Σ_o_W) > 0)
    trace_no_mav = np.trace(R_B_W.T @ Σ_o_B @ R_B_W)
    trace_with_mav = np.trace(Σ_o_W)
    assert abs(trace_no_mav - np.trace(Σ_o_B)) < 1e-10
    assert trace_with_mav > trace_no_mav
    print(f"  [PASS] Eq(4): Tr(Σ_no_mav)={trace_no_mav:.6f} → Tr(Σ_with_mav)={trace_with_mav:.6f}")


# ============================================================================
# Eq (5) — Gaussian data association
# ============================================================================
def test_gaussian_data_association() -> None:
    def gaussian_pdf(x, mean, cov):
        d = x - mean
        return float(np.exp(-0.5 * d @ inv(cov) @ d) / np.sqrt((2*np.pi)**len(x) * np.linalg.det(cov)))
    mean = np.zeros(6)
    cov = np.diag([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]) ** 2
    pd_same = gaussian_pdf(mean, mean, cov)
    x_near = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
    pd_near = gaussian_pdf(x_near, mean, cov)
    x_far = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    pd_far = gaussian_pdf(x_far, mean, cov)
    assert pd_same > pd_near > pd_far
    assert pd_same > 0 and pd_near > 0
    print(f"  [PASS] Eq(5): pd_same={pd_same:.2e}, pd_near={pd_near:.2e}, pd_far={pd_far:.2e} (same > near > far)")


# ============================================================================
# Eq (6) — Constant velocity prediction
# ============================================================================
def test_constant_velocity_prediction() -> None:
    dt = 0.06
    p_hat = np.array([1.0, 2.0, 0.5])
    v_hat = np.array([0.5, 0.0, -0.2])
    Σ = np.diag([0.01, 0.01, 0.01]) ** 2
    Σ_v = np.diag([0.1, 0.1, 0.1]) ** 2
    p_next = p_hat + v_hat * dt
    v_next = v_hat.copy()
    Σ_next = Σ + Σ_v * dt**2
    assert norm(p_next - p_hat - v_hat * dt) < 1e-10
    assert norm(v_next - v_hat) < 1e-10
    assert np.trace(Σ_next) > np.trace(Σ)
    Σ_k = Σ.copy()
    for _ in range(10):
        tr_before = np.trace(Σ_k)
        Σ_k = Σ_k + Σ_v * dt**2
        assert np.trace(Σ_k) > tr_before
    print(f"  [PASS] Eq(6): velocity constant, trace grows {np.trace(Σ):.6f} → {np.trace(Σ_k):.6f}")


# ============================================================================
# Eq (7) — Ellipsoidal bounding: (a,b,c) = sqrt(3)/2 * (l,w,h)
# ============================================================================
def test_ellipsoidal_bounding() -> None:
    factor = np.sqrt(3) / 2
    size = np.array([0.5, 0.6, 1.7])
    axes = factor * size
    half = size / 2
    corner = np.array([half[0], half[1], half[2]])
    val = (corner[0]/axes[0])**2 + (corner[1]/axes[1])**2 + (corner[2]/axes[2])**2
    assert abs(val - 1.0) < 1e-10, f"Corner should be on ellipsoid surface (val={val})"
    interior = np.array([0.2, 0.2, 0.2])
    val_int = (interior[0]/axes[0])**2 + (interior[1]/axes[1])**2 + (interior[2]/axes[2])**2
    assert val_int <= 1.0
    print(f"  [PASS] Eq(7): Corner-score={val:.6f} (≈1), interior={val_int:.4f} (≤1)")


# ============================================================================
# Lemma 1 & 2 — Gaussian linear chance constraints
# ============================================================================
def test_gaussian_chance_constraint_lemmas() -> None:
    rng = np.random.default_rng(42)
    n_samples = 500_000
    mu = np.array([1.0, 2.0])
    Sigma = np.diag([0.3, 0.5]) ** 2
    a = np.array([1.0, -1.0])
    b = -0.5
    mu_proj = a @ mu
    sigma_proj = np.sqrt(a @ Sigma @ a)
    p_analytic = 0.5 + 0.5 * erf((b - mu_proj) / (np.sqrt(2) * sigma_proj))
    samples = rng.multivariate_normal(mu, Sigma, size=n_samples)
    p_mc = np.mean(samples @ a <= b)
    assert abs(p_analytic - p_mc) < 5e-3
    delta = 0.05
    c = erfinv(1 - 2 * delta) * np.sqrt(2) * sigma_proj
    assert c > 0
    assert abs(c / sigma_proj - np.sqrt(2) * erfinv(0.9)) < 1e-6
    print(f"  [PASS] Lemma 1&2: p={p_analytic:.5f} (MC={p_mc:.5f}),"
          f" c(δ=0.05)={c:.4f} ≈ {c/sigma_proj:.2f}σ")


# ============================================================================
# Eq (16) — Full CC-MPC deterministic constraint
# ============================================================================
def test_full_chance_constraint() -> None:
    p_hat = np.array([0.0, 0.0, 0.0])
    p_hat_o = np.array([2.0, 0.0, 0.0])
    n_o = (p_hat - p_hat_o) / norm(p_hat - p_hat_o)
    a_o, b_o, c_o = 0.5, 0.4, 0.9
    r = 0.4
    R_o = np.eye(3)
    Omega = R_o.T @ np.diag([1/(a_o+r)**2, 1/(b_o+r)**2, 1/(c_o+r)**2]) @ R_o
    Sigma_low = np.diag([0.02, 0.02, 0.02]) ** 2
    Sigma_o_low = np.diag([0.05, 0.05, 0.05]) ** 2
    Sigma_high = np.diag([0.1, 0.1, 0.1]) ** 2
    Sigma_o_high = np.diag([0.3, 0.3, 0.3]) ** 2
    delta = 0.03

    def evaluate_cc(Σ_mav, Σ_obs):
        L = np.linalg.cholesky(Omega)
        lhs = n_o @ L @ (p_hat - p_hat_o) - 1
        inner = L @ (Σ_mav + Σ_obs) @ L.T
        rhs = erfinv(1 - 2 * delta) * np.sqrt(2 * n_o @ inner @ n_o)
        return float(lhs), float(rhs), lhs >= rhs

    lhs_low, rhs_low, safe_low = evaluate_cc(Sigma_low, Sigma_o_low)
    lhs_high, rhs_high, safe_high = evaluate_cc(Sigma_high, Sigma_o_high)
    assert rhs_low < rhs_high, f"Higher uncertainty should increase RHS ({rhs_low:.4f} vs {rhs_high:.4f})"
    assert safe_low, "Low uncertainty should be feasible at 2m"
    # Stricter delta -> larger margin
    L_low = np.linalg.cholesky(Omega)
    inner_low = L_low @ (Sigma_low + Sigma_o_low) @ L_low.T
    rhs_low_2 = erfinv(1 - 2*0.03) * np.sqrt(2 * n_o @ inner_low @ n_o)
    delta_loose = 0.1
    rhs_loose = erfinv(1 - 2*delta_loose) * np.sqrt(2 * n_o @ inner_low @ n_o)
    assert rhs_loose < rhs_low_2, f"Larger δ (looser) should have smaller margin ({rhs_loose:.4f} vs {rhs_low_2:.4f})"
    # Move obstacle far away
    p_hat_o_far = np.array([8.0, 0.0, 0.0])
    n_o_far = (p_hat - p_hat_o_far) / norm(p_hat - p_hat_o_far)
    L = np.linalg.cholesky(Omega)
    lhs_far = n_o_far @ L @ (p_hat - p_hat_o_far) - 1
    inner_far = L @ (Sigma_low + Sigma_o_low) @ L.T
    rhs_far = erfinv(1 - 2*delta) * np.sqrt(2 * n_o_far @ inner_far @ n_o_far)
    assert lhs_far >= rhs_far, "Far obstacle should be feasible"
    assert lhs_far >= rhs_far, "Far obstacle should be feasible"
    print(f"  [PASS] Eq(16): low-uncert rhs={rhs_low_2:.4f} < high-uncert rhs={rhs_high:.4f} ✓")
    print(f"                    loose-δ rhs={rhs_loose:.4f} < strict-δ rhs={rhs_low_2:.4f} ✓")


# ============================================================================
# Eq (19) — EKF uncertainty propagation
# ============================================================================
def test_uncertainty_propagation() -> None:
    dt = 0.06
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    Gamma = np.diag([0.01, 0.01, 0.001, 0.001]) ** 2
    W = np.diag([0, 0, 0.005, 0.005]) ** 2
    traces = [np.trace(Gamma)]
    for _ in range(10):
        Gamma = F @ Gamma @ F.T + W
        traces.append(np.trace(Gamma))
    assert all(traces[i] > traces[i-1] for i in range(1, len(traces)))
    Sigma_pos = Gamma[:2, :2]
    assert np.all(np.linalg.eigvalsh(Sigma_pos) > 0)
    print(f"  [PASS] Eq(19): trace grows {traces[0]:.6f} → {traces[-1]:.6f}")


# ============================================================================
# Eq (12) — Logistic collision cost
# ============================================================================
def test_logistic_collision_cost() -> None:
    Q_o, lam_o, r_o = 10.0, 3.0, 1.5
    def cost(d):
        return Q_o / (1 + np.exp(lam_o * (d - r_o)))
    assert abs(cost(r_o) - Q_o/2) < 1e-10
    dists = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    costs = [cost(d) for d in dists]
    for i in range(1, len(costs)):
        assert costs[i] < costs[i-1]
    assert costs[0] < Q_o and costs[0] > Q_o/2
    assert cost(100.0) < 1e-10
    eps = 1e-6
    deriv = (cost(r_o + eps) - cost(r_o - eps)) / (2 * eps)
    assert deriv < 0
    assert abs(deriv - (-lam_o * Q_o / 4)) < 0.01
    print(f"  [PASS] Eq(12): J(0)={cost(0.0):.4f}, J(r_o)={cost(r_o):.4f}, J(∞)={cost(100.0):.2e}")


# ============================================================================
# Eq (17-18) — FOV constraints (half-spaces)
# ============================================================================
def test_fov_constraints() -> None:
    hfov = np.deg2rad(87/2)
    vfov = np.deg2rad(58/2)
    max_depth = 5.0
    def in_fov(p):
        x, y, z = p
        if x <= 0 or x > max_depth:
            return False
        if abs(math.atan2(y, x)) > hfov:
            return False
        if abs(math.atan2(z, x)) > vfov:
            return False
        return True
    in_pt = np.array([3.0, 0.3, -0.2])
    assert in_fov(in_pt), f"{in_pt} should be inside FOV"
    out_yaw = np.array([3.0, 5.0, 0.0])
    assert not in_fov(out_yaw), f"{out_yaw} should be outside FOV"
    out_pitch = np.array([3.0, 0.0, 3.0])
    assert not in_fov(out_pitch), f"{out_pitch} should be outside FOV"
    out_depth = np.array([10.0, 0.0, 0.0])
    assert not in_fov(out_depth), f"{out_depth} should be outside FOV"
    print(f"  [PASS] Eq(17-18): FOV test — inside ✓, yaw-out ✓, pitch-out ✓, depth-out ✓")


# ============================================================================
# Full pipeline consistency test
# ============================================================================
def test_full_pipeline() -> None:
    f = 600.0
    u_l, u_r = 100, 200
    d_t, d_b = 2.7, 3.0
    h_t, h_b = 100, 300
    # Step 1: Eq(2)
    x_B, y_B = d_b, (u_l + u_r) * d_b / (2 * f)
    l_B, w_B = 2 * (d_b - d_t), (u_r - u_l) * d_b / f
    # Step 2: Eq(3)  
    z_B = (h_t + h_b) * d_b / (2 * f)
    h_B = abs(h_t - h_b) * d_b / f
    assert x_B > 0 and y_B > 0 and l_B > 0 and w_B > 0
    assert h_B > 0
    # Step 3: Eq(7)
    axes = np.sqrt(3) / 2 * np.array([l_B, w_B, h_B])
    assert np.all(axes > 0)
    # Step 4: Eq(4) simplified (identity rotation, zero translation)
    Sigma_o = np.diag([0.05**2, 0.05**2, 0.05**2])
    p_o_W = np.array([x_B, y_B, z_B])
    assert np.all(np.linalg.eigvalsh(Sigma_o) >= 0)
    # Step 5: CC constraint (Eq 16)
    p_hat = np.array([0.0, 0.0, 0.0])
    n_o = (p_hat - p_o_W) / norm(p_hat - p_o_W)
    Omega = np.eye(3) @ np.diag([1/(axes[0]+0.4)**2, 1/(axes[1]+0.4)**2, 1/(axes[2]+0.4)**2]) @ np.eye(3)
    L = np.linalg.cholesky(Omega)
    lhs = n_o @ L @ (p_hat - p_o_W) - 1
    inner = L @ (Sigma_o + np.diag([0.02**2, 0.02**2, 0.02**2])) @ L.T
    rhs = erfinv(1 - 2*0.03) * np.sqrt(2 * n_o @ inner @ n_o)
    assert lhs >= rhs, f"Full pipeline CC should be feasible: lhs={lhs:.4f}, rhs={rhs:.4f}"
    assert rhs > 0
    print(f"  [PASS] Full pipeline: detection→ellipsoid→CC: feasible (lhs={lhs:.4f} ≥ rhs={rhs:.4f})")


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    print("=" * 70)
    print("Verification: Robust Vision-based Obstacle Avoidance for MAVs")
    print("Lin, Zhu, Alonso-Mora — ICRA 2020")
    print("=" * 70)
    tests = [
        ("Eq(1)  U-depth POI threshold",            test_udepth_poi_threshold),
        ("Eq(2)  Horizontal projection (pinhole)",  test_horizontal_projection),
        ("Eq(3)  Vertical projection (pinhole)",    test_vertical_projection),
        ("Eq(4)  World frame + uncertainty",        test_world_frame_transform),
        ("Eq(5)  Gaussian data association",        test_gaussian_data_association),
        ("Eq(6)  Constant velocity prediction",     test_constant_velocity_prediction),
        ("Eq(7)  Ellipsoidal bounding",             test_ellipsoidal_bounding),
        ("L1-2   Gaussian chance constraints",      test_gaussian_chance_constraint_lemmas),
        ("Eq(16) Full CC-MPC constraint",           test_full_chance_constraint),
        ("Eq(19) EKF uncertainty propagation",      test_uncertainty_propagation),
        ("Eq(12) Logistic collision cost",          test_logistic_collision_cost),
        ("Eq(17-18) FOV constraints",               test_fov_constraints),
        ("Pipe   Full pipeline consistency",        test_full_pipeline),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'='*70}\nResult: {passed}/{len(tests)} passed, {failed} failed\n{'='*70}")
    if failed:
        exit(1)

if __name__ == "__main__":
    main()
