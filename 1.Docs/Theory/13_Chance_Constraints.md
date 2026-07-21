---
title: 'Chương 13: Chance Constraints'
chapter: 13
tags:
- ccmpc
- probability
- gaussian
- chance-constraints
phase: control-theory
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 13
- Chance Constraints
- Ch.13
---

## 13.1 Introduction

A **chance constraint** is a probabilistic inequality of the form:

$$\mathbb{P}(g(\mathbf{x}) \leq 0) \geq 1 - \delta$$

where $\mathbf{x}$ is a random variable, $g(\cdot)$ defines the constraint, and $\delta \in (0, 0.5)$ is the **violation probability threshold**.

**Meaning**: The probability that the constraint $g(\mathbf{x}) \leq 0$ is satisfied must be at least $1 - \delta$.

In collision avoidance: $g(\mathbf{x})$ defines whether a collision occurs, and $\delta$ is the acceptable collision probability per time step (typically $\delta = 0.03$, i.e., 97% confidence).

## 13.2 Gaussian Linear Chance Constraints

### Lemma 1: Probability Computation

Given $\mathbf{x} \sim \mathcal{N}(\hat{\mathbf{x}}, \boldsymbol{\Sigma})$, the probability of a linear inequality is:

$$\mathbb{P}(\mathbf{a}^T\mathbf{x} \leq b) = \frac{1}{2} + \frac{1}{2}\text{erf}\left(\frac{b - \mathbf{a}^T\hat{\mathbf{x}}}{\sqrt{2\mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a}}}\right)$$

where $\text{erf}(x) = \frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2}dt$ is the standard error function.

**Derivation**: Since $\mathbf{x}$ is Gaussian, $\mathbf{a}^T\mathbf{x}$ is a scalar Gaussian:
- Mean: $\mu = \mathbf{a}^T\hat{\mathbf{x}}$
- Variance: $\sigma^2 = \mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a}$

Then $\mathbb{P}(\mathbf{a}^T\mathbf{x} \leq b) = \Phi\left(\frac{b - \mu}{\sigma}\right)$ where $\Phi$ is the standard normal CDF. Using $\Phi(z) = \frac{1}{2} + \frac{1}{2}\text{erf}(z/\sqrt{2})$ gives the result.

### Lemma 2: Deterministic Reformulation

$$\mathbb{P}(\mathbf{a}^T\mathbf{x} \leq b) \leq \delta \iff \mathbf{a}^T\hat{\mathbf{x}} - b \geq \text{erf}^{-1}(1-2\delta)\sqrt{2\mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a}}$$

where $\text{erf}^{-1}(\cdot)$ is the inverse error function.

**Derivation**: Starting from Lemma 1, we want $\mathbb{P}(\mathbf{a}^T\mathbf{x} \leq b) \leq \delta$:

$$\frac{1}{2} + \frac{1}{2}\text{erf}\left(\frac{b - \mathbf{a}^T\hat{\mathbf{x}}}{\sqrt{2\mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a}}}\right) \leq \delta$$

Solving for the erf argument:

$$\text{erf}\left(\frac{b - \mathbf{a}^T\hat{\mathbf{x}}}{\sqrt{2\mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a}}}\right) \leq 2\delta - 1$$

Since $\delta < 0.5$, we have $2\delta - 1 < 0$. Taking $\text{erf}^{-1}$ (odd function):

$$\frac{b - \mathbf{a}^T\hat{\mathbf{x}}}{\sqrt{2\mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a}}} \leq \text{erf}^{-1}(2\delta - 1) = -\text{erf}^{-1}(1-2\delta)$$

Rearranging:

$$\mathbf{a}^T\hat{\mathbf{x}} - b \geq \text{erf}^{-1}(1-2\delta)\sqrt{2\mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a}}$$

## 13.3 Interpretation

The deterministic constraint has the form:

$$\underbrace{\mathbf{a}^T\hat{\mathbf{x}} - b}_{\text{nominal margin}} \geq \underbrace{c(\delta) \cdot \sigma}_{\text{uncertainty margin}}$$

where:
- $\mathbf{a}^T\hat{\mathbf{x}} - b$: How far the mean is from the constraint boundary
- $\sigma = \sqrt{2\mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a}}$: Scaled standard deviation along direction $\mathbf{a}$
- $c(\delta) = \text{erf}^{-1}(1-2\delta)$: Confidence level factor

### Confidence Level Factors

| $\delta$ | Confidence | $1-2\delta$ | $\text{erf}^{-1}(1-2\delta)$ | $\approx$ |
|----------|------------|-------------|-------------------------------|----------|
| 0.50 | 50% | 0.00 | 0.000 | 0σ |
| 0.16 | 84% | 0.68 | 0.703 | 0.99σ |
| 0.05 | 95% | 0.90 | 1.163 | 1.64σ |
| **0.03** | **97%** | **0.94** | **1.3299** | **1.88σ** |
| 0.01 | 99% | 0.98 | 1.645 | 2.33σ |
| 0.003 | 99.7% | 0.994 | 1.943 | 2.75σ |
| 0.00135 | 99.865% | 0.9973 | 2.121 | 3.00σ |

> [!important] 
> For a one-dimensional one-sided projection, $\delta=0.03$ corresponds to $1.88\sigma$, and a one-sided $3\sigma$ tail corresponds to $\delta\approx0.00135$. A three-dimensional Mahalanobis ellipsoid of radius 3 contains about 97.1% probability, however, so the papers' phrase “3σ confidence ellipsoid” is not the same as a one-dimensional $3\sigma$ half-space quantile.

## 13.4 Why Not 3σ?

The meaning of “3σ” depends on dimension and geometry. For a 3D Gaussian, a Mahalanobis-radius-3 ellipsoid has coverage $P(\chi_3^2\leq9)\approx0.9707$. Bounding-volume approaches use that geometric confidence region and can be more conservative than a locally oriented half-space bound.

The bounding-volume approach is:
- **Overly conservative**: In cluttered environments, the inflated volumes make the problem infeasible
- **Computationally cheaper**: No erf computation needed — just check geometric intersection
- **Compared with a tighter local bound**: The CC approach evaluates the Gaussian probability of a containing half-space, yielding an upper bound rather than the exact ellipsoidal collision probability

The experimental results (Zhu & Alonso-Mora, Table II; 50 trials per method and noise level) show:
- Deterministic MPC: 64% success rate at moderate noise
- Bounding volume (3σ): 100% reported success rate at moderate noise, with longer trajectories
- CC-MPC ($\delta = 0.03$): 100% reported success rate at moderate noise, with shorter trajectories

These finite-trial success rates are empirical results, not proofs of zero collision risk.

## 13.5 The Inverse Error Function

$\text{erf}^{-1}(x)$ is not available in all math libraries. Implementation approaches:

### Newton's Method (used in the codebase)

```python
def erfinv(y, tol=1e-12):
 if abs(y) == 1: return copysign(inf, y)
 # Winitzki 2008 initial guess
 a = 0.147
 t = 2/(pi*a) + log(1 - y*y)/2
 x = sign(y) * sqrt(sqrt(t*t - log(1-y*y)/a) - t)
 # Newton iteration
 for _ in range(50):
 fx = erf(x) - y
 if abs(fx) < tol: break
 x -= fx / (2/sqrt(pi) * exp(-x*x))
 return x
```

### Common Values

```python
import scipy.special as sp
sp.erfinv(0.94) # → 1.32992 (for δ = 0.03)
sp.erfinv(0.90) # → 1.1631 (for δ = 0.05)
```

## 13.6 Vector Form for Obstacle Avoidance

In the obstacle avoidance context, the chance constraint takes the specific form:

$$\mathbb{P}\left(\|\mathbf{p}_i - \mathbf{p}_o\|_{\boldsymbol{\Omega}} \leq 1\right) \leq \delta$$

where the papers use $\|\mathbf{p}\|_{\boldsymbol{\Omega}} := \mathbf{p}^T\boldsymbol{\Omega}\mathbf{p}$ for a weighted squared norm. Under the conventional norm definition one would write $\sqrt{\mathbf{p}^T\boldsymbol{\Omega}\mathbf{p}}$; both give the same boundary when compared with 1.

This is **not** linear in $\mathbf{p}_i - \mathbf{p}_o$ (it's quadratic), so Lemma 2 cannot be directly applied. We must first **linearize** the collision condition.

## 13.7 Linearization of Collision Condition

### Step 1: Affine Transformation to Unit Sphere

Choose $\mathbf{U}$ such that $\mathbf{U}^T\mathbf{U}=\boldsymbol{\Omega}$ and apply $\tilde{\mathbf{p}}=\mathbf{U}\mathbf{p}$. If $\mathbf{L}=\operatorname{chol}(\boldsymbol{\Omega})$ is lower triangular with $\mathbf{L}\mathbf{L}^T=\boldsymbol{\Omega}$, then $\mathbf{U}=\mathbf{L}^T$:

$$\|\mathbf{p}_i - \mathbf{p}_o\|_{\boldsymbol{\Omega}} \leq 1 \iff \|\tilde{\mathbf{p}}_i - \tilde{\mathbf{p}}_o\| \leq 1$$

Under this transformation, the Gaussian distributions become:

$$\begin{aligned}
\tilde{\mathbf{p}}_i &\sim \mathcal{N}(\mathbf{U}\hat{\mathbf{p}}_i,\; \mathbf{U}\boldsymbol{\Sigma}_i\mathbf{U}^T) \\
\tilde{\mathbf{p}}_o &\sim \mathcal{N}(\mathbf{U}\hat{\mathbf{p}}_o,\; \mathbf{U}\boldsymbol{\Sigma}_o\mathbf{U}^T)
\end{aligned}$$

### Step 2: Linearize Unit Sphere to Half-Space

The sphere $\|\tilde{\mathbf{p}}_i - \tilde{\mathbf{p}}_o\| \leq 1$ is approximated by the half-space:

$$\tilde{\mathcal{C}} = \{\mathbf{x} \mid \mathbf{n}^T(\tilde{\mathbf{p}}_i - \tilde{\mathbf{p}}_o) \leq 1\}$$

where $\mathbf{n} = \frac{\hat{\tilde{\mathbf{p}}}_i - \hat{\tilde{\mathbf{p}}}_o}{\|\hat{\tilde{\mathbf{p}}}_i - \hat{\tilde{\mathbf{p}}}_o\|}$ is the unit vector from obstacle to robot.

**Key property**: $\mathcal{C} \subset \tilde{\mathcal{C}}$, therefore $\mathbb{P}(\mathbf{x} \in \mathcal{C}) \leq \mathbb{P}(\mathbf{x} \in \tilde{\mathcal{C}})$. Using the half-space gives an **upper bound** on collision probability.

### Step 3: Apply Lemma 2

Now we have a linear constraint $\mathbf{n}^T(\tilde{\mathbf{p}}_i - \tilde{\mathbf{p}}_o) \leq 1$, which is in the form $\mathbf{a}^T\mathbf{x} \leq b$:

- $\mathbf{a} = \mathbf{n}$ (unit normal in transformed space)
- $\mathbf{x} = \tilde{\mathbf{p}}_i - \tilde{\mathbf{p}}_o$ (relative position)
- $b = 1$ (unit sphere radius)

Applying Lemma 2:

$$\mathbf{n}^T(\hat{\tilde{\mathbf{p}}}_i - \hat{\tilde{\mathbf{p}}}_o) - 1 \geq \text{erf}^{-1}(1-2\delta)\sqrt{2\mathbf{n}^T(\tilde{\boldsymbol{\Sigma}}_i + \tilde{\boldsymbol{\Sigma}}_o)\mathbf{n}}$$

### Step 4: Transform Back to Original Space

Substituting $\tilde{\mathbf{p}}=\mathbf{U}\mathbf{p}$ and $\tilde{\boldsymbol{\Sigma}}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{U}^T$:

$$\boxed{\mathbf{n}^T\mathbf{U}(\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_o) - 1 \geq z_{1-\delta}\sqrt{\mathbf{n}^T\mathbf{U}(\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_o)\mathbf{U}^T\mathbf{n}}}$$

where

$$\boxed{\mathbf{n}=\frac{\mathbf{U}(\hat{\mathbf{p}}_i-\hat{\mathbf{p}}_o)}{\|\mathbf{U}(\hat{\mathbf{p}}_i-\hat{\mathbf{p}}_o)\|}},\qquad z_{1-\delta}=\sqrt{2}\,\operatorname{erf}^{-1}(1-2\delta).$$

This is the tight transformed-space normal from the 2019 derivation. Equation (16) of the 2020 paper writes the normal using the untransformed mean direction; that alternative remains a containing-half-space construction for a unit vector but is generally less tight for anisotropic ellipsoids. Do not mix its normal definition with the 2019 derivation without labeling the variant.

## 13.8 Verification (Python)

The following code verifies Lemmas 1 and 2 numerically:

```python
import numpy as np
from scipy.special import erf, erfinv

# Setup
np.random.seed(42)
mu = np.array([1.0, 2.0])
Sigma = np.diag([0.3, 0.5])**2
a = np.array([1.0, -1.0])
b = -0.5

# Lemma 1: Probability computation
mu_proj = a @ mu
sigma_proj = np.sqrt(a @ Sigma @ a)
p_analytic = 0.5 + 0.5 * erf((b - mu_proj) / (np.sqrt(2) * sigma_proj))
# Monte Carlo verification
samples = np.random.multivariate_normal(mu, Sigma, 500_000)
p_mc = np.mean(samples @ a <= b)
# |p_analytic - p_mc| < 0.005 ✓

# Lemma 2: Deterministic reformulation
delta = 0.05
c = erfinv(1 - 2*delta) * np.sqrt(2) * sigma_proj
# c ≈ 0.959 ≈ 1.64σ ✓
```

## 13.9 Practical Considerations

### Choice of δ

- **Too small (δ < 0.01)**: Overly conservative, may cause infeasibility
- **Too large (δ > 0.1)**: Insufficient safety margin
- **Paper setting**: δ = 0.03 in both experimental studies. It is not a universal recommendation; the application should allocate and validate risk at the system level.

### Receding-Horizon Justification

The chance constraint is applied per step, not over the whole trajectory. Fast replanning provides an engineering rationale, but it does not make the per-step constraint equal to a whole-trajectory risk guarantee:
1. The MPC re-plans at every control cycle (~16 Hz)
2. At each re-plan, the initial state is updated with new measurements
3. For one constrained collision event at each of $N$ steps, the union bound gives at most $\min(1,N\delta)$; with multiple robots or obstacles, sum the allocated risks over all relevant events
4. With discounted chance constraints (Eq. 17 in Zhu & Alonso-Mora), early steps are weighted more heavily

### Discounted Chance Constraints

$$\sum_{k=1}^{N} \gamma^k \mathbb{P}(\mathbf{x}_k \in \mathcal{C}_k) \leq \delta_o$$

where $\gamma \in (0, 1)$ is a discount factor. Our per-step constraint $\mathbb{P}(\mathbf{x}_k \in \mathcal{C}_k) \leq \delta_o$ guarantees this when $\gamma < 0.5$ (Lemma 3 in Zhu & Alonso-Mora, 2019).

## 13.10 Required Numerical Verification

After implementing the corrected transformation, the verification suite should check:

- **Lemma 1**: Analytical half-space probability matches Monte Carlo sampling.
- **Lemma 2**: The deterministic inequality bounds the sampled half-space violation rate.
- **Rotated ellipsoids**: Verify $\|\mathbf{U}\mathbf{d}\|^2=\mathbf{d}^T\boldsymbol{\Omega}\mathbf{d}$ for non-axis-aligned obstacles.
- **Collision bound**: Monte Carlo estimates of true ellipsoid collision probability do not exceed the half-space bound within sampling tolerance.
- **Regression**: Axis-aligned and rotated cases produce consistent results.

Previous “13/13 tests passing” claims must be rerun after the Cholesky correction before being treated as validation evidence.

## 13.11 Prerequisites and Related Chapters

> [!info] Prerequisites
- Probability theory (Gaussian distributions)
- Linear algebra (quadratic forms, Cholesky)

> [!info] Used In
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — Core constraint formulation
- [[15_Obstacle_Avoidance|Ch.15: Obstacle Avoidance]] — Collision probability computation

> [!info] See Also
- [[14_Covariance_Propagation|Ch.14: Covariance Propagation]] — Computing Σ at each step
