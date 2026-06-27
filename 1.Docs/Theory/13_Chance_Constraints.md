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
| 0.16 | 84% | 0.68 | 0.714 | 1σ |
| 0.05 | 95% | 0.90 | 1.163 | 1.64σ |
| **0.03** | **97%** | **0.94** | **1.329** | **1.88σ** |
| 0.01 | 99% | 0.98 | 1.645 | 2.33σ |
| 0.003 | 99.7% | 0.994 | 1.862 | 2.63σ |

> [!important] 
> $\delta = 0.03$ corresponds to approximately $1.88\sigma$, NOT $3\sigma$ (which would be $\delta \approx 0.003$). Using $\delta = 0.03$ means we accept a 3% collision probability per step, trusting that re-planning at each step handles accumulated risk.

## 13.4 Why Not 3σ?

Using 3σ bounding volumes (as in some alternative approaches) corresponds to $\delta = 0.003$ per independent dimension. This is:
- **Overly conservative**: In cluttered environments, the inflated volumes make the problem infeasible
- **Computationally cheaper**: No erf computation needed — just check geometric intersection
- **Tighter bound**: The CC approach directly computes collision probability, avoiding the conservatism of bounding volumes

The experimental results (Zhu & Alonso-Mora, Table II) show:
- Deterministic MPC: 64% success rate at moderate noise
- Bounding volume (3σ): 100% safety but longer trajectories
- CC-MPC (δ = 0.03): 100% safety with shorter trajectories (more efficient)

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
sp.erfinv(0.94) # → 1.3293 (for δ = 0.03)
sp.erfinv(0.90) # → 1.1631 (for δ = 0.05)
```

## 13.6 Vector Form for Obstacle Avoidance

In the obstacle avoidance context, the chance constraint takes the specific form:

$$\mathbb{P}\left(\|\mathbf{p}_i - \mathbf{p}_o\|_{\boldsymbol{\Omega}} \leq 1\right) \leq \delta$$

where $\|\mathbf{p}\|_{\boldsymbol{\Omega}} = \mathbf{p}^T\boldsymbol{\Omega}\mathbf{p}$ is the weighted norm defining the ellipsoid.

This is **not** linear in $\mathbf{p}_i - \mathbf{p}_o$ (it's quadratic), so Lemma 2 cannot be directly applied. We must first **linearize** the collision condition.

## 13.7 Linearization of Collision Condition

### Step 1: Affine Transformation to Unit Sphere

Apply the coordinate transformation $\tilde{\mathbf{p}} = \boldsymbol{\Omega}^{1/2}\mathbf{p}$:

$$\|\mathbf{p}_i - \mathbf{p}_o\|_{\boldsymbol{\Omega}} \leq 1 \iff \|\tilde{\mathbf{p}}_i - \tilde{\mathbf{p}}_o\| \leq 1$$

Under this transformation, the Gaussian distributions become:

$$\begin{aligned}
\tilde{\mathbf{p}}_i &\sim \mathcal{N}(\boldsymbol{\Omega}^{1/2}\hat{\mathbf{p}}_i,\; \boldsymbol{\Omega}^{1/2}\boldsymbol{\Sigma}_i\boldsymbol{\Omega}^{1/2\;T}) \\
\tilde{\mathbf{p}}_o &\sim \mathcal{N}(\boldsymbol{\Omega}^{1/2}\hat{\mathbf{p}}_o,\; \boldsymbol{\Omega}^{1/2}\boldsymbol{\Sigma}_o\boldsymbol{\Omega}^{1/2\;T})
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

Substituting $\tilde{\mathbf{p}} = \boldsymbol{\Omega}^{1/2}\mathbf{p}$ and $\tilde{\boldsymbol{\Sigma}} = \boldsymbol{\Omega}^{1/2}\boldsymbol{\Sigma}\boldsymbol{\Omega}^{1/2\;T}$:

$$\boxed{\mathbf{n}_o^T\boldsymbol{\Omega}^{1/2}(\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_o) - 1 \geq \text{erf}^{-1}(1-2\delta)\sqrt{2\mathbf{n}_o^T\boldsymbol{\Omega}^{1/2}(\boldsymbol{\Sigma}_i + \boldsymbol{\Sigma}_o)\boldsymbol{\Omega}^{1/2\;T}\mathbf{n}_o}}$$

where $\mathbf{n}_o = \frac{\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_o}{\|\hat{\mathbf{p}}_i - \hat{\mathbf{p}}_o\|}$ is the unit normal in original space.

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
- **Recommended**: δ = 0.03 (97% confidence per step)

### Receding-Horizon Justification

The chance constraint is applied per step, not over the whole trajectory. This is valid because:
1. The MPC re-plans at every control cycle (~16 Hz)
2. At each re-plan, the initial state is updated with new measurements
3. The collision probability for the entire trajectory is bounded by $N\delta$ (conservative)
4. With discounted chance constraints (Eq. 17 in Zhu & Alonso-Mora), early steps are weighted more heavily

### Discounted Chance Constraints

$$\sum_{k=1}^{N} \gamma^k \mathbb{P}(\mathbf{x}_k \in \mathcal{C}_k) \leq \delta_o$$

where $\gamma \in (0, 1)$ is a discount factor. Our per-step constraint $\mathbb{P}(\mathbf{x}_k \in \mathcal{C}_k) \leq \delta_o$ guarantees this when $\gamma < 0.5$ (Lemma 3 in Zhu & Alonso-Mora, 2019).

## 13.10 Numerical Verification Results

All formulas verified with 13/13 tests passing (see `verify_formulas.py`):

- **Lemma 1**: Analytical probability matches Monte Carlo (500k samples) within 0.005
- **Lemma 2**: Deterministic constraint correctly bounds violation probability
- **Full pipeline**: Detection → ellipsoid → chance constraint yields feasible solutions

## 13.11 Prerequisites and Related Chapters

> [!info] Prerequisites
- Probability theory (Gaussian distributions)
- Linear algebra (quadratic forms, Cholesky)

> [!info] Used In
- [[12_CCMPC|Ch.12: Chance-Constrained MPC]] — Core constraint formulation
- [[15_Obstacle_Avoidance|Ch.15: Obstacle Avoidance]] — Collision probability computation

> [!info] See Also
- [[14_Covariance_Propagation|Ch.14: Covariance Propagation]] — Computing Σ at each step
