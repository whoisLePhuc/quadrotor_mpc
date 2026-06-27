---
title: 'Chương 4: Rotation Matrix'
chapter: 4
tags:
- quadrotor
- kinematics
- rotation
- SO3
- linear-algebra
phase: foundations
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 4
- Rotation Matrix
- Ch.4
---

## 4.1 Introduction

A **rotation matrix** $\mathbf{R} \in SO(3)$ represents a rigid-body rotation in 3D space. It satisfies:

$$\mathbf{R}^T\mathbf{R} = \mathbf{I}, \quad \det(\mathbf{R}) = 1$$

Rotation matrices are the fundamental building block for expressing orientation in the quadrotor model.

## 4.2 Elementary Rotations

### Rotation about X-axis (Roll)

$$\mathbf{R}_X(\phi) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\phi & -\sin\phi \\ 0 & \sin\phi & \cos\phi \end{bmatrix}$$

**At $\phi = 0$**: $\mathbf{R}_X(0) = \mathbf{I}$ (no rotation) 
**At $\phi = \pi/2$**: Y-axis rotates to Z-axis

### Rotation about Y-axis (Pitch)

$$\mathbf{R}_Y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}$$

**At $\theta = \pi/2$**: X-axis rotates to Z-axis (nose pointing up)

### Rotation about Z-axis (Yaw)

$$\mathbf{R}_Z(\psi) = \begin{bmatrix} \cos\psi & -\sin\psi & 0 \\ \sin\psi & \cos\psi & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**At $\psi = \pi/2$**: X-axis rotates to Y-axis (nose pointing east)

## 4.3 Composite Rotations (ZYX Euler)

The body-to-world rotation for a quadrotor is:

$$\boxed{\mathbf{R}_B^W = \mathbf{R}_Z(\psi)\mathbf{R}_Y(\theta)\mathbf{R}_X(\phi)}$$

**Order**: Yaw → Pitch → Roll (applied from right to left)

Explicitly:

$$\mathbf{R}_B^W = \begin{bmatrix}
c\psi c\theta & c\psi s\theta s\phi - s\psi c\phi & c\psi s\theta c\phi + s\psi s\phi \\
s\psi c\theta & s\psi s\theta s\phi + c\psi c\phi & s\psi s\theta c\phi - c\psi s\phi \\
-s\theta & c\theta s\phi & c\theta c\phi
\end{bmatrix}$$

where $c\alpha = \cos\alpha$, $s\alpha = \sin\alpha$.

### Small Angle Approximation

For $\phi, \theta \ll 1$ rad and any $\psi$:

$$\mathbf{R}_B^W \approx \begin{bmatrix}
\cos\psi & -\sin\psi & \theta\cos\psi + \phi\sin\psi \\
\sin\psi & \cos\psi & \theta\sin\psi - \phi\cos\psi \\
-\theta & \phi & 1
\end{bmatrix}$$

## 4.4 The 2D Yaw Rotation $\mathbf{R}_Z(\psi)$

The dynamics model uses the 2×2 version extensively:

$$\mathbf{R}_Z(\psi) = \begin{bmatrix} \cos\psi & -\sin\psi \\ \sin\psi & \cos\psi \end{bmatrix}$$

This rotates horizontal vectors from body to world frame:

$$\begin{bmatrix} a_x^W \\ a_y^W \end{bmatrix} = \mathbf{R}_Z(\psi) \begin{bmatrix} a_x^B \\ a_y^B \end{bmatrix}$$

In the velocity dynamics:

$$\begin{bmatrix} \dot{v}_x \\ \dot{v}_y \end{bmatrix} = \mathbf{R}_Z(\psi) \begin{bmatrix} g\tan\theta \\ -g\tan\phi \end{bmatrix} - k_D\begin{bmatrix} v_x \\ v_y \end{bmatrix}$$

**Physical interpretation**: The body-frame accelerations $[g\tan\theta, -g\tan\phi]^T$ (forward and right) are rotated by the yaw angle to get world-frame accelerations.

## 4.5 The FOV Rotation

For FOV constraints, only yaw rotation matters (camera is assumed to look in the horizontal plane):

$$\mathbf{p}^B = \mathbf{R}_Z(\psi)^T (\mathbf{p}^W - \mathbf{p}_{\text{cam}}^W)$$

$$\mathbf{n}^W = \mathbf{R}_Z(\psi) \mathbf{n}^B$$

## 4.6 The Obstacle Rotation $\mathbf{R}_o$

Obstacles have an orientation (yaw angle) in the world frame:

$$\mathbf{R}_o = \mathbf{R}_Z(\psi_o) = \begin{bmatrix} \cos\psi_o & -\sin\psi_o & 0 \\ \sin\psi_o & \cos\psi_o & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

This is used in the collision matrix $\boldsymbol{\Omega}_{io}$:

$$\boldsymbol{\Omega}_{io} = \mathbf{R}_o^T \text{diag}\left(\frac{1}{(a+r)^2}, \frac{1}{(b+r)^2}, \frac{1}{(c+r)^2}\right) \mathbf{R}_o$$

The rotation transforms the diagonal axes matrix from obstacle-aligned coordinates to world coordinates.

## 4.7 Properties of Rotation Matrices

### Inverse = Transpose

$$\mathbf{R}^{-1} = \mathbf{R}^T$$

This means transforming back from world to body is simply:

$$\mathbf{p}^B = \mathbf{R}_B^{W\;T}(\mathbf{p}^W - \mathbf{t})$$

### Orthogonality

$$\mathbf{R}^T\mathbf{R} = \mathbf{I}$$

Each column (and row) is a unit vector, and columns are mutually orthogonal.

### Determinant

$$\det(\mathbf{R}) = 1$$

Ensures right-handed coordinate system is preserved.

### Norm Preservation

$$\|\mathbf{R}\mathbf{x}\| = \|\mathbf{x}\|$$

Rotations preserve distances — a sphere stays a sphere under rotation.

### Covariance Transformation

For a Gaussian random variable $\mathbf{p} \sim \mathcal{N}(\hat{\mathbf{p}}, \boldsymbol{\Sigma})$:

$$\mathbf{R}\mathbf{p} \sim \mathcal{N}(\mathbf{R}\hat{\mathbf{p}}, \mathbf{R}\boldsymbol{\Sigma}\mathbf{R}^T)$$

This is used when transforming obstacle covariances between frames.

## 4.8 Verification (Python)

```python
import numpy as np

def Rx(phi):
 c, s = np.cos(phi), np.sin(phi)
 return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def Ry(theta):
 c, s = np.cos(theta), np.sin(theta)
 return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def Rz(psi):
 c, s = np.cos(psi), np.sin(psi)
 return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# Verify SO(3) properties
R = Rz(0.5) @ Ry(0.3) @ Rx(0.2)
assert np.allclose(R @ R.T, np.eye(3)) # orthogonality
assert np.allclose(R.T, np.linalg.inv(R)) # inverse = transpose
assert abs(np.linalg.det(R) - 1.0) < 1e-10 # determinant = 1

# Verify norm preservation
v = np.array([1.0, 2.0, 3.0])
assert abs(np.linalg.norm(R @ v) - np.linalg.norm(v)) < 1e-10
```

## 4.9 Prerequisites and Related Chapters

> [!info] Prerequisites

> [!info] Used In
- [[02_Quadrotor_Dynamics|Ch.2: Quadrotor Dynamics]] — $\mathbf{R}_Z(\psi)$
- [[03_Coordinate_Frames|Ch.3: Coordinate Frames]] — Frame transformations
- [[15_Obstacle_Avoidance|Ch.15: Obstacle Avoidance]] — $\boldsymbol{\Omega}_{io}$
- [[16_Optimization|Ch.16: Optimization Formulation]] — FOV constraints

> [!info] See Also
- [[05_Euler_Angles|Ch.5: Euler Angles]] — Alternative parameterization
- [[06_Quaternion|Ch.6: Quaternion]] — Alternative representation
