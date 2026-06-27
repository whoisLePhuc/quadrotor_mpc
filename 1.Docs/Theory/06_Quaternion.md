---
title: 'Chương 6: Quaternion'
chapter: 6
tags:
- quadrotor
- kinematics
- quaternion
- attitude
phase: foundations
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 6
- Quaternion
- Ch.6
---

## 6.1 Introduction

A **quaternion** $\mathbf{q} = [w, x, y, z]^T$ is a 4-parameter representation of 3D rotation that avoids gimbal lock. While the CC-MPC state vector uses Euler angles, quaternions are used in the implementation for:
- Conversion between MuJoCo simulation and the MPC
- VIO state representation
- Interpolation and numerical stability

## 6.2 Definition

A unit quaternion (rotation quaternion) satisfies:

$$\mathbf{q} = \begin{bmatrix} w \\ x \\ y \\ z \end{bmatrix}, \quad \|\mathbf{q}\| = \sqrt{w^2 + x^2 + y^2 + z^2} = 1$$

The quaternion represents a rotation by angle $\alpha$ about axis $\mathbf{v} = [v_x, v_y, v_z]^T$:

$$\mathbf{q} = \begin{bmatrix} \cos(\alpha/2) \\ v_x \sin(\alpha/2) \\ v_y \sin(\alpha/2) \\ v_z \sin(\alpha/2) \end{bmatrix}$$

## 6.3 Rotation Matrix from Quaternion

$$\mathbf{R}(\mathbf{q}) = \begin{bmatrix}
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{bmatrix}$$

For a unit quaternion, this produces a valid $SO(3)$ matrix.

## 6.4 Quaternion from Euler Angles

ZYX convention (yaw-pitch-roll):

$$\mathbf{q} = \begin{bmatrix}
\cos\frac{\phi}{2}\cos\frac{\theta}{2}\cos\frac{\psi}{2} + \sin\frac{\phi}{2}\sin\frac{\theta}{2}\sin\frac{\psi}{2} \\
\sin\frac{\phi}{2}\cos\frac{\theta}{2}\cos\frac{\psi}{2} - \cos\frac{\phi}{2}\sin\frac{\theta}{2}\sin\frac{\psi}{2} \\
\cos\frac{\phi}{2}\sin\frac{\theta}{2}\cos\frac{\psi}{2} + \sin\frac{\phi}{2}\cos\frac{\theta}{2}\sin\frac{\psi}{2} \\
\cos\frac{\phi}{2}\cos\frac{\theta}{2}\sin\frac{\psi}{2} - \sin\frac{\phi}{2}\sin\frac{\theta}{2}\cos\frac{\psi}{2}
\end{bmatrix}$$

**Implementation** (from `utils.py`):
```python
def euler_to_quat(roll, pitch, yaw):
 cr, sr = cos(roll*0.5), sin(roll*0.5)
 cp, sp = cos(pitch*0.5), sin(pitch*0.5)
 cy, sy = cos(yaw*0.5), sin(yaw*0.5)
 return np.array([
 cr*cp*cy + sr*sp*sy,
 sr*cp*cy - cr*sp*sy,
 cr*sp*cy + sr*cp*sy,
 cr*cp*sy - sr*sp*cy,
 ])
```

## 6.5 Euler Angles from Quaternion

ZYX convention:

```python
def quat_to_euler(q):
 w, x, y, z = q
 roll = atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
 pitch = asin(clamp(2*(w*y - z*x), -1, 1))
 yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
 return roll, pitch, yaw
```

**Singularity**: At pitch = ±90° ($w*y - z*x = \pm 0.5$), roll and yaw are coupled. The clamp to $[-1, 1]$ handles numerical edge cases.

## 6.6 Quaternion Operations

### Multiplication (Composition)

$$\mathbf{q}_1 \otimes \mathbf{q}_2 = \begin{bmatrix}
w_1w_2 - x_1x_2 - y_1y_2 - z_1z_2 \\
w_1x_2 + x_1w_2 + y_1z_2 - z_1y_2 \\
w_1y_2 - x_1z_2 + y_1w_2 + z_1x_2 \\
w_1z_2 + x_1y_2 - y_1x_2 + z_1w_2
\end{bmatrix}$$

### Conjugate (Inverse for unit quaternion)

$$\mathbf{q}^* = \begin{bmatrix} w \\ -x \\ -y \\ -z \end{bmatrix}$$

### Normalization

$$\mathbf{q}_{\text{normalized}} = \frac{\mathbf{q}}{\|\mathbf{q}\|}$$

## 6.7 When to Use Quaternions vs. Euler Angles

| Context | Use | Reason |
|---------|-----|--------|
| MPC state vector | Euler angles | Simple dynamics, small angles |
| MuJoCo simulation | Quaternions | Gimbal-lock free, simulation standard |
| VIO output | Quaternions | Avoids singularities in estimation |
| Visualization | Euler angles | Intuitive for humans |
| Interpolation | Quaternions | SLERP gives smooth interpolation |
| Jacobian computation | Euler angles | Clean partial derivatives |

## 6.8 Prerequisites and Related Chapters

> [!info] Prerequisites

> [!info] Used In
- [[02_Quadrotor_Dynamics|Ch.2: Quadrotor Dynamics]] — Attitude representation alternative
- [[18_Implementation_Notes|Ch.18: Implementation Notes]] — MuJoCo interface

> [!info] See Also
- [[05_Euler_Angles|Ch.5: Euler Angles]] — The representation used in the MPC
