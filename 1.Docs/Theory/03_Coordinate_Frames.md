---
title: 'Chương 3: Coordinate Frames'
chapter: 3
tags:
- quadrotor
- kinematics
- frames
- transformations
phase: foundations
cssclass: theory-note
created: '2026-06-27'
aliases:
- Chapter 3
- Coordinate Frames
- Ch.3
---

## 3.1 Introduction

Describing the motion of a quadrotor requires multiple coordinate frames. The relationship between these frames is fundamental to:
- Expressing sensor measurements in a global reference
- Converting control commands to body-frame actions
- Computing relative positions for obstacle avoidance

## 3.2 Frame Definitions

### World Frame $W$ (Inertial)

- **Origin**: Fixed point in space (e.g., takeoff location)
- **Axes**: 
 - $X_W$: Horizontal (typically North or forward)
 - $Y_W$: Horizontal (typically East or left)
 - $Z_W$: Vertical upward (opposite gravity)

- **Used for**: Global position, goal specification, obstacle world positions

### Body Frame $B$ (Quadrotor-Fixed)

- **Origin**: Center of mass of the quadrotor
- **Axes** (right-hand rule):
 - $X_B$: Forward direction (between rotors 1 and 2 for "+" config)
 - $Y_B$: Left direction
 - $Z_B$: Upward (perpendicular to rotor plane)

- **Used for**: Control forces, sensor mounting, body-frame velocities

### Camera Frame $C$ (Sensor-Fixed)

- **Origin**: Camera optical center
- **Axes**:
 - $X_C$: Forward (depth direction, into the scene)
 - $Y_C$: Right (image x-direction)
 - $Z_C$: Down (image y-direction, standard computer vision)

- **Used for**: Depth image interpretation, obstacle detection

### Image Frame (Pixel)

- **Origin**: Top-left corner of image
- **Axes**:
 - $u$: Horizontal (column, 0 to width-1)
 - $v$: Vertical (row, 0 to height-1, downward)

- **Used for**: U-depth maps, bounding box coordinates

## 3.3 Frame Transformations

### Body → World

A point $\mathbf{p}^B$ in body frame transforms to world frame via:

$$\mathbf{p}^W = \mathbf{R}_B^W \mathbf{p}^B + \mathbf{t}_B^W$$

where:
- $\mathbf{R}_B^W \in SO(3)$: Rotation matrix from body to world (the MAV's attitude)
- $\mathbf{t}_B^W \in \mathbb{R}^3$: Position of body origin in world frame (the MAV's position)

### World → Body

The inverse transformation:

$$\mathbf{p}^B = \mathbf{R}_B^{W\;T}(\mathbf{p}^W - \mathbf{t}_B^W) = \mathbf{R}_W^B(\mathbf{p}^W - \mathbf{t}_B^W)$$

where $\mathbf{R}_W^B = \mathbf{R}_B^{W\;T}$.

### Camera → Body → World

$$\mathbf{p}^W = \mathbf{R}_B^W(\mathbf{R}_C^B \mathbf{p}^C + \mathbf{t}_C^B) + \mathbf{t}_B^W$$

For simplicity, the camera is assumed to be at the body origin ($\mathbf{t}_C^B = \mathbf{0}$) with its forward axis aligned with $X_B$ and downward axis with $Z_B$ (standard for forward-facing depth cameras).

### Yaw-Only Transformation

For FOV constraints, only the yaw rotation is needed (camera is assumed to point in the horizontal plane):

$$\mathbf{p}^B = \mathbf{R}_Z(\psi)^T(\mathbf{p}^W - \mathbf{t}_B^W)$$

where $\mathbf{R}_Z(\psi)$ is the rotation matrix about the world Z-axis.

## 3.4 Uncertainty Transformation

When transforming random variables between frames, the covariance transforms as:

$$\boldsymbol{\Sigma}^W = \mathbf{R}_B^W \boldsymbol{\Sigma}^B \mathbf{R}_B^{W\;T}$$

This is the **push-forward of covariance** under rotation. The MAV's own position uncertainty adds to the transformed obstacle uncertainty:

$$\boldsymbol{\Sigma}_o^W = \mathbf{R}_B^{W\;T} \boldsymbol{\Sigma}_o^B \mathbf{R}_B^W + \boldsymbol{\Sigma}^W$$

Note the transpose convention: $\mathbf{R}_B^{W\;T}\boldsymbol{\Sigma}_o^B\mathbf{R}_B^W$ rotates the obstacle covariance from body to world. This matches the paper's Eq. (4).

## 3.5 Practical Considerations

### Axis Conventions

The CC-MPC papers use a **Z-up world frame**:
- $Z$ positive = upward (opposite gravity)
- This differs from NED (North-East-Down) used in aerospace

### Euler Angle Sequence

The papers use **ZYX Euler angles** (yaw-pitch-roll), which is standard for quadrotors:

$$\mathbf{R}_B^W = \mathbf{R}_Z(\psi)\mathbf{R}_Y(\theta)\mathbf{R}_X(\phi)$$

This means:
1. First rotate by roll $\phi$ about body X
2. Then pitch $\theta$ about new Y
3. Then yaw $\psi$ about world Z

### Altitude Convention

- $z = 0$ at ground level
- $z > 0$ means above ground
- Minimum altitude constraint: $z \geq 0.1$ m (ground effect avoidance)

## 3.6 Implementation

```python
def body_to_world(p_body, position, roll, pitch, yaw):
 """Transform point from body to world frame."""
 R = euler_to_rotation(roll, pitch, yaw) # ZYX rotation matrix
 return R @ p_body + position

def world_to_body(p_world, position, yaw):
 """Transform point from world to body frame (using only yaw)."""
 Rz = yaw_to_rotation(yaw) # Z-axis rotation
 return Rz.T @ (p_world - position)

def transform_covariance(Sigma_body, R_body_to_world):
 """Transform covariance from body to world frame."""
 return R_body_to_world.T @ Sigma_body @ R_body_to_world
```

## 3.7 Prerequisites and Related Chapters

> [!info] Prerequisites

> [!info] Used In
- [[02_Quadrotor_Dynamics|Ch.2: Quadrotor Dynamics]] — Velocity in body vs. world
- [[15_Obstacle_Avoidance|Ch.15: Obstacle Avoidance]] — Detection frame transforms
- [[16_Optimization|Ch.16: Optimization Formulation]] — FOV constraint formulation

> [!info] See Also
- [[04_Rotation_Matrix|Ch.4: Rotation Matrix]] — $\mathbf{R}_Z(\psi)$, $\mathbf{R}_B^W$
- [[05_Euler_Angles|Ch.5: Euler Angles]] — $\phi, \theta, \psi$ definition
- [[06_Quaternion|Ch.6: Quaternion]] — Alternative attitude representation
