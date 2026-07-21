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
 - $X_W$: First horizontal axis of the chosen local map frame
 - $Y_W$: Second horizontal axis, chosen with $X_W$ and $Z_W$ to form a right-handed frame
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

If the camera uses $X_C$ forward, $Y_C$ right, $Z_C$ down while the body uses $X_B$ forward, $Y_B$ left, $Z_B$ up, the frames are not identical. For a co-located, forward-facing camera the idealized extrinsic rotation is $\mathbf{R}_C^B=\operatorname{diag}(1,-1,-1)$. A real implementation should use the calibrated camera-to-body extrinsics.

### Yaw-Only Approximation

The source implementation constructs its FOV from the current pose. A yaw-only transform is a project-specific approximation that is valid only when roll/pitch, camera translation, and camera-to-body axis differences are deliberately neglected:

$$\mathbf{p}^B = \mathbf{R}_Z(\psi)^T(\mathbf{p}^W - \mathbf{t}_B^W)$$

where $\mathbf{R}_Z(\psi)$ is the rotation matrix about the world Z-axis.

## 3.4 Uncertainty Transformation

When transforming random variables between frames, the covariance transforms as:

$$\boldsymbol{\Sigma}^W = \mathbf{R}_B^W \boldsymbol{\Sigma}^B \mathbf{R}_B^{W\;T}$$

This is the **push-forward of covariance** under rotation. The MAV's own position uncertainty adds to the transformed obstacle uncertainty:

$$\boxed{\boldsymbol{\Sigma}_o^W = \mathbf{R}_B^W \boldsymbol{\Sigma}_o^B \mathbf{R}_B^{W\;T} + \boldsymbol{\Sigma}_{\text{MAV}}^W}$$

This follows from $\mathbf{p}_o^W=\mathbf{R}_B^W\mathbf{p}_o^B+\mathbf{p}_{\text{MAV}}^W$ and assumes the body-frame detection error is independent of the MAV position error. If correlations are retained, cross-covariance terms must also be included.

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

def world_to_yaw_aligned_body(p_world, position, yaw):
 """Yaw-only approximation; this is not a full world-to-body transform."""
 Rz = yaw_to_rotation(yaw) # Z-axis rotation
 return Rz.T @ (p_world - position)

def transform_covariance(Sigma_body, R_body_to_world):
 """Transform covariance from body to world frame."""
 return R_body_to_world @ Sigma_body @ R_body_to_world.T
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
