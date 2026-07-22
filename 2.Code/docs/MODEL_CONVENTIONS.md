# Model conventions

## Controller state

$$x=[p_x,p_y,p_z,v_x,v_y,v_z,\phi,\theta,\psi]$$

## Command

$$u=[\phi_c,\theta_c,v_{z,c},\dot\psi_c]$$

- World frame: right-handed Cartesian frame; altitude is positive upward.
- Angles are radians in code and configuration unless a key explicitly ends in `_deg`.
- Covariances use the same state order as `x`.
- Ellipsoid `size` is the full axis length used by the source formulation; the collision radius of
  the MAV is added during construction of the collision matrix.
- Chance residual greater than or equal to zero is locally safe; negative means the softened
  probabilistic constraint is violated.

The 13-state track uses `[position, velocity, quaternion, body rates]` and generalized
`[thrust deviation, tau_x, tau_y, tau_z]`. An adapter is required before comparing its controls to
the 9-state track.
