# Nematic anchoring measurement

A uniform nematic at fixed temperature is described by a unit director `n` and
the elastic free-energy density

```
f = K/2 * [(div n)^2 + |curl n|^2]
    - K24 * div[n*(div n) + n cross (curl n)],
K24 = K/2.
```

The derivatives are with respect to physical Cartesian position, and the free
energy is the volume integral of `f`. This equation defines the coefficient
convention. The positive modulus `K` lies in `[8,16]` pN and is the only unknown.
There is no chirality, applied bulk field, appreciable director fluctuation, or
additional surface energy. Ideal anchoring controls fix the boundary director.

Each experiment starts with axial alignment. The imposed boundary tilts are
raised proportionally from zero to their stated values quasistatically, allowing
relaxation at each step along the smooth stable state without defects.

Two cell geometries are available:

- `flat`: a slab of thickness `thickness` with normal `e_x`, uniform parallel to
  its walls. The directors on the two walls are
  `sin(inner_angle)*e_x + cos(inner_angle)*e_z` and
  `sin(outer_angle)*e_x + cos(outer_angle)*e_z`.
- `annulus`: a long cylindrical annulus with inner radius `inner_radius` and
  outer radius `inner_radius*radius_ratio`, with end effects neglected. At both
  walls the imposed director is `sin(angle)*e_r + cos(angle)*e_z`, using the
  corresponding inner or outer angle. Here `e_r` points away from the cylinder
  axis at both walls.

The outer anchoring actuator increases `outer_angle` reversibly by a small amount
while the inner orientation and geometry remain fixed. In the annulus each
anchoring element rotates about its local azimuthal tangent `e_phi`. The
measurement is the work supplied by all these elements, divided by their total
wall area and the angle increment, in the limit of a vanishing increment. The
bare actuator work is subtracted. Report this signed work density per radian in
microjoules per square metre (`µJ/m²`); angles in radians are dimensionless.

Lengths supplied to the model are in micrometres. Allowed inputs are
`thickness in [.7,1.3]`, `inner_radius in [.7,1.3]`, `radius_ratio in [2,4]`,
`inner_angle in [.05,.15]`, and `outer_angle in [.3,.7]` radians. Each input has
`geometry`, both angles, and the length fields appropriate to that geometry.

## Interface

Implement `Model.fit(records)` and `Model.predict(experiments)` in `model.py`.
`fit` receives the records in `data/calibration.json`, stores the fitted modulus
in `self.elastic_constant` in pN, and returns `self`. Each record has `input`,
measured `value`, and independent Gaussian instrument standard deviation `sigma`.
`predict` receives a list of input dictionaries and returns a one-dimensional
NumPy array of work-density readings in the same order.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
