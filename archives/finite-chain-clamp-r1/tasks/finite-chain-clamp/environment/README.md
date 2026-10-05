# Chain measurements

A dilute chain in three-dimensional space consists of `links = N` bonds and N+1 classical point beads. Bead 0 is pinned at the origin. Adjacent beads have the central potential

    U_s = (k_s/2) * sum_j (|r_j-r_(j-1)| - b)^2.

There is no bending energy, excluded volume, other interaction, or background flow. All preparations share the unknown bond rest length `b` in [0.8, 1.2]. The full Cartesian canonical equilibrium with a heat reservoir is established at every finite stiffness. The readouts below take `k_s` to infinity at fixed finite N, b, temperature and other controls.

Lengths are in a fixed unit L0, energies in E0, and `temperature` is k_B times the physical temperature in E0 units. Forces are in E0/L0 and spring stiffnesses in E0/L0^2.

Two measurement settings are used:

- `mode = "force"`: a constant force `force = F` acts on the last bead in +x through potential -F*x_N. The output is its equilibrium mean x coordinate.
- `mode = "clamp"`: the last bead has the additional potential K*(x_N-X)^2/2, where `extension = X`. Its y and z coordinates are free. The output is the signed mean +x force K*(X-x_N) exerted by the clamp on the chain. Establish equilibrium at finite k_s and K, first take k_s to infinity at fixed K, then take K to infinity at fixed X.

The allowed controls are integer N from 4 through 12 and temperature from 0.6 through 1.4. In force mode, F is from 0.3 through 3.0. In clamp mode, X/N is from 0.15 through 0.60 in units L0.

## Data and interface

`data/calibration.json` contains independent measurements with keys `input`, `value`, and `sigma`. Each input has `mode`, `links`, `temperature`, and the applicable `force` or `extension`. Every listed sigma is the fixed instrument standard deviation 0.006 L0 for the measured mean extension. Repeated settings are independent measurements.

Implement `Model.fit(records)` to estimate the common b, save it as `self.link_length`, and return `self`. `Model.predict(experiments)` must return a finite NumPy array of shape `(len(experiments),)` in the listed order and units. Inputs to prediction use the same schema and allowed ranges.
