# Stellar tracer counts

Massless collisionless tracers orbit an imposed spherical potential
Φ(r) = −m/(b + √(b² + r²)). Adopt units with G = 1. Length, speed and tracer
number are in fixed instrument units throughout. Initially (m,b) = (1,1).
The stationary, isotropic number density in six-dimensional phase space
(dN per d³x d³v) is (−E/0.4)^p for negative specific energy E and zero
otherwise. Its normalization is known. The common exponent p lies in [2,4].

Each experiment starts with a fresh population. The apparatus first changes
the potential arbitrarily slowly to the pair in `slow`. Take the adiabatic
limit for every finite bound orbit. It then switches, in order, to the
potentials in `steps`, each in negligible time. After each switch unbound
tracers escape, and the readout averages over complete radial phase mixing
in the current potential before the next operation. The system is isolated
and tracers never collide. There is no thermal bath or replenishment.

In the final potential the counter resolves a narrow bin in specific energy E
and angular momentum **magnitude** L. The orbit at the bin center has turning
radii `peri` and `apo`. The readout is the differential d²N/(dE dL) there,
integrated over all positions, velocity directions and orbital phases. Take
the limit of vanishing bin widths. Counting units include the full angular
solid angles.

Inputs are dictionaries with `slow` = [m,b], `steps` a list of zero to two
[m,b] pairs, and 0.12 ≤ peri < apo ≤ 10. Each mass is in [0.45,2.5] and each
scale length in [0.12,2.5]. Calibration uses static fields. Each row of
`data/calibration.json` has an `input` and a positive `value` rounded to
nine significant figures. Implement `Model.fit(records)` returning self with
fitted scalar `exponent`, and `Model.predict(experiments)` returning a finite
one-dimensional NumPy array in input order.

Some experiments replace the list of instantaneous steps by a long sequence of weak potential fluctuations. In these runs `steps` is empty and `train` supplies a positive dimensionless exposure, a coupling [a,c] and a positive binding_limit. The population first reaches the stated slow field (m,b). A detector removes all stars whose binding energy −E is at or below binding_limit in that field. Each cycle then:

1. Changes (m,b) instantaneously to (m exp(a ε), b exp(c ε)).
2. Allows complete radial phase mixing there, with unbound tracers escaping.
3. Restores (m,b) arbitrarily slowly, repeats the binding-energy cut and allows complete radial phase mixing before the next cycle.

The counter measures the same final differential number after N cycles. Take ε → 0 and N → ∞ with N ε² tending to exposure. Use the stated order of mixing and restoration within each cycle. Exposure lies in [0.05, 0.65], both coupling entries in [−1,1], and binding_limit in [0.025,0.25] in energy units. For bins outside the surviving energy range the count is zero. The initial distribution, calibration and detector units are the same for both protocols.
