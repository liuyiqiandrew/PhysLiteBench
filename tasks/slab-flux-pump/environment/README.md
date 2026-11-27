# Slab pickup and voltage taps

An infinite hard type-II superconducting slab occupies −1 ≤ x ≤ 1. Currents
flow along y and induction points along z. Use slab half-thickness as length
unit, the pinning field scale as induction unit, and μ₀ = 1. The quasistatic
local critical current magnitude is j/(1 + |B|). The common scale j lies in
[0.85,1.15]. Use the ideal critical state with isotropic pinning, negligible
lower critical field, creep, surface barriers, reversible magnetization and
displacement current. Flux stays pinned where the current is subcritical.
During flux motion the electric field points along the local critical
current. The electric field unit times the time unit is the induction unit
times the length unit.

Each fresh slab starts at B = 0. Its imposed face inductions at x = −1 and
x = +1 are h + i and h − i, respectively. First ramp i from zero to the
specified bias at h = 0, then hold i fixed. Ramp h from zero to `low`, from
`low` to `high`, and back to `low`. Record the **following** cycle from low
to high and back to low. All ramps are linear and quasistatic. The bias is
signed. Fields satisfy |i| ≤ 0.28 and −1.6 ≤ low < high ≤ 1.6.

* `crest` measures the spatial mean induction through the slab at the high
  endpoint of the recorded cycle.
* `trough` measures the same quantity at the low endpoint at the end.
* `impulse` measures the time integral of E_y just inside the **left** face
  over the full recorded cycle, positive along +y. Taps have unit separation
  and negligible lead-loop area.

Each experiment is a dictionary with fields `bias`, `low`, `high`, and
`readout`. Data in `data/calibration.json` use records with an `input`
experiment and a `value` rounded to eight decimal places. Implement
`Model.fit(records)` returning self and setting scalar `pinning`, and
`Model.predict(experiments)` returning a flat finite NumPy array in order.

All readouts are dimensionless in the stated units. Predictions are assessed to
absolute error 0.003 for mean induction and 0.008 for cycle impulse. Calibration
predictions should reproduce the rounded readings to within 0.00001.
