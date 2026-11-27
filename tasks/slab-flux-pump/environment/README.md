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

## Rotating-field measurements

A second drive on the same slab rotates the induction in the yz plane. These
experiments use no transport bias. Both faces see the same rotating vector
H (cos ωt, sin ωt) in (y,z) components. Measurements are on the left near
surface. The two penetrated layers stay separated by a virgin core. Take
the steady co-rotating state reached with zero trapped field in that core.
The angular frequency is slow enough for the critical-state law above.

For a local induction of magnitude b, the admissible currents fill an
ellipse. Its semiaxis perpendicular to the local induction is j/(1+b).
The semiaxis parallel to the induction equals that value times the input factor
`cutting` (which may be below one). In flux motion the electric field
points along the outward normal to the ellipse at the actual current.
The same law reduces to the scalar law in the longitudinal experiments.

Rotating inputs supply `amplitude` H in [0.10,0.32], `sense` (the sign
of ω, +1 or −1), `cutting` in [0.45,2.6], and `depth` in [0,0.9]
measured into the material from the left face. At phase ωt=0 the probe
reads `induction` or `electric` along `axis` y or z. The latter is
the electric field divided by |ω|. Specify the quantity in `readout`.

## Rotating a pre-threaded laminate

A second batch has a uniform normal induction `normal` along +x, applied
before the rotating drive. Its magnitude is in [0.18,0.32]. The core keeps
this normal induction and zero tangential induction. At each outer face the
tangential induction rotates as above with H in [0.17,0.22]. No current flows
in x. In the bulk the critical surface is an ellipsoid of revolution about
the **total local induction**. Its two perpendicular semiaxes equal
j/(1+|B|), and its parallel semiaxis equals that value times `cutting`.
During motion the full three-dimensional E is an outward normal to the
ellipsoid. The bulk above and below a sheet is the same superconductor.

A nonmagnetic conducting sheet of negligible thickness lies at depth
`coating` in [0.045,0.095] inside each face. Consider the left sheet and
its nearby probes. The sheet carries tangential current per unit width

    K = conductance [e + hall sgn(normal) x̂ × e],

where e is the local tangential E divided by |ω|. The real inputs
`conductance` in [0,1.7] and `hall` in [−0.65,0.65] include the choice
of frequency and units. Use Maxwell's interface conditions and zero normal
voltage drop in the sheet. All fields rotate steadily with the drive except
normal components, which are time independent.

These runs supply `normal`, `coating`, `conductance` and `hall` together.
The tangential `induction` and `electric` readouts use the same phase,
units and axes as above. At the sheet use the limit from the coating.
Two additional readouts omit `axis`.

* `normal_voltage` measures the integral of Eₓ/|ω| from the left surface
  to the probe at `depth`, with positive integration direction into the slab.
* `sheet_power` measures the cycle mean of K·E/|ω| at the left sheet
  per unit sheet area. No `depth` is required.

Absolute error tolerances are 0.001 for rotating induction, 0.0006 for
tangential electric field, 0.00025 for normal voltage and 0.00015 for sheet power.

## Laminate with a superconducting film

A `film` input selects a double-critical-state bulk. In this material cutting
and depinning have independent thresholds. Resolve the three-dimensional
current into its component along the total local induction and its vector
component perpendicular to that induction. Their respective bounds are
`cutting` times j/(1+|B|) and j/(1+|B|). Both bounds must hold in the bulk
on either side of the sheet. At a critical current during flux motion E is
an outward normal to the full admissible set, allowing the full outward
normal cone where surfaces meet. The transport condition Jₓ = 0 still holds.

The original Hall conductor now shares the sheet with an isotropic,
zero-thickness superconducting film in parallel. The input `film` gives
its critical sheet current κ in [0.015,0.27], in the same units as K.
Add the film current to the conducting current in the magnetic interface
condition. At zero tangential electric field the film can carry any sheet
current of magnitude ≤κ. With a nonzero tangential electric field its
current is critical and along that field. The ideal film has negligible
kinetic inductance. The sheet has zero normal voltage drop.

Start with zero tangential induction throughout and raise the rotating drive
from zero to its measured amplitude. Use the resulting steady periodic
state. The input ranges and probes remain as in the pre-threaded laminate,
except that `cutting` ranges from 0.30 to 1.35 for a film experiment.
The sheet-power probe measures the work on the **total** sheet current.
