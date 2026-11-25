# Physics review of seven revision-1 candidates

Reviewed 2026-10-01 by a separate reviewing agent, following `AGENTS.md` and
`docs/ADDING_TASKS.md`. The review covered each public apparatus description,
starter, solution, independent reference, public test, hidden metric, hint, and
image boundary. No task code was changed by the reviewer.

The seven tasks have identifiable calibration parameters and exact calibration
equivalence between the stated shortcut and physical model. No unresolved
physical or interface blocker was found after the real-gas domain and radiative
far-field clarifications described below. Agent difficulty remains an empirical
question; this review does not establish a Luna pass rate.

## Numerical checks

Checks used Python 3.13, NumPy 2.3.3, and SciPy 1.16.3, evaluating both predictors
at the true parameter on every calibration and hidden input. These checks do
not replace fitted noisy-data controls or Harbor runs.

| Task | Shortcut hidden error across the three groups | Largest oracle/reference hidden discrepancy |
|---|---:|---:|
| Hall bar | 0.448–0.835 | 1.03e-16 |
| Moving inductor | 0.107–0.157 | 2.87e-10 |
| Effusive beam | 0.0867–0.2500 | 8.48e-17 |
| Real-gas expansion | 1.000 | 8.38e-16 |
| Induced dipole | 1.000 | 1.85e-12 |
| Radiative plates | 0.221–0.536 | 5.21e-16 |
| Magnetic equilibrium | 0.873–0.994 | 8.55e-9 |

Errors use the implemented grading scales: absolute RMSE for probabilities and
magnetization; cooling-relative RMSE for expansion; group RMS normalization for
current, force, and heat flux. Thresholds are 0.025 for effusion and magnetization,
and 0.04 for the other tasks. Every group separates its shortcut well beyond
numerical reference error. Calibration predictor differences are zero up to
floating-point roundoff; numerical free-energy minimization differs from the
magnetic calibration oracle by at most 1.81e-8.

## Findings by task

- **Hall bar:** Insulating sidewalls, steady charge redistribution, and imposed
  central longitudinal field determine the measured current. The reference
  correctly imposes zero transverse current on the positive-carrier Drude tensor.
  Mobility is identified by the zero-field current slope. Signed magnetic fields
  are exercised. The API requests only current, so transverse voltage conventions
  cannot affect grading. [TU Delft Drude notes](https://solidstate.quantumtinkerer.tudelft.nl/3_drude_model/)

- **Moving inductor:** The public reversible flux linkage and prescribed actuator
  motion imply the full derivative of `L(x)*I`. Calibration fixes the core and
  identifies resistance from transients. The solution integrates flux; the
  independent reference integrates current with the motional voltage included.
  A separate zero-resistance, zero-voltage moving-core check conserves flux to
  machine precision. All actual task resistances remain positive and all tested
  inductances remain positive. [Variable-inductor flux relation](https://www.mathworks.com/help/sps/ref/variableinductor.html)

- **Effusive beam:** The README explicitly specifies outward crossing events,
  equal detection efficiency, and no angular or residence-time selection. Flux
  weighting leaves the transverse Maxwell distributions unchanged while changing
  the normal marginal. Quadrature of the weighted density independently checks
  the analytic axial CDF. Calibration contains nonzero transverse thresholds and
  identifies temperature. [UT Austin effusion derivation](https://farside.ph.utexas.edu/teaching/sm1/Thermalhtml/node88.html)

- **Real-gas expansion:** Fixed-volume heat increments identify molar heat
  capacity while hiding the volume-dependent internal energy. The reference
  integrates the thermodynamic volume derivative independently of the solution's
  closed expression. The revised public domain uses `Cv=15..40` and expansion
  initial temperatures `400..550 K`: its largest possible cooling is 72 K,
  giving a minimum allowed final temperature of 328 K. This exceeds
  `Tc=298.350028 K`; minimum hidden temperature is 417.032967 K. Heating states
  are also above `Tc`, and all volumes exceed the excluded volume. The initial
  lower-temperature proposal was replaced before evaluation. Hidden errors
  normalize by cooling, with no zero-cooling hidden case. [Oxford thermodynamic consistency derivation](https://www-thphys.physics.ox.ac.uk/people/AlexanderSchekochihin/A1/2014/handout13_2015.pdf)

- **Induced dipole:** Reversible polarization, fixed external field sources, and
  an on-axis profile of a three-dimensional electrostatic field are specified.
  Internal polarization energy in the independent reference corrects the
  permanent-dipole shortcut. The laboratory force conversion is correct:
  `(1e-24 C m^2/V)*(1e3 V/m)*(1e6 V/m^2)=1e-15 N`.
  The reference energy unit is `1e-18 J`, whose derivative per millimeter gives
  the same force unit. Uniform-field dipole measurements identify polarizability.
  [MIT induced-polarizability lecture](https://ocw.mit.edu/courses/8-421-atomic-and-optical-physics-i-spring-2014/b92385ebc9928752f36a31085e76da58_jgSn1mB8uSI.pdf)

- **Radiative plates:** The final README specifies separation much greater than
  thermal wavelengths, excluding near-field transfer. Opaque diffuse gray
  surfaces, unit view factor, and maintained temperatures justify the radiosity
  reference. The two-equation radiosity solve checks the scalar formula without
  copying it. Black opposing panels identify plate-1 emissivity and make the
  shortcut exact during calibration. Hidden groups include both flux signs.
  [INL parallel-plate verification](https://mooseframework.inl.gov/bison/verification/ECAR_131/ECAR_131.html)

- **Magnetic equilibrium:** The Hamiltonian, thermodynamic limit, and fully
  equilibrated canonical preparation define the physical branch without
  metastability ambiguity. Calibration temperatures exceed every allowed
  coupling, and nonzero fields identify coupling. The reference minimizes the
  entropy-based free energy globally; the solution brackets the stable root with
  the field's sign. All 72 hidden shortcut roots report convergence, satisfy the
  self-consistency equation to at most 5.56e-17, and lie inside `[-1,1]`.
  Their free-energy curvature ranges from -0.3800 to -0.0977: these are unstable
  stationary states, not failed numerical root solves. Hidden fields include
  both signs and exclude zero. [Tong mean-field free energy](https://www.damtp.cam.ac.uk/user/tong/sft/sfthtml/S1.html)

## Remaining validation

The subsequently completed
[local validation report](../../results/candidate-validation.json) was inspected.
All seven fitted oracles pass calibration, parameter recovery, and hidden
thresholds; all seven completed shortcuts pass calibration and parameter
recovery but fail every hidden group. Across 256 fresh calibration-noise
realizations per task, every fit passes the calibration threshold and the
largest parameter error is 1.24%. The first and last realizations cross-check
the shortcut fit. Hidden groups are then evaluated at the observed fitted
parameter minima and maxima, where the control separation remains. These are
calibration Monte Carlo and hidden parameter-extrema checks, not exhaustive
hidden Monte Carlo. Seeds and per-task values are retained in the report.

The public image copies only apparatus files, starter code, public tests, and
calibration data. Hints, true parameters, hidden preparations, references, and
this review remain outside it. Private calibration copies protect verification
against public-data edits. The common scalar-record API and laboratory units
are documented, and hidden tests use finite, correctly shaped outputs.

No further physics correction is requested. Fresh Harbor oracle/shortcut runs
and reviewed unhinted/hinted Luna trials must be recorded separately; this
review makes no Harbor or agent success claim. Count magnetic failures as physical state
selection only when their submitted root residuals remain small, and distinguish
any agent coding or numerical failures from the intended assumptions.
