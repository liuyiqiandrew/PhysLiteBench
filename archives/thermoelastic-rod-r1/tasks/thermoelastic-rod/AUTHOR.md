# Thermoelastic rod, revision 1

The intended error is replacing a fixed total-length constraint by a local
constant-stress heat capacity everywhere. Calibration does not distinguish them.
The public model is linear thermoelasticity after free transverse relaxation.
With beta=E*alpha, entropy density increments satisfy

    ds = (c_e/T0) dtheta + beta depsilon,
    sigma = E*epsilon - beta*theta.

Quasistatic force balance makes stress spatially uniform, and the clamps impose
mean(epsilon)=0. Thus epsilon=alpha*(theta-mean(theta)), giving

    c_s*theta_dot - (c_s-c_e)*mean(theta_dot) = k*theta_xx,
    c_s = c_e + T0*E*alpha^2 = 3.4e6 J/(m^3 K).

The constant spatial mode has heat capacity c_e*V=5 J/K; all nonzero cosine
modes have c_s*V=17 J/K. Initial mean-zero insulated cosine modes keep their
mean zero, making the shortcut exact and identifying k from exponential decay.
The finite thermal body changes the mean during hidden heat exchange, exposing
the nonlocal mechanical constraint. Both controls use the same conductivity.

The oracle is a normalized cosine Galerkin model with a thermal-contact port.
The independent reference evolves cell entropy and bath energy. A separate
linear system solves cell temperature, strain, and uniform reaction stress;
finite-volume heat fluxes use half-cell conduction in series with the contact.
It does not assign the oracle's mode heat capacities. The closed system's
linearized energy is 5*mean(theta)+6*theta_bath, not 17*mean(theta)+6*theta_bath.
Conduction and contact cause entropy production only at the next order in
small departures, consistently with linearized thermoelastic balance.

All temperature outputs are departures in K. Hidden groups use absolute RMSE
with threshold 0.025 K, avoiding normalization by a nearly zero moment. The
large ideal expansion coefficient remains within the stated linear small-strain
regime for departures at most 1 K. Mechanical inertia/dissipation are absent.

Calibration seed 9310 uses k=145 W/(m K), 160 independent readings, and sigma
0.00146259 K. Noise seed 19310 generates 256 additional realizations. The
checked-in fit gives k=145.11079 and reduced chi-square 1.01743 for both controls.
Oracle hidden errors are at most 0.000559 K; shortcuts give 0.0727–0.1449 K.
Independent reference refinement from 128 to 256 cells changes predictions by
at most 1.19e-5 K; the 64-mode oracle differs by at most 0.001042 K, below one
measurement sigma and far below the hidden threshold. Its closed-energy error
is below 8e-12 J, and it reaches the independently predicted common final
 temperature within 2e-11 K.

All 256 calibration fits pass; maximum relative parameter error is 0.824%.
Hidden checks at the observed fitted-parameter extrema give oracle errors below
0.000817 K and shortcut errors above 0.07269 K. This is a 256-fit calibration
Monte Carlo followed by two-parameter-extrema sensitivity checks, not 256 full
hidden evaluations or an exhaustive noise bound.

Run `scripts/validate_thermoelastic_rod.py`; ordinary validation reads both frozen
data copies and writes `jobs/thermoelastic-rod-validation/summary.json`.
Use `--generate` only for intentional regeneration before freezing a revision.
The completed shortcut is `scripts/thermoelastic_rod_baseline.py`. A teammate
reviewed the thermodynamic derivation, entropy/strain reference, units, initial
mechanical preparation, and half-cell contact law. Agent difficulty is pending.
