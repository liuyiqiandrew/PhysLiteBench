# Finite-extensible polymer stress prototype

Prototype only; no model evaluations. The physical concept passed peer review;
root authorized a complete staged package if the independent checks pass.

Two identical overdamped beads move in a plane under prescribed uniform
incompressible extensional flow `u=(a*x,-a*y)`. Each bead has drag `zeta`,
independent thermal forcing with covariance `2*zeta*T`, and the beads interact
only through `U(Q)=-Qmax²*log(1-Q²/Qmax²)/2` (spring constant one). They are dilute
and have no hydrodynamic interaction. A Hookean connector is the `Qmax=infinity`
limit. The unknown common bead drag lies in[3.2,4.8]; temperature energy is[.8,1.2].
The rheometer measures excess planar mechanical stress `Sigma_xx-Sigma_yy`,
subtracts the solvent, and divides by known polymer areal number density. This
readout has energy units and is the connector force dipole difference; isotropic
Brownian stress cancels.

The completed source uses the standard Peterlin preaveraging of nonlinear spring
force. In reduced variables `Wi=zeta*a/4` and `b=Qmax²/T`, its positive steady
conformation is `Cxx=T/(f-2Wi)`, `Cyy=T/(f+2Wi)`, with
`f=1/[1-tr(C)/Qmax²]`. A monotone scalar root determines the unique physical
branch `f>max(1,2|Wi|)`. Its stress is `f*(Cxx-Cyy)`. This is a correct mathematical
implementation of a known constitutive approximation, not a force sign error.

Nonzero signed Hookean rates up to.22 and varied temperatures identify the drag
exactly for both models. The entire drag range stays below the Hookean extension
instability. Hidden finite connectors have `Qmax in[sqrt6,sqrt12]`, hence
`b in[5,15]`; rates around.35–.95 give substantial stress and healthy separation.
All used spring-force moments are integrable at the natural zero-flux boundary.

For symmetric extension the exact stationary density is proportional to
`(1-Q²/Qmax²)^(Qmax²/(2T))*exp[zeta*a*(Qx²-Qy²)/(4T)]` inside the connector disk.
The oracle integrates its second moments using a radial Jacobi/Bessel reduction
and the exact steady second-moment balance. The independent reference integrates
the actual force dipole directly in nested Cartesian coordinates over the disk.
An additional differently weighted radial integral checks the force singularity,
and the source's own conformation equation is checked directly.

All48 domain corner/random checks pass. Source errors are22.8–68.0%, while
nonzero stresses range.626–12.64. Independent Cartesian force/reference error is
below1.44e-9 relative, refinement below1.30e-9, and direct radial force agreement
below1.30e-13. The source equation residual is7.3e-14. Forty-five signed Hookean
calibration cases agree exactly with the source and an independent OU Lyapunov
solve; minimum relaxation stability margin is.1967. Zero-flow isotropy and
large-Qmax convergence are checked.

Preserved exploratory limitations: `screen.py` initially used the probability
Jacobi weight to evaluate an endpoint-singular force and gave a small quadrature
error. The final force integral uses exponent `b/2-1` instead. The first
supplementary `b=10000` limit test overflowed SciPy's unnormalized Jacobi weight;
its source/report are preserved as `*-before-limit-fix.*`. The final outside-domain
limit uses direct radial quadrature with an exponentially bounded tail. Neither
issue affects the selected domain or the physical source approximation.

A broad archive/staging/task author-file search for Peterlin, FENE,
finite-extensibility and nonlinear-spring found no packaged task. Archived
active-trap Gaussian propulsion and elastic-ring rigid/stiff statistics share
only broad statistical-closure themes. Here the physical distinction is the
nonlinear molecular-force preaverage in a dilute constitutive law. Difficulty
is unmeasured; it must be established by fresh trials, not by these margins.

Primary background: [On the Peterlin approximation for finitely extensible
dumbbells](https://doi.org/10.1016/S0377-0257(96)01497-8) and
[Validity of a macroscopic description in dilute polymeric solutions](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.62.1441).
