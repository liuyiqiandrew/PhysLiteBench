# FENE rotating-flow prototype

Status: scientifically checked same-family revision; no model evaluations.
The r1 result1/3 remains preserved in archives/fene-stress-r1.

Known planar flow is `u=(e*x-omega*y,omega*x-e*y)`. The unchanged microscopic
finite spring and per-bead fluctuation–dissipation law define a stationary disk
Fokker–Planck problem. Source Peterlin preaveraging solves the exact full-flow
Lyapunov covariance and a positive scalar self-consistency root. Hookean
calibration can span signed rotation and extension while matching exactly.

Mixed extension and rotation produces a circulating stationary distribution;
it cannot use the potential-flow Gibbs shortcut. Pure rotation still preserves
the radial equilibrium distribution and is checked separately. The measurement
is laboratory tensile stress per molecule, from actual spring force.

prototype.py implements a Jacobi-weighted weak disk Galerkin solver and an
independent positive conservative polar Scharfetter–Gummel probability solver.
The latter integrates the nonlinear force directly. validate.py checks48
corner/random cases, sign symmetries, pure-rotation and r1 limits, constitutive
residuals, positive covariance, finite extension and flow power. The four FV
refinements reduce error by four; Richardson agreement is below3.3e-6 absolute.
Spectral stress degree24→28 changes below1.9e-12 absolute. A small negative
spectral tail mass up to3.24e-6 is disclosed; physical moments agree with the
positive independent probability solution. Selected nonzero-rotation cases have
37–64% Peterlin error across the fitted drag range.

The initial validate.py launch from the repository root failed to import its
sibling prototype module before any science ran. Launching from this directory
resolved only that path issue. No physical equation or data were altered.

The complete candidate is staged separately at staging/fene-stress-r2. Physics
review and root independently approved the concept and governing conventions.
Primary support: https://arxiv.org/abs/0902.4298 .
