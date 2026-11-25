# Viscoelastic waving sheet: numerical prototype

This is a scientifically checked prototype, not a task package or a measured
agent result. The final source and report are frozen for source review. No
canonical task, calibration file, shared ledger or model evaluation was changed.

The anchored sheet has material motion `x=X, y=a cos(kX-omega*t)`. The fluid is
incompressible Oldroyd-B with known density, solvent fraction `.25`, relaxation
time `1 s`, and unknown zero-frequency kinematic viscosity `nu0` in
`[.7,1.4] mm^2/s`. The tested wave numbers are `[.7,1.3] /mm` and angular
frequencies `[.5,8] /s`. Harmonic velocities are divided by amplitude, giving
`1/s`; mean pumping is divided by amplitude squared, giving `1/(mm*s)`.
Stationary response is taken before the far-field and small-amplitude limits.
The infinitesimal amplitude avoids any large-extension polymer instability.

The source retains exact complex-viscosity harmonic flow, wall Taylor expansion
and mean fluid convection. It uses the linear laboratory Jeffreys constitutive
law for polymer stress at both orders. This is a completed linear-response
approximation extended to a quadratic observable, not a generally objective
nonlinear material model. It differs from r1's omitted fluid momentum transport.

Write `G_ij=partial_j u_i`. The first-order conformation is
`C1=(G1+G1^T)/(1/lambda-i*omega)`. Its nonlinear mean stress is
`N=eta_p*<G1 C1+C1 G1^T-u1.grad(C1)>`. The exact mean shear balance is
`eta0*U2' + Nxy = rho*<u1*v1>`, with zero far-field imposed shear. The physical
pumping is therefore the wall Taylor velocity plus the integral of
`(rho*<u1*v1>-Nxy)/eta0`. The completed source omits only `Nxy`.

The integrated prediction uses exact decaying harmonic modes with stable
exponential differences. The independent reference solves the harmonic
vorticity boundary equations and then the second-order mean momentum BVP. It
assembles the conformation forcing componentwise and differentiates it locally;
it does not use the exact harmonic solution or the pumping integral. Tensor
kinematics are shared definitions, not a second independent constitutive law.

The final report checks 12 structured domain controls and 12 random controls.
Maximum pumping disagreement is `4.50e-9` relative; depth/tolerance refinement
changes the BVP result by at most `2.76e-9`. The broad domain includes accidental
near-equivalence (minimum relative error `.001794`), so it does not support a
uniform separation claim. A separate 27-point grid with frequencies `4,6,8 /s`
has source error at least `.76298` and physical pumping at least `.48830`.
These are sampled margins, not a proof over a continuous parameter domain.

The mean momentum identity, differentiated polymer stress, signed wave reversal
and positive harmonic work are checked. Wall power equals integrated volume
loss within `1.2e-15` relative on the three additional controls. Both Newtonian
limits (`beta=1` or `lambda=0`) remove the nonlinear polymer correction exactly.
For fixed `omega*lambda=2` and increasing viscosity, the pumping ratio approaches
`.4`, agreeing with the creeping-flow result
`(1+beta*De^2)/(1+De^2)` in [Lauga, Phys. Fluids 19, 083104
(2007)](https://arxiv.org/html/cond-mat/0703548). The final tested ratio at
`nu0=10000` is `.400096`; that control is a limiting check outside the proposed
fitted-parameter range.

Calibration uses signed complex first-harmonic velocities at 12 controls
(two wave numbers, three frequencies, two heights), hence 48 real readings.
The identity `U+iV=omega*exp(-s*h)` and the `.68851 rad` maximum phase difference
between heights `.5` and `.6 mm` prove global injectivity on the proposed range.
All 29 noiseless parameter recoveries have relative error below `6.10e-9`.
For fixed instrument sigma `.002 /s`, the worst local standard-error estimate is
`.000685 mm^2/s`; 64 synthetic noise draws at `nu0=1.05` have maximum fitted
relative error below `.001482`. This prototype is not the full 256-draw package check.

The first run used an unweighted `expm1` in an infinite quadrature tail. One
domain case produced NaN, and its aggregate maximum check failed to reject it.
The exact original code/report and explanation are preserved under
`history/first-tail-evaluation`. The final version uses weighted exponentials,
explicit finite checks and JSON serialization with `allow_nan=False`. All final
reported values are finite. This was a numerical defect during development,
not an agent outcome or evidence of physical model difficulty.

The rejected corotational variant and full archive overlap are preserved in
`staging/waving-sheet-r2-feasibility`. In planar incompressible harmonic flow,
its changed quadratic stress is isotropic and does not alter pumping. Archived
dumbbell-stress already tests affine Hookean transport; FENE r1/r2 test nonlinear
spring preaveraging. This is a related constitutive task and the same waving
apparatus as r1, whose two passes and one physical failure remain preserved.

The numerical evidence supports considering a full package after source peer
review. Empirical difficulty is unknown. No task build or evaluation is implied.
