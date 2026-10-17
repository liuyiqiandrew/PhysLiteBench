# Rotating reservoir, revision 6: collision selection by relative gas flux

This candidate retains a rotating gas reservoir and its heat calorimeter, but constrains the tracer to a unit ring with one tangential kinetic degree of freedom. The public apparatus specifies the complete relative-speed collision rate, equal-mass elastic exchange, motor process, and stationary mean/noise measurements. The common drag and its [.4,1.1] range, the bounded temperature/rotation/flip box, .04 prediction/.03 parameter/1.5 reduced chi-square gates, and 600/60-second limits remain unchanged. Scientific validation is distinct from the requested 0/3 unhinted gpt-5.6-luna high outcome.

## Previous evidence and distinct approximation

Revision 5 gave three physical passes and no exceptions. The submissions recognized and correctly implemented temporal heat correlations; the parent's 73 off-grid cases agreed below 1.2e-14. Its source, forward reference, rational fourth-moment certificate, reports, Docker controls, all native jobs/frozen trees/trajectories are preserved in archives/rotating-reservoir-r5, whose manifest verifies 212 files. Older revisions remain separately archived. This revision tests the collision-selection law, rather than an omitted noise correlation or tighter gate. Its completed source retains exact constant-clock moments and the full Poisson noise correction.

## Apparatus, heat and calibration

Tracer inertia, ring radius and gas effective mass are one. The ideal contact has exactly one tangential degree of freedom, with no undisclosed radial gas energy. Tracer energy is v²/2. Its radial constraint does no work. Thermostat A obeys dv=-gamma*v dt+sqrt(2*gamma*Ta)dW. The independent symmetric motor flips at kappa, with flow u=s*Omega. Flips change neither tracer velocity nor energy and record no instantaneous tracer work or collision heat; empty-reservoir motor costs are separate.

With lambda=3.6, eta=flux_fraction and fixed reference speed1, the exact differential event rate is

    W_s(v,dg)=lambda*((1-eta)+eta*|v-g|)*phi_Tb(g-u) dg.

It defines both total hazard and conditional colliding-mark distribution. Phi is the incoming spatial Maxwell density before selection. A collision exchanges v'=g,g'=v, conserving tangential momentum and laboratory energy. The calorimeter adds h=((v-u)²-(g-u)²)/2. The identity

    Delta E_particle+h=u*(g-v)

identifies flow-maintaining motor work. Velocity is physically unbounded. Mean measures lim E[Q_B(t)]/t; noise measures lim Var[Q_B(t)]/t in the stationary joint ensemble, including all event, mark, thermal and motor randomness.

The former two-dimensional trap calibration is intentionally replaced: its conductance differs from this ring's. There are 288 new records, calibration seed 71063, sigma=.001, four temperature contrasts, three flip rates and 24 replicates. All use eta=Omega=0 and default mean. The exact calibration law is

    mu=gamma*nu/(2*gamma+nu)*(Ta-Tb), nu=3.6.

If c is the fitted coefficient, gamma=c*nu/(nu-2*c), so the contrasts identify true gamma=.67. Public/private records agree exactly. No collision parameter is inferred. The validator reads existing records unless explicit --regenerate-calibration is requested, and verifies the intentional change from revision 5. The ring, every control and all event/heat rules are public.

## Exact generator and noise

For the backward velocity functions,

    L f_s=-gamma*v*f'_s+gamma*Ta*f''_s
      +kappa*(f_-s-f_s)+integral W_s(v,dg)*(f_s(g)-f_s(v)).

With stationary pi, define H1 f=integral W*h*f(g), H2 1=integral W*h², g_heat=H1 1, mu=pi*g_heat. The centered Poisson corrector satisfies L phi=mu-g_heat, pi*phi=0. The full variance rate is

    c2=pi*H2 1+2*pi*H1 phi.

The post-collision argument g in H1 is essential. The oracle uses a positive Scharfetter-Gummel finite-volume OU generator and literal midpoint jump quadrature. Normalized stationary and centered Poisson solves place the gauge at the largest probability, avoiding an ill-conditioned tail gauge. It combines 241/482 velocity cells by second-order Richardson extrapolation. Eta=0 uses an exact degree-four polynomial closure instead. Numerical tail boundaries are truncations, not physical walls.

The independent reference differentiates the continuous OU generator by Chebyshev collocation, eliminates endpoint Neumann derivatives, and integrates the literal jump kernel by split Gauss-Legendre quadrature at g=v. It uses barycentric interpolation and a left stationary functional. It imports no oracle or finite-volume flux. Separate91/121/151 node, quadrature and extent 7/9/11 refinements establish boundary/cusp convergence. Agreement of these different discretizations is supplemented by thermodynamics and analytic limits.

## Completed Maxwell approximation and diagnostic groups

The source replaces the actual flux-dependent clock by the equilibrium pair-average rate

    nu=3.6*((1-eta)+eta*sqrt(4*Tb/pi)),
    W_source(v,dg)=nu*phi_Tb(g-u) dg.

The average follows from independent v,g~N(u,Tb), whose relative speed has mean sqrt(4Tb/pi). This is a coherent Maxwell-molecule approximation applied to the specified hard-flux contact. It keeps all controls. Five monomials per label give exact stationary moments through degree 4; three per label give the exact quadratic Poisson corrector. All Gaussian moments, post-jump heat and temporal correlations are retained. The only error is averaging velocity-conditioned event selection. It is exact at eta=0 and therefore fits calibration and passes constant-rate anchors. The public starter differs only by leaving fit unfinished.

At Omega=0 and Ta=Tb=T, constant-rate equilibrium has mean zero, noise2*gamma*nu*T²/(2*gamma+nu), and shot term nu*T². These closed anchors check the correction sign and factor 2. In the isolated author limit gamma=0,Omega=0, Q_B=(v(0)²-v(t)²)/2 telescopes; mean and noise rates vanish even with hard flux. Fixed nonzero flow gives the corresponding rotating-frame bounded-energy identity; switching flow need not give zero noise.

Three four-case groups cover equilibrium, temperature bias and both directions of rotating hard-flux noise; a fourth includes constant-rate mean/noise anchors. Initial true-parameter source RMS errors are .10656,.09824,.26063 and roundoff for the anchor. The unchanged .04 gate uses each group's nonzero true RMS readout. This is not a claim that all allowed inputs discriminate.

## Full-domain stationarity and thermodynamics

For every even p>=2, OU gives -p*gamma*|v|^p plus lower-degree diffusion. Discarding negative jump loss gives the uniform upper bound

    L_jump |v|^p <= lambda*(M_p+|v|*M_p+M_(p+1)),

where M_q bounds incoming absolute Gaussian moments throughout the public box. Young bounds absorb lower powers into negative OU drift, giving L|v|^p<=-c|v|^p+C. The at-most-linear hazard and moment bounds preclude explosion; elliptic OU forcing, Gaussian reset irreducibility and motor switching give a unique geometrically ergodic stationary law. All moments exist, including the sixth controlling the flux times squared-heat integrand. An analogous exponential Lyapunov estimate for exp(a*v²), a<1/(2*max(Ta,Tb)), gives Gaussian tail control. These arguments apply throughout the domain; numerical corners supplement them.

Local detailed balance follows because the flux is symmetric under v,g exchange:

    W(v,dg)/W(g,dv)=phi_Tb(g-u)/phi_Tb(v-u)=exp(h/Tb).

Stationary Q_A=gamma*(E[v²]-Ta), Q_A+Q_B=P_motor, where P_motor=E integral W*u*(g-v). Thermal entropy Q_A/Ta+Q_B/Tb is nonnegative with motor work separate. The validator checks literal per-event work/heat and detailed balance, discrete and continuum stationary first laws, reversal, positive noise, normalization/conditional masses, tails and stationary/Poisson residuals. It examines 144 corner/drag preparations and 32 random interiors, alongside analytic limits and independent refinement. Numerical tail boundaries are not hidden apparatus.

## Reproduction and evaluation boundary

Run OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B scripts/validate_rotating_reservoir.py from this candidate root with NumPy, SciPy, pytest. The report results/rotating-reservoir-r6-validation.json records source hashes, timings, independent discrepancies, all 256 refits at unchanged noise seed 146824 and local verifier metrics. Both controls must fit calibration/parameters. Oracle passes all four groups; source fails the three flux groups and passes Maxwell anchors. Local public/private pytest expects oracle 9/9 and source 6 pass/3 physical failures under 60 seconds.

The initial exploratory finite-volume prototype's tail gauge was corrected before packaging; distinct exploratory versions remain preserved. Production code has no temporary prototype imports. The ordinary Docker image contains only README, starter, public test and calibration; solution, reference, hidden cases, true drag, author diagnosis and optional hint remain private. Parent actual-source physics review precedes freezing. Fresh Docker controls and all three unhinted Luna-high trials remain separate evidence, and every pass/error/failure must be retained.

## Recorded local validation

The packaged revision-6 validator passed in 72.48 seconds. The cold independent Chebyshev reference evaluated all 16 hidden inputs in .534 seconds. Oracle/reference disagreement was 1.264e-6; the largest independent node, quadrature or tail refinement was 8.819e-10, and the finite-volume grid refinement was 8.607e-7. Across all 144 corner/drag cases, the largest reference absolute discrepancy was 2.903e-6, relative noise discrepancy 1.627e-6 and continuum first-law error 3.841e-6. The independently discretized first law was accurate to 1.754e-12. All 32 interior comparisons agreed within 1.203e-6. These numerical errors are far below the .04 prediction gate.

The minimum noise was .235636. The positive finite-volume Markov generators gave a minimum computed probability of -2.351e-15, a roundoff residual; normalization/conditional masses agreed within 2.155e-13. Probability beyond the seven-thermal-scale region was at most 2.876e-12, and boundary-cell probability at most 8.022e-16. Maximum stationary and Poisson residuals were 6.496e-13 and 3.252e-9. The minimum computed thermal entropy was -2.418e-11 at its exact zero boundary. Exact Maxwell anchors, source first law, isolated telescoping noise and microscopic heat/work/detailed-balance identities passed at 1.066e-14, 5.330e-15, 2.579e-11 and 1.777e-15 respectively.

Across 256 noisy refits, oracle diagnostic error was at most .000781; the smallest completed-source diagnostic error was .097665, with maximum Maxwell anchor error .000192. Relative drag error was at most .000853 and reduced chi-square at most 1.20249. Isolated public/private pytest gave oracle 9/9 in 1.128 seconds and source 6 pass/3 intended physical failures in .848 seconds. No verifier or task limits changed. Fresh Docker and frozen Luna results remain separate evidence.
