# Rotating reservoir revision 8: finite recirculating contact

Revision 7 produced three physical passes, including one original 600-second solving timeout whose installed source passed. Its full independent native-parameter review and Gaussian-SCGF ablation are preserved in results/rotating-reservoir-r7-trial-review.json. All 389 predecessor files, original plain/oracle/shortcut jobs, native sessions, exceptions, science and nested dependencies are verified under archives/rotating-reservoir-r7. No revision-7 trials were repeated. This revision changes the actual reservoir to one retained gas contact that carries recoil into later thermalization and collisions.

## Complete apparatus and thermal measurement

The public apparatus supplies a unit-inertia rotor velocity v, a retained unit-inertia gas velocity g and an independent motor s=+/-1. OU bath A thermalizes v at unknown common drag gamma and Ta. OU bath B thermalizes g at known lambda toward u=s*Omega and temperature Tb. Independent random collisions at nu[(1-eta)+eta|v-g|], nu=3.6, elastically swap v and g. They conserve total laboratory energy and tangential momentum. No particle is removed or redrawn. Motor flips at kappa change only the bath-B flow target, with no particle impulse, instantaneous particle work or heat. The public README states every state, stochastic increment, collision rule, measurement and parameter bound; no hidden interaction or preparation is introduced.

The thermal bath-B calorimeter follows the shared-Wiener Ito current

    dQb = lambda[(g-u)^2-Tb]dt - sqrt(2lambda Tb)(g-u)dWb.

This is minus the relative velocity times its thermostat force in Stratonovich form. Collisions and flips have zero instantaneous Qb. The mean is lambda(E[(g-u)^2]-Tb); the noise is lim Var(Qb(t))/t in energy squared per time. Continuous motor work is u times the thermostat force, with mean P=lambda(Omega^2-Omega E[s g]). Bath A mean heat is JA=gamma(E[v^2]-Ta), and JA+JB=P at stationarity. The same Wb drives both g and Qb, so its response cross term cannot be replaced by an independent detector noise.

The unknown remains gamma in [.4,1.1]. Known public bounds are Ta in [.4,2], Tb in [.4,1.4], lambda in [.2,3], Omega in [-.65,.65], kappa in [.8,2] and eta in [0,1]. Mean remains the default readout; the other readout is thermal heat noise. The preceding fresh-gas finite-bias readout is not advertised for this different calorimeter. The function API and array convention are retained.

## Deliberate calibration regeneration and identifiability

All 288 revision-7 temperature/motor calibration preparations are retained, with known relaxation_rate=1.3 added, eta=Omega=0 and the same sigma=.001, calibration_seed=71063 and noise_seed=146824. Finite contact changes the mean conductance, so all calibration values are deliberately regenerated in matched public/private files. Their individual Gaussian noise realizations are retained. Reusing the old fresh-gas values would be physically incorrect.

For the constant clock and zero circulation, exact second moments x=E[v^2] and y=E[g^2] obey

    (2gamma+nu)x-nu y=2gamma Ta,
    -nu x+(2lambda+nu)y=2lambda Tb.

Thus JB=C(Ta-Tb), C=gamma lambda nu/[2gamma lambda+nu(gamma+lambda)]. Its derivative in gamma is strictly positive for positive lambda, so the known lambda and nonzero contrasts identify gamma. Fitting the uncertainty-weighted slope gives gamma=C lambda nu/[lambda nu-C(2lambda+nu)]. The private true gamma=.67 appears only in verifier metadata/reference. The public source has no true-parameter default. Its Model.fit is intentionally unfinished; the separate completed approximation implements the fit.

## Completed Gaussian moment/response approximation

The public source and scripts/rotating_reservoir_baseline.py implement the same Gaussian moment/response physics, with only the source's unfinished fit differing. They retain both velocities, finite gas relaxation, recoil, motor-conditioned means and the full pair covariance. They do not reset the gas, ignore eta or kappa, or use independent collision shot noise.

Their five exact mean/second-moment equations are closed by a motor-conditioned joint Gaussian for the flux averages. With D=v-g and S=v+g, the necessary folded-normal integrals E[|D|D] and E[|D|DS] are evaluated analytically. At eta=0 that assumption disappears from these low-order equations: all first/second moments and mean calibration are exact even though the joint stationary law need not be Gaussian. The source then uses its assumed stationary Gaussian law to project the exact kinetic generator onto six quadratic observables and evaluate a quadratic Poisson response. It includes the correct shared-Wiener term as well as the direct noise. This is an explicit Gaussian moment/response approximation, not an assertion that this projected generator is an exact realizable Gaussian Markov path law. Positive covariance, projected decay, nonnegative noise, moment stationarity and energy balance are checked over the full box.

There is no exact noise branch or high-order answer in the public source. In particular, eta=0 noise is still approximated; no discontinuous exact eta=0 exception conceals its nonGaussian law. Completed calibration, physically defined current, coherent assumed moment/response equations and accurate quadrature distinguish this control from a broken formula or unfinished numerical implementation.

## Exact kinetic calculation and independent reference

The private oracle solves the continuous joint kinetic generator in normalized Hermite polynomials. The motor-conditioned minus density is the simultaneous velocity reflection of the plus density; the reduced generator includes kappa[f(-v,-g)-f(v,g)]. Under an isotropic Gaussian integration weight, the elastic swap is reversible. Rotating to relative and total velocity and substituting z=+/-sqrt(2 variance t) lets Gauss-Laguerre/Hermite quadrature integrate the |v-g| collision matrix exactly within the finite polynomial subspace. The basis variance is the positive known max(Ta,Tb), so cold equilibrium uses its actual thermal weight. This numerical basis choice introduces no extra physical input. Degree34 is checked against degree42, a fixed positive alternative weight, equilibrium controls and an independent grid method. At eta=0 the degree-four polynomial generator closes exactly, including the motor, so that private branch is an exact special limit of the same physics.

For the continuous calorimeter, the first two Feynman-Kac generator derivatives are

    K1 f = 2lambda Tb(g-u)partial_g f - lambda[(g-u)^2-Tb]f,
    K2 f = 2lambda Tb(g-u)^2 f.

The oracle solves stationary moments and L r1=(-mean)1-K1 1 with stationary zero gauge, then evaluates noise=<K2 1>+2<K1 r1>. The derivative term includes the shared-Wiener correlation. No collision or motor heat mark is added.

The self-contained verifier imports no oracle, prototype or archive. It uses a positive Cartesian finite-volume generator: independent Scharfetter-Gummel OU transitions, exact node swaps and exact simultaneous node reflections. It marks ONLY bath-B OU edges by minus the relative kinetic-energy difference. Its stationary probabilities and full marked-edge Poisson corrector give heat noise without Hermite polynomials, continuous Feynman-Kac matrices or Gaussian closure. Richardson grids81/162 remove the leading mesh error. Their finite-mesh calorimeter is an approximation to the unbounded continuum apparatus, not a physical velocity cutoff. Extent and resolution refinement are checked independently.

## Independent anchors and causal distinction

The root teammate independently implemented the full degree-four raw monomial generator and degree-two Poisson response at eta=0, including arbitrary circulation and motor persistence, in results/prototypes/rotating-reservoir-r8-root/exact_constant_collision.py. It agrees with the independently assembled Hermite method over 96 preparations. For the simpler zero-flow anchor, let A=E[v^4], B=E[g^4] and C4=E[v^2 g^2]. Exact equations give

    (4gamma+nu)A-nu B=12gamma Ta x,
    -nu A+(4lambda+nu)B=12lambda Tb y,
    C4=(gamma Ta y+lambda Tb x)/(gamma+lambda),
    C4-x*y=-nu(x-y)^2/[2(gamma+lambda)].

Thus the retained contact develops negative energy correlations and nonzero fourth cumulants even when its first/second moments fit perfectly. A zero-mean Gaussian at the same x,y has C4=x*y and B=3y^2. The exact quadratic Poisson coefficients are av=lambda nu/D and ag=lambda(2gamma+nu)/D, D=4gamma lambda+2nu(gamma+lambda), yielding

    noise=2lambda Tb y
          +2lambda[av(C4-x*y)+ag(B-y^2)]
          -8lambda Tb ag y.

At eta=Omega=0 its quadratic response is exact, so replacing only those Gaussian fourth moments by the exact moments repairs the noise at unchanged gamma and unchanged calibration. At finite circulation the quadratic response still closes at eta=0, and nonGaussian conditional third moments can also contribute alongside the fourth moments. At Ta1.6,Tb.4,lambda2,Omega0, the exact noise is .8226015627733412; the approximation is .6806016007922262, a 17.26% miss. At Omega.6,kappa.8 the independent exact noise is1.3728843451653558, versus1.169358893270394, a14.82% miss. This identifies the physical source of failure before the additional flux kinetics is needed.

Uniform-temperature, zero-flow noise obeys equilibrium FDT: noise=2T^2*dJB/dTa. For eta=0 it reduces to 2C T^2; the root independently checks mixed/full flux. With gamma=Omega=0, Qb is minus a bounded total-energy difference, so its asymptotic noise is zero. The oracle/reference also check exact momentum/energy conservation of swaps, motor reversal symmetry, heat/motor balance, nonnegative stationary thermal entropy production, covariance/noise positivity and numerical refinement.

## Valid domain, groups and evidence

For E=(v^2+g^2)/2, elastic swaps and motor flips leave E unchanged. The generator obeys LE <= -min(2gamma,lambda)E + gamma Ta+lambda Tb+lambda Omega^2/2. For exp(aE), choose a=.2 throughout the box: the leading terms are -a gamma(1-aTa)v^2 and -a lambda(1-aTb)g^2, with positive uniform margins. Completing a square bounds the flow term. Nondegenerate OU diffusion and this energy control give a unique stationary law with finite energy moments and a well-defined stationary heat-noise response. The symmetric collision intensity is invariant under the swap and does not introduce an inelastic heat port.

Three predeclared four-case groups examine constant-contact nonGaussian energy noise, partial relative-flux contact noise and full-flux contact noise. At least one group has exact constant-clock anchors, isolating the correlation error before flux numerics; the others probe the same retained-contact physics. Four mean/equilibrium-noise anchors protect the calibrated and uniform limits. No near-zero-current relative amplification is used: noise denominators are finite. The inherited parameter .03, calibration chi-square1.5 and prediction relative RMS .04 gates and600/60-second resource limits are unchanged.

The approved proposal audit is results/rotating-reservoir-r8-finite-contact-proposal-audit.json, its independent root review is results/rotating-reservoir-r8-proposal-peer-review.json, and all positive and negative screens are retained under results/prototypes/rotating-reservoir-r8-brownian. Mean-only separation was too weak, including strong thermal-contrast attempts; it was not used as a diagnostic. Initial fixed-weight numerical diagnostics are preserved. Proposal gaps were15.6-16.4%, independent grid/continuous errors about1e-4relative, and14-case cold grid evaluation8.31seconds. Final actual-source science,256 direct noisy fits, local oracle/shortcut controls and complete source hashes are recorded in results/rotating-reservoir-r8-validation.json.

Reproduce after packaging with `OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B scripts/validate_rotating_reservoir.py`. No paid evaluation is authorized by this document. Fresh Docker controls and exactly three frozen unhinted Luna High trials are separate steps, performed only after final peer review and freeze. Source difficulty is an empirical result, not guaranteed by the physical separation.
