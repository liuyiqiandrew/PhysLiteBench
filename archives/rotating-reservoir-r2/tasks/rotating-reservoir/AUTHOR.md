# Rotating reservoir, revision 2

This revision adds a finite compliant contact and a deforming reservoir flow. The source already has the successful r1 frame correction and the exact full Gaussian state. It reports particle-relative mechanical transfer as the outer bath's calorimetric heat. The finite connector also receives elastic work from the spatial velocity gradient, which reaches the outer thermostat in the stationary state.

The preceding r1 remains unchanged. Its original neutral initial trial failed physically; the later fresh three produced one physical pass and two physical failures. The passing trial fu84PgC explicitly replaced laboratory power by the relative-velocity heat at the particle. The two failures retained laboratory power with equally accurate fitted drag. The complete read-only evidence review, all native input hashes and the original Maxwell-memory exploration are indexed in the preserved [bounded review](../../prototype/ASSESSMENT.md). This is a same-family scientific revision, not an unchanged retry or a new independent family. No r2 model evaluation has occurred.

## Apparatus and physical heat

The main particle has mass one, laboratory position r, velocity v, harmonic stiffness K and stationary bath A with unknown drag gamma. An overdamped coupler q is linked to it by a spring h=b/tau, where b=.9. Bath B acts on q with affine flow `u(x)=A x`, `A=Omega R+diag(s,-s)`. Its thermostat and flow-maintaining actuator are separate. All forces, independent noises, physical contacts, units and stationary preparation are given publicly.

Set `z=-h(r-q)`, the spring force on the main particle. The exact transformed dynamics are

```
dr = v dt
 dv = (-K r - gamma v + z)dt + sqrt(2 gamma T_a)dW_a
 dz = [A z - z/tau - (b/tau)(v-A r)]dt + sqrt(2b T_b)/tau dW_b.
```

The sign of the last independent Wiener process is immaterial. The supplied predictor solves the complete six-dimensional Lyapunov equation. It retains affine stretching, all thermal noise, particle–coupler correlations and the exact parameter dependence. Its output is the completed mechanical transfer

`Q_source = -<z dot (v-A r)>`.

This extends the previous successful moving-frame repair. It is a well-defined particle-relative power, but it is not generally the heat at the coupler's bath contact. With spring force `F_q=-z`, the actual bath-B heat increment is the Stratonovich work `F_q dot (dq-Aq dt)`, positive into the bath. Its stationary mean is

`Q_B = <|z|²>/b - 2 T_b/tau`.

This convention is ordinary local bath heat at q. The white-noise subtraction follows from the two coupler coordinates. Only a stationary mean is requested, so no finite-window kinetic boundary statistic is implicit in the overdamped q model.

For the stored connector energy `U_s=tau|z|²/(2b)`, Itô energy balance gives

`d<U_s>/dt = (tau/b)<z dot A z> - <|z|²>/b - <z dot (v-A r)> + 2 T_b/tau`.

Therefore, at stationarity,

`Q_B = Q_source + (tau/b)<z dot E z>`,

where E is the symmetric part of A. Rotation alone makes the extra term zero. Strain does mechanical work on the finite connector separation. This persists at long recording times; it is not a missing stationary endpoint-energy term. The actuator power is `<z dot A q>`. With `Q_A=gamma*(<|v|²>-2 T_a)`, the first law is `Q_A+Q_B=<z dot A q>`.

Primary background is Speck, Mehl and Seifert, [Role of External Flow and Frame Invariance in Stochastic Thermodynamics](https://arxiv.org/pdf/0712.0232), especially equations2–3 and12. The author manuscript was read, including its distinction between bath-relative heat and elastic work in a flow. It does not derive this mixed inertial/overdamped apparatus; the explicit construction and balance above do that.

## Calibration and inference

All calibration has zero strain but includes finite contact time, anisotropic traps, both rotation signs and nonzero thermal biases. The source and physical predictions coincide exactly for every gamma. Static-flow settings also have

`Q_B=(T_a-T_b) sum_i b/[k_i*tau²+(gamma+b)*tau+1+b/gamma]`.

An included setting has tau=.2 and T_a−T_b=.6. Its derivative is strictly positive for gamma in [.4,1.1], since `b/gamma²−.2>0` throughout. A conservative derivative lower bound is .04150. This setting alone proves global identification; the remaining records enrich calibration. Forty-one noiseless fits across the whole interval recover gamma within7.22e-9.

There are20 distinct settings, independently repeated16 times for320 records. The fixed instrument sigma=.001 is independent of response and gamma. Public and private data are identical. Private true gamma=.73, calibration seed173051 and noise seed173053 are excluded from the public image. Both completed controls use bounded uncertainty-weighted fitting and the same None default/fit API as the unfinished public model.

The frozen data give gamma=.7294810874 and reduced chi-square .915787. Oracle group errors are at most .000201; the three diagnostic source errors are .52759, .46317 and .40152. The source also passes the shared rotation anchor. The gates remain3% parameter error, reduced chi-square1.5 and4% group RMS-normalized prediction error.

All256 independent noise fits pass calibration and parameter checks. Every oracle passes every hidden group; every source fails the three diagnostic groups and passes the anchor. Worst oracle error is .000432, minimum source diagnostic error .40138 and maximum parameter error .1532%. Across41 gamma values, group separation remains at least .39124. Diagnostic physical heats stay above .25407 and source powers above .09845. Healthy signals, rather than a near-zero normalization, account for the separation.

## Continuous stability and independent reference

The actual (r,v,q) drift is affine in gamma, kx, ky, h=b/tau, Omega and strain. The full allowed box has64 vertices. A common quadratic matrix P was found and rounded to eight decimal places. Exact rational LDL factorizations prove both `P-I/1000` and `-A_drift.T P-P A_drift-I/1000` positive definite at every vertex. Convexity then proves stability over the entire box. Temperatures affect only noise. There was no reduction of the previously proposed strain or contact-time ranges to obtain this proof.

The certificate, exact verifier and initial numerical search are preserved in [development](../../development/exact-stability-certificate.json). The development search is not an agent dependency. The task uses only NumPy and SciPy.

The oracle works in (r,v,z) coordinates using a Lyapunov solve. The independent reference builds the actual (r,v,q) equations, propagates independent bath impulses, integrates their covariance over time, and evaluates the local q-contact heat. It does not use the source's relative-work output or the added elastic-stress identity. All16 graded cases are checked independently and refined in integration tolerance. Scored disagreement is8.89e-15; calibration numerical bias is2.67e-11 sigma. Ninety-six additional corner/interior reference cases have maximum absolute error3.46e-13.

Additional checks cover256 domain corners and128 interiors, positive covariance, nonnegative total reservoir entropy production, independent-coordinate drift residual, first law, exact equilibrium Gibbs covariance, rotation reversal and static conductance. Their balance errors are below4e-14. The small-contact-time limit recovers the instantaneous moving-bath model and removes the connector discrepancy. Values below the allowed .2 contact time are labeled author-only limits.

The bounded precursor's exploratory tau=.9 and1 cases remain preserved outside the package domain [.2,.8]. Its initial results and the sampled stability qualification were not overwritten; the later exact certificate resolves that qualification for the package.

## Scope, prior overlap and causal classification

The archived dumbbell-stress task passed3/3 plain and3/3 hinted by restoring affine covariance transport. This source already includes that transport. Thermal-bodies r11 passed3/3 by restoring bath correlations; these are also present. Thermal-bodies r12 passed3/3 by adding stored-energy terms to finite-time heat variance; this task measures a stationary mean affected by continuous connector strain work. The shared spring/flow background is acknowledged. It is not evidence that the new task will be difficult.

The public apparatus fixes the contact location and thermostat signal without giving the heat formula or a solver recipe. The exact generic instruction permits replacement of every implementation component. The Docker boundary copies only README, starter, public test and calibration data. A solver that derives the q-contact heat but makes a trace, sign, parameter or numerical error must be classified separately from the intended physical readout failure. The earlier successes mean this revision may still be solved readily; no empirical difficulty is claimed before evaluation.

Scientific/local reports are [validation](../../results/rotating-reservoir-r2-validation.json) and [controls](../../results/rotating-reservoir-r2-local-controls.json). Local oracle passes7/7 in .82seconds; the completed source passes4 tests and fails exactly the3 diagnostic tests in .83seconds. The original600/60-second agent/verifier limits remain unchanged. Final hashes and the independent source/input review are recorded in the stage's results directory.

Reproduce author checks with `uv run --no-project --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python -B staging/rotating-reservoir-r2/scripts/validate_rotating_reservoir.py`. Add `--generate` only when deliberately regenerating calibration. No Docker or model calls are part of that validator.
