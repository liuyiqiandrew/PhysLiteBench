# Survivor transport, revision 1

The supplied predictor solves the exact radial capture problem and uses the resulting late-time survivor density to average the liquid velocity. That average is a valid instantaneous endpoint measurement. The apparatus instead measures accumulated axial displacement among particles selected to survive the entire interval. Earlier points on those paths are selected by their subsequent survival as well as their preceding evolution. This changes the drift even though the radial state, loss exponent and liquid flow are already correct.

This package has scientific validation and isolated local controls. No model-agent outcomes are claimed. The exact neutral instruction allows replacing the entire implementation. The public fit is unfinished, the diffusivity starts at None, and no fitted truth or physical correction is exposed.

## Physical ensemble and readout

The tube is unbounded axially. A finite pulse begins at z=0 with uniform transverse density. Particles diffuse isotropically, advect with a prescribed plug-plus-Poiseuille velocity and disappear irreversibly at a Robin wall. The wall translates at the plug velocity, so the relative Poiseuille profile respects its tangential velocity. The radial reaction boundary is unchanged by this axial translation. Capture is nonsaturating, with no reinjection or stored wall state. The known profile makes a separate liquid-flow solver unnecessary.

Let L be the self-adjoint radial diffusion generator in area measure `2*pi*r dr`. Its principal mode is

    phi(r) = J0(q*r/R),
    q*J1(q) = (capture*R/D)*J0(q),
    lambda = D*q²/R².

The relevant root lies between zero and the first positive J0 zero; capture0 gives q0. Endpoint survivor density is proportional to phi. An interior point of a long surviving path receives a forward propagation factor and a remaining-survival factor. Their product is proportional to phi squared. Therefore

    source drift   = integral(u*phi*r dr)/integral(phi*r dr),
    physical drift = integral(u*phi²*r dr)/integral(phi²*r dr).

Both are ordinary positive-weight averages bounded by the minimum and maximum prescribed velocity. Both reverse sign on flow reversal and acquire the same additive shift under plug translation. They share the exact loss exponent. The source is a definite endpoint-distribution closure for the accumulated readout, rather than an inaccurate radial equation or a malformed conditional probability. It need not be represented as the complete path law of the physical captured process.

The readout is `lim E[z(t)-z(0) | alive at t]/t`. It is not the derivative of an unconditioned displacement, the mean instantaneous endpoint velocity, or a first-passage flux. The asymptotic drift is independent of the positive initial ground-mode overlap. A finite-time conditional mean can have a constant intercept; the validator compares successive late-time slopes rather than asserting an exactly zero intercept. Ordinary axial diffusion contributes no mean drift.

## Calibration and grading

Calibration contains21 settings and288 readings: nine positive-capture loss settings at three radii and three capture rates, repeated16 times; eight positive-capture plug-flow drifts and four reflecting Poiseuille drifts, each repeated12 times. Both controls coincide exactly on every setting for every allowed D. The latter two sets test the drift interface and flow sign while loss measurements identify the parameter.

At fixed physical capture rate, the principal loss is the minimum of the Robin Rayleigh quotient

    [D*integral(|grad phi|²)+capture*surface_integral(phi²)]/integral(phi²).

Its D derivative is `integral(|grad phi|²)/integral(phi²)>0` for positive capture. The complete calibration objective has one minimum over D in [.7,1.4].25 noiseless full-range fits, including both endpoints, recover D within6.49e-9 relatively. The smallest sampled loss sensitivity is .05445.

Instrument sigma is fixed at .005 inverse seconds for loss and .002 micrometers/second for drift. It depends only on the readout units, not the response or unknown. Private true D is1.05; calibration and noise seeds are271021 and271027. Data are generated from the independent reference and copied identically to public and private locations. Its numerical calibration bias is at most2.72e-6 sigma. Both controls fit D=1.050295509 with reduced chi-square1.10585.

The standard prediction gate is .04 group normalized RMS; parameter recovery is3% and reduced chi-square must be below1.5. The grader checks values, shapes and finiteness without restricting solver choices. Three diagnostic groups use ordinary Poiseuille flow with zero plug translation and both flow signs. Their reference absolute drift signals are at least .5336 micrometers/second; there is no background subtraction or near-zero denominator. Nominal source errors are .12652, .12721 and .12471. A separate shared-anchor group passes both controls.

## Independent reference and checks

The private reference discretizes radial diffusion conservatively on annular cells. Face conductances, cell areas and a half-cell Robin resistance determine the generator directly. A square-root area transform makes it symmetric. Adding `s*u+D*s²` forms the axial moment-generating operator. The derivative of its principal eigenvalue at s0 gives the measured asymptotic drift. This reference does not use the Bessel mode or squared-mode quadrature. Counting-field and256/512-cell Richardson extrapolations are independently refined.

The full validation report records:

- 64 corner and random cases across the public parameter domain. Maximum oracle/reference error divided by `max(1,abs(reading))` is5.00e-9. Modes are positive, loss exponents nonnegative, and both drift averages stay within the imposed velocity range.
- 512/1024 refinement changes of at most7.41e-9 on the same normalization. Capture flux balances total radial loss within8.77e-13.
- An independent zeroth/first axial-moment propagation from uniform injection. Late successive slopes approach the oracle within7.34e-6 relatively on its128-cell mesh, with positive principal spectral gaps.
- Exact reflecting/plug-flow calibration agreement, flow reversal and uniform velocity shifts. Current controls match the preserved192-point prototype to2.58e-14.
- 256 independent noise draws: both fits pass calibration and parameter tests; all oracle predictions pass; all completed shortcuts fail each diagnostic group. Maximum oracle hidden error is .0002366 and minimum source diagnostic error .12468. The source gap stays above .12401 across the entire fitted D interval.

The actual isolated local verifier gives oracle7/7 in .39seconds and shortcut4passes/3 intended diagnostic failures in .40seconds. Scientific validation took3.35seconds. The ordinary600-second agent and60-second verifier limits are unchanged. These deterministic controls establish correctness and separation, not empirical agent difficulty.

A trial retaining endpoint-density averaging for the accumulated drift is a physical-model failure. A trial selecting the correct surviving-path ensemble but making a Bessel, radial-measure, spectral-derivative or numerical error must be classified separately. A fixed-parameter causal repair, if needed after evaluation, changes the drift weighting while keeping the fitted D, radial mode, loss and calibration unchanged.

## Scope and sources

The preserved prototype contains a broad archive keyword scan and five direct author-record comparisons. Archived tube-dispersion discards shear-history variance under reflecting walls. Here the absorbing radial solution is exact and the failure concerns survival-selected mean displacement. Gated-capture concerns a switching coating's stationary uptake, and fixed-trap-memory concerns repeated visits to fixed random rates. The present apparatus has neither switching nor disorder. Pressure-surface-waves and acoustic-cavity-pressure concern mechanical boundary or mean-state constraints. Related transport and conditioning ideas are acknowledged; this is not a claim of a new mathematical principle.

[Biswas and Sen, Taylor dispersion with absorbing boundaries: A Stochastic Approach](https://arxiv.org/abs/cond-mat/0701392), equations6–14 and their discussion, derive the surviving-path weighting and explicitly distinguish it from endpoint averaging. Their displayed example is planar; this circular Robin implementation and the annular reference are independently checked. [Basic characteristics of Taylor dispersion in a laminar tube flow with wall absorption](https://doi.org/10.1016/j.ijheatmasstransfer.2017.02.051) gives primary background on tube loss and transport moments.

Reproduce the scientific checks from the repository root with:

    uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/survivor-transport-r1/scripts/validate_survivor_transport.py

Add `--generate` only for intentional data replacement. The prototype files are preserved unchanged. Root owns Docker controls and any frozen model evaluations.
