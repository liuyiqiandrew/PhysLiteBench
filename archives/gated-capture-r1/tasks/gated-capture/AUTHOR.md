# Gated capture: author record

This staged candidate has completed scientific validation and local verifier controls. No model-agent trials have run, so its empirical difficulty is unknown. The public package uses the neutral instruction allowing replacement of the entire implementation, an unfinished fit with `diffusivity=None`, a fixed measurement uncertainty, and a .04 hidden normalized RMS gate. The completed shortcut is separate from the public starter.

## Physical distinction

The coating is one common external two-state gate. Absorption is linear, irreversible and nonsaturating, and switching is independent of particle positions and events. For the requested mean, the joint concentrations `c_i(r)=E[n(r) 1_(G=i)]` obey closed equations:

    D (c'' + 2 c'/r) + Q c = 0,
    Q = [[-k01, k10], [k01, -k10]],
    D c_i'(a) = kappa_i c_i(a),
    c(infinity) = pi c_infinity,
    pi = (k10, k01)/(k01+k10).

The positive capture rate is `4 pi a^2 sum_i kappa_i c_i(a)`. The common gate can correlate different particles, but those higher correlations do not enter this mean. The maintained reservoir and full joint stationary preparation are stated publicly.

The exact radial modes are

    c(r) = pi c_infinity + A pi a/r
           + B (1,-1) a/r exp[-q(r-a)],
    q = sqrt((k01+k10)/D).

A two-by-two Robin boundary system determines A and B. Summing the diffusive flux gives `J=-4 pi a D A`. This is the oracle's readout.

The completed shortcut solves the stationary spherical diffusion problem with `kappa_bar=pi_0 kappa_0+pi_1 kappa_1`. Its rate is

    J_bar = 4 pi a^2 c_infinity kappa_bar D/(D+a kappa_bar).

This is the fast-switching homogenization, with positive concentration, exact particle conservation and correct spherical geometry. It uses both rates through their stationary probabilities. It loses their overall timescale: finite gate persistence correlates surface concentration with the active reactivity. That is the physical approximation, rather than an incomplete boundary solver or an arbitrary ignored input. The prototype preserves the rejection of an earlier exact-joint-state/wrong-product-output idea; the staged source uses the internally consistent homogenized model instead.

## Calibration and grading

Calibration uses 18 distinct equal-reactivity settings at three radii, three nonzero reactivities and two pairs of switching rates, with 16 independent measurements of each setting (288 records). If both reactivities equal kappa, the two models agree for every D. The reaction-diffusion crossover identifies D:

    dJ/dD = 4 pi a c_infinity (a kappa)^2/(D+a kappa)^2 > 0.

The smallest derivative in the sampled allowed interval is .26507. Noiseless fits at 31 D values spanning [.7,1.3] recover D within 3.26e-8. Neither measurement sigma nor the controls encode the unknown D. Sigma is the fixed instrument value .01; the reference-generated clean data differ from the analytic oracle by at most .000330 sigma. The generated true D is .95; both controls fit .950125604 with reduced chi-square 1.00548.

The grader checks calibration, fitted D, three unequal-reactivity groups and an additional equal-reactivity anchor group. Each prediction must have the exact one-dimensional output shape and finite values. It does not inspect solver choices or compare implementation text. The prediction gate is .04, the parameter gate .03 and the calibration reduced chi-square gate 1.5.

## Independent reference and checks

The private reference integrates a radial first-order boundary-value problem for the two joint concentrations and radial diffusive fluxes. It imposes each inner Robin boundary directly and a deterministic outer reservoir. It extracts the inner reaction count, not the oracle's far-field mode coefficient. Three finite outer radii are extrapolated quadratically in inverse radius. No analytic infinite-domain mode coefficients enter this solver.

`results/gated-capture-r1-validation.json` records the complete checks:

- All 256 independent calibration-noise draws pass calibration and parameter gates for both controls. The largest oracle hidden error is .00020938; the smallest shortcut error among the three diagnostic groups is .432219. Equal-reactivity anchors pass both.
- At the supplied calibration, the shortcut hidden errors are .638952, 1.405187 and .432669. Signals are positive and at least 1.85547, so these are not near-zero normalization effects.
- Across 144 domain corners, including zero reactivity, the maximum independent reference error divided by `max(1,abs(rate))` is 2.301e-7. Joint profiles remain nonnegative and bounded by their reservoir values. Direct reaction flux and far-field supply agree.
- Doubling outer radii changes the three independently refined diagnostic rates by at most 2.926e-8 relative. Numerical inner reaction and outer diffusive flux agree within 1.78e-15 in the scored preparations.
- State relabeling, joint kinetic-rate scaling, the slow-switching mixture of fixed-state capture rates, and the fast-switching homogenized limit pass. The completed shortcut's own scalar Robin solution is also positive and conservative.
- The diagnostic shortcut gap stays above .40522 throughout the sampled D interval. The allowed domain uses finite reactivities and positive switching rates, avoiding singular perfect-absorption or reducible-gate limits.

`results/gated-capture-r1-local-controls.json` records actual local pytest runs: the oracle passes 7/7 in 1.02 seconds, while the shortcut passes four tests and fails precisely the three diagnostic groups in .87 seconds. These are deterministic author controls, not agent outcomes. Scientific checks took 4.24 seconds. The task keeps 600-second agent and 60-second verifier limits.

## Scope and prior work

The preserved prototype includes an archive scan of 254 AUTHOR files and direct comparisons with competitive adsorption, blocking ion transport, membrane partitioning and driven diffusion. No identical gated-reactive-boundary family was found. This task shares the broad idea of correlations surviving coarse graining with other candidates, but uses stationary reaction-diffusion at a common switching surface. It does not claim novelty for the underlying physics.

Bressloff's [Stochastically switching diffusion with partially reactive surfaces (2022)](https://arxiv.org/abs/2205.13985) treats a switching conformational state of each diffusing particle. That apparatus differs from the common surface gate specified here. Its linear one-particle reaction-diffusion framework is relevant, while the common-gate first-moment closure above is an explicit adaptation: independence of switching rates and linear uptake close the same equations for `E[n 1_G]`. The equivalence does not extend to general multiparticle correlations or capture-count noise, which are not requested.

Szabo, Shoup, Northrup and McCammon's [Stochastically gated diffusion-influenced reactions, J. Chem. Phys. 77, 4484–4493 (1982)](https://doi.org/10.1063/1.444397) is a bibliographic background reference. Its full text was not obtained in this work. The source derivation and independent reference above provide the validation for this specific apparatus.

The original prototype files and their manifest are retained unchanged. The source provenance freezes this final package separately; numerical results make no prediction about whether Luna will recognize the approximation.
