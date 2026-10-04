# Circular detector, revision 1

Author-only. The task distinguishes a local worldline-correlation approximation from the full stationary vacuum response. The source is not a thermal Unruh guess: it has a positive nonthermal spectrum, the exact vacuum spontaneous-emission commutator, the correct cusp calibration, and the circular interval through sixth order in proper-time separation. All known circular inputs affect it.

[Hodgkinson and Louko](https://arxiv.org/abs/1109.4377) discuss regulator-free detector-response integrals. [Biermann et al.](https://arxiv.org/abs/2007.09523) study the dependence of circular detector response on radius and gap as well as acceleration. These support the stated weak-coupling, stationary, pointlike scalar-field model. The public task defines the canonical field, positive-frequency vacuum, complete worldlines, transition direction and proper-time readout. It does not state an approximate or corrected response formula.

## Physical response and the source closure

For a stationary timelike trajectory, write D(s) for the positive squared timelike separation at proper-time lag s. In 3+1-dimensional massless Minkowski vacuum, the excitation rate per unit coupling is

    R_up(E) = (1/(2*pi²)) integral_0^infinity cos(E*s) [1/s² - 1/D(s)] ds.

The subtraction removes the inertial coincident singularity. The downward rate is R_up(E)+E/(2*pi). The latter commutator relation alone does not determine the excitation spectrum.

The cusp worldline has D(s)=s²+a²*s⁴/12 exactly. Its Fourier transform gives a/(8*pi*sqrt(3))*exp(-2*sqrt(3)*E/a). Calibration includes both upward and downward rates for this actual trajectory, not just a supplied response law.

For the circle,

    D(s)=gamma²*s²-4*R²*sin(gamma*v*s/(2*R))²,
    R=gamma²*v²/a.

Its local expansion is s²+a²*s⁴/12-a⁴*s⁶/(360*gamma²*v²)+O(s⁸). The completed source uses

    D_local(s)=s²+a²*s⁴/[12*(1+c*s²)],
    c=a²/(30*gamma²*v²).

This is positive for every real nonzero s and matches the expansion through s⁶. Its exact Fourier excitation rate is

    a²/(48*pi*sqrt(c+a²/12))*exp[-E/sqrt(c+a²/12)].

It also adds the exact spontaneous term to downward transitions. The resulting positive spectrum and commutator are consistent with a stationary Gaussian correlation approximation. It reduces exactly to the cusp response when c=0. It is not the vacuum correlation restricted to the prescribed circular trajectory. The missing physics is the global trajectory dependence of the sampled field, not a failed Fourier transform or numerical quadrature.

The oracle keeps the full circular interval. It rescales lag by a, evaluates 1-sinc(x)² with its regular series near zero, and uses adaptive cosine quadrature of the inertial-subtracted correlator. The public speed interval stays below .85, and all gaps are positive. The prescribed small coupling and stationary leading-order measurement exclude switching transients, recoil and population depletion.

## Independent reference and checks

The reference expands the vacuum field in cylindrical modes about the circular orbit. Define Omega=v/R and omega_m=m*Omega-E/gamma. For either signed transition energy E, only omega_m>0 contributes. The rate is

    sum_m omega_m/(2*pi*gamma) * integral_0^(pi/2)
        sin(theta) * J_m(omega_m*R*sin(theta))² dtheta.

This mode calculation does not use the time-domain interval, its Taylor expansion or a local response correction. In particular, downward rates are computed directly with negative signed gap, independently checking the commutator. The cusp reference uses numerical Fourier integration of its regular correlator instead of the analytic source formula.

Across hidden inputs and allowed corners, the independent mode response and oracle differ by at most 2.05e-12 before multiplying by coupling. Raising the mode cutoff from400 to600 changes no reported digits, and angular order80 to112 changes results by at most7.78e-16. The direct mode commutator agrees to2.78e-16. Both rate directions remain positive. Scaling both acceleration and gap scales the rate correctly. Halving lag reduces the source's interval error by factors251–256, consistent with an eighth-order mismatch. The ultrarelativistic fixed-acceleration limit approaches the cusp rate; at v=.9999 the relative discrepancy is2.93e-5. No integration warnings occur in the tested domain.

## Calibration and uncertainty

The unknown coupling is .012, allowed [.006,.02]. There are144 cusp records: four accelerations, six gap/acceleration ratios, both transition directions and three repetitions. Instrument sigma=5e-7 is fixed independently of the unknown coupling or clean rate. Generation seed947081 and noise seed947083 are private. Generation is explicit with `--generate`; ordinary validation preserves the frozen calibration. Public and private copies are identical.

The checked-in fit is .0120002550454 with reduced chi-square1.05546. Correct hidden normalized errors are about2.13e-5; source errors are .1553, .1500 and .1449. All256 independent noisy calibrations pass calibration, parameter and oracle gates. Worst oracle error is8.72e-5 and the smallest source error is .14483, against the standard .03 group gate. These are calibration-noise checks at fixed hidden inputs, not exhaustive hidden-domain sampling.

The actual local harness passes7/7 for the oracle; the completed source has4 passes and the3 intended hidden failures. Runs take about1.1s. The instruction is byte-identical to the neutral task instruction; no algorithm or model-form preservation is required. Docker copies only README, starter, public tests and calibration. The hint, reference, private tests and this author note remain outside the agent image. No agent evaluation is claimed before the root launches a frozen batch.

Reports are `results/circular-detector-validation.json`, `results/circular-detector-local-controls.json` and `results/circular-detector-source-provenance.json`. Final independent source review is recorded in `results/materials-hardening-physics-review.json`.
