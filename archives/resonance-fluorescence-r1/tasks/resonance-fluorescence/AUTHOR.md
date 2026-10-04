# Resonance-fluorescence counting

The source keeps the full optical Bloch dynamics and stationary emitter state,
but uses population noise as nondestructive emission-intensity noise. It adds
Poisson shot noise to the integrated symmetrized excited-population covariance.
This is a coherent semiclassical second-cumulant approximation. The integrated
symmetrized covariance is nonnegative, so its variance is at least its mean.
It is not claimed to specify a positive classical intensity process at every
higher order. The physical counter measures actual radiative emissions, whose
subsequent correlations use the emission-conditioned emitter state.

Both controls fit the mean counts exactly for every calibration setting. The
mean is `eta*Gamma*p_e*t`, with
`p_e=Omega²/(Gamma²+2*Omega²+4*Delta²)`. One positive efficiency is therefore
strictly identifiable by bounded weighted least squares. The 24 unique settings
vary drive, signed detuning, decay and duration, each repeated six times. The
known instrument sigma is fixed at .0004 independently of signal and efficiency.
Seeds are 241341 (calibration), 241343 (noise), and 241347 (domain).

In the explicit public `(g,e)` basis, the source constructs the exact generator
and evaluates the time-integrated connected population covariance with a block
matrix exponential. Its seed is `{n,rho_ss}/2-p_e*rho_ss`. The oracle changes
that seed to `sigma_minus*rho_ss*sigma_plus-p_e*rho_ss`. Both have prefactor
`2*(eta*Gamma)²` and the same mean. That replacement is the physical detection
conditioning, rather than a change in Bloch dynamics, parameter fitting, time
normalization, or numerical integration.

The independent verifier directly evolves three 2×2 density-matrix factorial
moments. Its stationary initial coherence is analytic, and it never imports the
oracle's Liouville vectorization or two-integrator covariance block. Separate
prototype checks use the tilted moment-generating matrix and Cauchy derivatives,
full number-resolved density matrices, and the resonant closed-form correlation.
The count cutoff is well beyond the expected photon number; 48→64 states leaves
the reported moments unchanged to roundoff. The radiative hazard is bounded by
Gamma, so discarded high-count tails also have a Poisson upper bound.

The public apparatus specifies the Hamiltonian basis, spontaneous decay, vacuum
Markov/RWA regime, stationary preparation and the actual recorded count. The
drive is absent from the detected port, while all emitter fluorescence,
including its elastic component, is counted. Broadband collection, independent
thinning and the absence of filtering, dead time and dark counts make the
measurement unique without stating a count-correlation formula. The neutral
instruction allows replacement of any implementation. Efficiency starts as
`None`; private data, true parameter, hint and this note are outside the image.

All 256 noise realizations pass calibration/parameter checks for both controls.
The oracle passes every hidden group. The source fails all three variance groups:
nominal relative errors .855, 1.144 and 1.332 against the .04 gate, while the
oracle is below 2.7e-6. Worst noisy oracle error is .000132; minimum shortcut
variance error is .85512. Hidden variances exceed .1768, avoiding near-zero
relative-error targets. Sixty-four corner/random parameter cases agree with the
independent moment ODE to 1.96e-12; contour refinement, stationary density,
thinning, detuning symmetry, dark limit and short-gate Poisson limit pass.
Local controls are 7/7 for the oracle and 4 passes plus three intended variance
failures for the shortcut, in .47 and .45 seconds. Runtime limits remain
600 seconds for the agent and 60 seconds for verification.

No model difficulty has been measured. The familiar antibunching mechanism may
be easy for an agent that attends to the detector. Classification must examine
the complete public trajectory: preserving the population-intensity closure is
the intended physical mistake, whereas correctly choosing photon conditioning
and then misimplementing its regression is an implementation failure.

Archive searches found no resonance-fluorescence count-variance apparatus.
Screened dressed-photodetection concerns transitions in an interacting energy
basis; triad-photon-phase concerns multiphoton interference. Electronic and
thermal counting tasks use different circuit/reservoir constraints. Shared
quantum-noise background is acknowledged without claiming different fundamental
theoretical principles.

Primary support: H. J. Kimble, M. Dagenais and L. Mandel,
[Photon Antibunching in Resonance Fluorescence](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.39.691),
Physical Review Letters 39, 691 (1977). The primary abstract was read; independent
counting derivations above fix this task's normalization and finite-gate readout.
