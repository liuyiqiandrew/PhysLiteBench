# Stationary resonance-fluorescence counts

Status: scientifically checked prototype; no packaged task or model evaluation.
The earlier proposal is preserved in staging/resonance-fluorescence-proposal.

Apparatus: a stationary driven two-level emitter has a known rotating-frame
Hamiltonian, known radiative decay and no other decay or dephasing. A broadband
counter collects fluorescence only, with the incident drive absent from its
port. Each photon is retained independently with one unknown efficiency. There
are no dark counts, dead time, saturation or spectral filtering. Repeated finite
gates sample independently prepared stationary ensembles. The readout is the
mean or variance of the actual recorded photon number. This operational
specification selects photon counting without supplying a correlation formula.

The completed source retains the exact optical Bloch generator, stationary
state and symmetrized connected population covariance. It treats population
fluctuations as nondestructive emission-intensity fluctuations and adds Poisson
shot noise to their integrated second cumulant. This is a semiclassical
second-cumulant closure, not a claim of a positive classical intensity process
at every order. Its excess variance is nonnegative. Rich mean-count calibration
identifies detector efficiency and matches exactly. The physical photon record
instead has emission-conditioned correlations.

The prototype checks 54 combinations of Rabi rate .6–1.8, detuning -.5–.5,
decay .8–1.2 and gate1–6, at efficiency .73. Count variances are .0818–1.475;
source relative errors are30.8–277.4%, with exact mean agreement2.3e-16.
Direct matrix-valued factorial-moment ODEs agree with independent counting-field
Cauchy derivatives to1.13e-12. Three full count-resolved distributions agree
within1.96e-12 and normalize to roundoff. A resonant closed-form correlation
check agrees to4.88e-13. Runtime is .44 seconds with one BLAS thread.

The small model and independent controls make numerical difficulty unlikely.
The familiar antibunching mechanism may be easy for a physics-aware agent;
no difficulty claim is made before a frozen empirical batch. An agent that
chooses counting backaction but miscodes its signs would be an implementation
failure, rather than the intended semiclassical observable mistake.

Archive audit found no resonance-fluorescence count-variance apparatus. Screened
dressed-photodetection concerns detection in an interacting energy basis;
triad-photon-phase concerns multiphoton interference. Electronic counting and
thermal-reservoir variance candidates have different constraints/readouts.
Related quantum noise background is acknowledged.

Concept and detection-port qualifications were independently reviewed by the
Physics teammate. The actual prototype source and all existing reports passed independent
Physics review without numerical reruns. A count-label correction derives54
settings from the actual product; no equations or numerical results changed. Primary support:
[Kimble, Dagenais and Mandel, PRL39,691 (1977)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.39.691).
The primary paper establishes observed resonance-fluorescence antibunching; the
prototype's direct derivations fix its counting normalization independently.
