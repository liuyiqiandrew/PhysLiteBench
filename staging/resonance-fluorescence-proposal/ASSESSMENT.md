# Resonance-fluorescence counting proposal

Status: concept reviewed; no prototype, packaged task, or model evaluation. Held
while the same-family FENE revision is prototyped.

A stationary, coherently driven two-level emitter has known rotating-frame
Hamiltonian `(Delta*sigma_z+Omega*sigma_x)/2` and spontaneous decay rate Gamma.
Broadband detection collects fluorescence with the incident drive excluded from
the detected port. The counter has no dark counts, filtering, dead time or
saturation. Each emitted photon is independently retained with unknown efficiency
eta; each finite gate begins in the stationary ensemble. Readout is either mean
or variance of the actual recorded count, across repeated preparations.

The completed source would solve the exact optical Bloch dynamics and stationary
state, fit eta from diverse nonzero mean-count settings, and approximate the
second count cumulant as Poisson shot noise plus the integrated symmetrized
excited-population covariance. This is a semiclassical population-intensity
second-cumulant closure. It is positive and at least Poisson, but should not be
claimed to specify a positive classical intensity process at all higher orders.
The physical detector instead counts emissions that condition the emitter via
the lowering transition. A number-resolved master equation and independent
counting-field generator would verify that distinction. No detector-conditioning
formula or diagnostic hint belongs in public instructions.

Physics reviewer approved the concept and detection-port qualifications on
2026-10-04. Root authorized a prototype, then prioritized FENE r2 before any
fluorescence numerical work. Expected difficulty is uncertain because
antibunching is familiar; no empirical failure claim is made.

Archive search for fluorescence, photocount, photon-count, antibunching, Mandel,
population noise and jump conditioning in task author/assessment files found no
same apparatus. Related screened dressed-photodetection tests detection in an
interacting energy basis, while triad-photon-phase tests multiphoton interference.
Prior electronic counting tasks use different reservoirs/circuit constraints.

Primary experimental support: H. J. Kimble, M. Dagenais and L. Mandel,
[Photon Antibunching in Resonance Fluorescence](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.39.691),
Physical Review Letters 39, 691 (1977). The primary abstract was read on 2026-10-04.
