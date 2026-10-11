# Superconducting heat R10

R9 passed3/3. This revision keeps its exact ordered quantum coupling and mean
spin kinetics and adds a low-frequency gate-work noise measurement. The public
apparatus specifies conditional tunneling/flip events and extensive size
scaling. Stationary spin potentials are internal stochastic states, not clamps.
The complete shortcut freezes them at their deterministic stationary values.

For an event with gate work w (in kelvin) and spin injection dS, let h_mu be
the derivative of mean work rate and J the stationary spin-drift derivative.
The low-frequency effective mark is w-h_mu J^-1 dS. Independent equal-energy
spin flips have no direct gate work but contribute through the same response.
Sum both tunneling directions and spin-flip shot noise. Spin susceptibility
cancels at zero frequency, but cannot justify freezing the populations.
Every tunnel event satisfies w=(drive_energy/2)*(dS_L+dS_R).

The independent reference assembles explicit Fock/Bogoliubov spin-channel
rates, obtains the work derivative by finite differences and flip intensities
by energy quadrature, then solves the4x4 linearized stationary response of the
joint spin/work jump-process Hamiltonian. No effective-mark function is shared.
Validate positivity, quantum spectral refinement, the work/spin identity,
the no-spin-loss conservation limit, fast-relaxation Poisson limit, static
barrier zero noise, mean-readout inheritance, and256 noisy calibration fits.
Exactly3 fresh unhinted GPT-5.6 Luna High trials follow scientific and Docker
controls. All prior scores and implementation-error diagnoses are preserved.
