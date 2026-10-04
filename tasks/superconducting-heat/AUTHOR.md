# Superconducting heat, revision 1

The physical distinction is the energy-dependent electron–hole composition of a Bogoliubov quasiparticle. The completed shortcut includes the superconducting density of states, the full Fermi distributions, unequal temperature-dependent gaps, phase interference, and correct conductance normalization. It approximates the interference ratio by its value at the larger gap edge. This is a low-temperature spectral approximation, applied outside its regime; it is not a missing phase input or a broken numerical integral.

For positive quasiparticle energy E, define N_j=E/sqrt(E²−Delta_j²) and M_j=Delta_j/sqrt(E²−Delta_j²), above both gap edges. The leading tunneling heat power is

    Q = 2 G/e² integral E [N_L N_R − M_L M_R cos(phi)] [f_L−f_R] dE.

The source replaces M_L M_R/(N_L N_R)=Delta_L Delta_R/E² by Delta_L Delta_R/Delta_max². It retains the exact phase-dependent cosine. Both controls agree exactly at calibration phases ±pi/2. The signal is linear in the sole unknown normal conductance, so weighted calibration identifies it uniquely.

The public Hamiltonians specify isotropic s-wave reservoirs and electron tunneling with a flat normal density of states and energy-independent real matrix element. The given gap law is the defining material law, not a claim that the tanh interpolation is the exact BCS self-consistency solution. All temperatures remain below both critical temperatures. Across the complete allowed domain, Delta_R−Delta_L is at least 0.9591 k_B K. The unequal edges have an integrable square-root singularity, avoiding the coincident-gap singularity of leading-order tunneling theory. Predictions are explicitly leading order in the individual channel transparencies.

The phase bias is static and voltage is zero, so it supplies no electrical power. The stationary Cooper-pair charge current carries no heat at this order; the dissipative interference heat term still remains. The measured energy is relative to the common chemical potential, and only the contact's electronic heat is included. These assumptions exclude finite-voltage phase dynamics, environmental heat channels, charging effects and lifetime broadening.

The oracle integrates the reduced spectral factor with a 160-node Gauss–Legendre rule after E=sqrt(Delta_max²+s²). The independent verifier constructs the four electron-like/hole-like branch amplitudes u_L u_R−exp(i phi)v_L v_R, using u_j²=(1+xi_j/E)/2 and v_j²=(1−xi_j/E)/2. It sums their squared moduli and integrates with adaptive quadrature in E=Delta_max+z², including both branch density-of-states Jacobians. The branch normalization is checked against the normal-state Fermi heat integral and normal conductance.

Both models are passive: each spectral factor is positive and heat flows from hotter to colder. They preserve phase reversal symmetry and zero heat at equal temperatures. The source converges to the correct edge-coherence approximation at low temperature. The independent amplitude sum agrees with the reduced oracle to 3.37e-14 pW on hidden cases; quadrature refinement is below 8.01e-14 pW, and domain-corner error below 1.21e-14 pW.

Calibration contains 144 measurements with fixed independent uncertainty 0.0005 pW. Fixed uncertainty carries no noiseless-response information. The private conductance is 45 microsiemens; calibration seed is 33021 and repeated-noise seed is 43021. Both controls fit 44.98353 microsiemens, with reduced chi-squared 1.09901. All 256 noisy fits pass; the largest relative conductance error is 0.0899%.

Hidden errors are group RMSE divided by the group's true RMS power, with a 0.04 limit. This evaluates positive and negative powers together without dividing by individual near-zero readings. Nominal oracle error is 0.000366; shortcut errors are 0.271, 0.102 and 0.189. Across fitted-parameter extrema, oracle error stays below 0.000899 and shortcut error above 0.1014. Local isolated verification gives oracle 7/7 in 0.54 seconds, and shortcut four passing calibration/interface/parameter tests plus three intended prediction failures in 0.46 seconds. Scientific validation does not establish the agent failure rate; model evaluation is pending.

See [the scientific report](../../results/superconducting-heat-validation.json). This distinct apparatus replaces the weak graded thermoelectric rod, whose complete revision and reviewed 3/3 outcomes are in [the archive](../../archives/thermoelectric-rod-neutral-r2/README.md). The task uses the exact approved neutral instruction.

Primary sources: [Guttman et al., Physical Review B 55, 3849 (1997)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.55.3849) derives the quasiparticle, interference and pair contributions and separates their finite-voltage behavior. [Giazotto and Martínez-Pérez, The Josephson heat interferometer (2012)](https://arxiv.org/abs/1205.3353) gives the stationary spectral heat formula and experimentally tests phase-dependent heat transport. The independent branch-amplitude calculation here follows directly from the supplied Hamiltonian.

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_superconducting_heat.py
```

Ordinary validation reads checked-in data; `--generate` intentionally rewrites public and private calibration copies together.

Independent final physics review is recorded under `superconducting-heat-r1` in [the peer report](../../results/constraint-hardening-physics-review.json). Its off-grid branch-reference check agrees within 4.61e-14 pW.
