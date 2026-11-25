# Bosonic lattice thermal Hall transport, revision 1

This file, the solution and the hidden reference are excluded from the agent image.

The apparatus fixes a positive two-band, number-conserving bosonic Hamiltonian, the orbital embedding, zero chemical potential, transport convention and order of limits. Frequency spectroscopy identifies the single exchange scale. The task instruction permits replacing the entire implementation; it names no correction or numerical method.

The starter evaluates the complete Bloch eigensystem, velocity matrices and interband energy-current vertices. Its bulk current response is mathematically consistent but does not give the calorimeter's transport coefficient. The distinction is the energy-magnetization contribution. In the convention A_n=i<u_n|grad u_n>, the transport response is

    kappa_xy = -T * integral_BZ sum_n Omega_n(k) c2(n_B(E_n/T)) d²k/(2*pi)².

For x=E/T, c2=x²/(exp(x)-1)-2*x*log(1-exp(-x))+2*Li2(exp(-x)). The source's response tensor is -Omega. The bulk expression instead weights it by Bose population and the squared mean of the two interband energies, divided by T.

The primary derivations are Eqs. 15, 23 and 25 of [Matsumoto and Murakami, Rotational motion of magnon and thermal Hall effect](https://arxiv.org/html/1106.1987). They distinguish transport from the uncorrected energy-current response.

The independent reference obtains Berry flux from gauge-invariant eigenvector plaquettes and computes minus the temperature derivative of boundary energy flow. It uses fourth-order finite differences and spatial Richardson extrapolation of 192/384 grids. It does not use the source's velocity-matrix formula or the solution's c2 kernel. A third curvature check uses the two-band d-vector identity. Both handedness signs, quantized Chern numbers, positive bands, parameter endpoints and low/high temperature corners are checked.

The fixed calibration contains 288 independent spectroscopic measurements with sigma=0.002. Its seed is 14728, noise-validation seed 92159, and author-only exchange scale is 1.04. Hidden relative RMS tolerance is 0.04. No uncertainty depends on the unknown scale or response. Public and private calibration copies are byte-identical.

Scientific validation: 256 independent noise realizations pass calibration, parameter recovery and the oracle; none pass the shortcut. At the frozen calibration fit the shortcut hidden errors are 0.661, 1.415 and 0.526; oracle errors are below 0.000159. The independent hidden reference and oracle agree within 8.1e-10 of the maximum response, and domain-corner relative error is below 1.5e-8. These are validity checks, not evidence of model difficulty.

Artifacts: `results/boson-hall-validation.json`, `results/boson-hall-local-controls.json`, `results/boson-hall-source-provenance.json`, and the independent peer review. No agent outcome is claimed before a fresh frozen run.
