# Thermal transition basis, revision 1

The supplied local Lindblad model uses bare lowering/raising operators and the thermal spectrum at the two bare gaps. It is a valid completely positive, trace-preserving generator and is exactly the specified bath model at zero spin coupling. Calibration identifies the common weak rate through transient populations. Under finite static interaction, the energy-resolved equilibrium reservoirs act between full-Hamiltonian eigenstates, not bare spin levels.

The oracle resolves each local system-bath operator into full-Hamiltonian Bohr-frequency components and groups equal-frequency transitions separately for each independent reservoir. The independent reference diagonalizes the two parity blocks and propagates a four-state classical population equation. Energy-eigenstate preparations ensure exact diagonality of the correct dynamics even though Bohr frequencies repeat. Stationary predictions are independently the full Gibbs state. All four energy eigenstates are bath connected; no inaccessible-sector issue is involved.

The shortcut neglects the interaction in bath transition energies and eigenoperators while retaining it in unitary evolution. This is a physical approximation, not an invalid Lindblad formula. The hidden equilibrium and transient outputs test its violation of full-Hamiltonian detailed balance. The unknown rate affects transient predictions but not the correct stationary state.

The weak rate range is small relative to distinct Bohr-frequency separations, and the known unequal bare gaps prevent calibration degeneracies. Measurement sigma is 0.0015 in probability/correlation units; hidden relative RMS limit is 2.5%. Run scripts/validate_thermal_transition_basis.py; regenerate only with --generate.

The local/global distinction for coupled spins with independent baths is discussed in [Cattaneo et al., 2019](https://arxiv.org/abs/1906.08893). This task prescribes the resolved weak-coupling regime; it does not claim that a global secular equation is valid in every coupling regime.
