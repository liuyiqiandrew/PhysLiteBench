# Membrane partition, revision 2

The apparatus, calibration data, unknown parameter, independent physical reference, oracle, and grading are unchanged from revision1. Revision1 was too easy: all three unhinted agents identified the independent binary-potential approximation and implemented the shared-potential model. Its complete source, controls, validation, and reviewed trials are preserved in archives/membrane-partition-r1; retained jobs remain untouched.

Revision2 supplies a different completed approximation. It combines cations into one ideal pool with the prepared population mean valence (N_A+2N_B)/(N_A+N_B), computes a common potential for that pool and the actual chloride pool, then resolves A and B in their prepared proportions. It solves this reduced nonlinear equilibrium exactly, preserves every ion amount and both bulk electroneutralities, and agrees exactly with all pure-salt calibration experiments. Unlike revision1, it uses only one potential and one chloride pool.

The physical error is averaging charge before applying the exponential electrochemical partition law. The correct state-resolved Boltzmann weights are P*exp(psi), P*exp(2psi), and P*exp(-psi); no single cation charge reproduces both positive species in mixed preparations. The common standard transfer factor P is identifiable from pure-salt concentrations and cannot be absorbed into the voltage because it multiplies every valence equally.

The oracle analytically eliminates finite amount balances and solves one monotone neutrality equation. The independent reference instead solves simultaneous log-concentration chemical-potential equations, species balances, and neutrality. The prediction threshold remains1.5% relative RMS; measurement sigma remains0.002mmol/L. Run scripts/validate_membrane_partition.py without regeneration to verify the preserved data and256 noise realizations.

The equilibrium framework is the ideal-ion bulk limit of [Donnan thermodynamics](https://www.uu.nl/sites/default/files/the_donnan_equilibrium_i.pdf). The task explicitly retains finite amount balances and excludes solvent exchange.
