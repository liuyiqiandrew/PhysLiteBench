# Force between normal-metal plates

Two identical nonmagnetic normal-metal half-spaces have flat parallel surfaces separated by a vacuum gap of width `separation` a. Their lateral dimensions and thicknesses are infinite. Both bodies and the electromagnetic field are in global thermal equilibrium at the specified temperature T. Use reduced units with hbar=c=k_B=1. There are no external fields, applied voltages, surface charges, roughness, magnetic order or superconductivity. Mechanical and electrostatic background forces are subtracted.

The local isotropic relative permittivity is the causal Drude response

    epsilon(omega) = 1 - omega_p²/[omega*(omega + i*relaxation)]

with the exp(-i*omega*t) convention. Relative magnetic permeability is one. The strictly positive known relaxation rate is residual material damping and remains at its specified value when temperature is changed, including at T=0. The only unknown is the plasma frequency omega_p, common to all measurements and in [4,9]. The stated local linear response and equilibrium quantum fluctuation-dissipation relation define the ideal material at every frequency and wavevector relevant here. Spatial dispersion, extra surface layers, lattice vibrations and temperature changes of the material parameters are excluded.

The readout is the normal electromagnetic interaction pressure, relative to infinite plate separation, in the equilibrium state of the complete field and dissipative bodies. Negative pressure denotes attraction. Every preparation reaches equilibrium before measurement; T=0 denotes its ground-state limit. This is force per area, in the reduced pressure unit, with no normalization by an ideal-reflector force.

Each input contains `separation` in [0.5,2], `relaxation` in [0.15,0.8], and `temperature`, either exactly zero or in [0.05,0.7]. Settings are exact. Calibration uses T=0 and several separations and relaxation rates. Records in `data/calibration.json` contain `input`, measured `value`, and independent Gaussian uncertainty `sigma`. The instrument uncertainty is 0.000002/separation^4, determined entirely by the known separation.

Implement `Model.fit(records)`, returning self and setting `plasma_frequency`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Use the same fitted plasma frequency for every input.
