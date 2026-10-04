# Pressure flow through a charged slit

An ideal symmetric monovalent electrolyte fills a long straight planar slit
between two identical stationary walls at y=-h and y=h. The half-gap h is
`half_gap`, between 0.2 and 0.45 micrometres. The liquid has uniform permittivity
epsilon=7e-10 F/m, density 1000 kg/m³, and unknown dynamic viscosity eta in
[0.0008,0.002] Pa s. Temperature is fixed at 300 K. Both ion species have the
same known diffusivity D=1e-10 m²/s, independent of viscosity in this material.
Use R=8.314462618 J/(mol K) and F=96485.33212 C/mol.

The walls carry equal known immobile charge `surface_charge` between -0.00035
and 0.00035 C/m². They are impermeable to ions and impose no slip on the liquid.
There is no separate surface current, charge regulation, compact layer, or slip.
The ions are ideal point solutes with charges +F and -F per mole. Reservoirs
maintain the chemical potentials of an uncharged bulk electrolyte whose
concentration of each species is `concentration`, between 0.006 and 0.03 mol/m³.
The transverse ion distribution is in electrochemical equilibrium. Use Poisson
electrostatics with the stated wall charge; do not assume thin diffuse layers
or small transverse potential. Reservoir potential defines the transverse
Boltzmann reference. There is no dielectric discontinuity to model outside
the fixed-charge boundary condition.

A steady small pressure gradient drives flow along z: pressure decreases by
`pressure_gradient` Pa/m along +z. Its signed magnitude is at most 1e5 Pa/m.
The reservoirs are electrically open, with no external path for charge transfer;
a stationary axial electric field has established itself. The cross-section
integral of total axial electric current is zero. Ion current includes both
migration and convection with the fluid. No axial concentration gradients or
end polarization are present in the measured fully developed region. Known
reservoir concentrations and wall charge are unchanged by the small axial
forcing. The transverse equilibrium is maintained locally.

Use the leading linear electrokinetic response to axial pressure and electric
forcing. The liquid obeys incompressible Newtonian creeping flow; retain the
force on its diffuse ionic charge. Neglect fluid inertia, axial diffusion of
velocity, Joule heating, and changes of viscosity or permittivity. The forcing
is small enough that transverse ion relaxation is fast and ionic advection
does not distort transverse equilibrium. Each reading is at steady state.

Implement Model.fit(records), returning self and setting viscosity in Pa s,
and Model.predict(experiments), returning a finite NumPy array of shape
(len(experiments),) in m/s. You may change the entire implementation. Inputs
provide half_gap in micrometres, concentration in mol/m³, surface_charge in
C/m², pressure_gradient in Pa/m, and observable. For observable="mean", return
the cross-sectional mean axial velocity. For observable="point", y is a
fraction of h between 0 and 0.85 and the requested position is y*h. A y field
may be present for a mean reading and is ignored. The same viscosity applies
to every preparation.

Calibration in data/calibration.json contains only neutral-wall measurements.
Each record has input, measured scalar value, and independent Gaussian standard
deviation sigma. Experimental settings are exact. Run python -m pytest -q
test_public.py to check the interface and calibration.
