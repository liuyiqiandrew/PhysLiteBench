# Deformable superconducting ring

A thin circular superconducting wire forms a ring of radius R. A radial support changes R quasistatically while preserving the total wire material volume and its circular cross section. Use reduced units: the length unit, flux quantum and geometric inductance scale are one. The wire radius is a(R)=0.012/sqrt(R), small compared with R throughout the allowed range. Use the defining one-dimensional London model with uniform current density across the section.

Use the ideal quadratic London model at fixed temperature. The condensate carrier density and material volume are fixed during deformation; condensation and ordinary elastic energies do not depend on the current. The known geometric self-inductance is

    Lg(R)=R*(log(8*R/a(R))-2).

This supplied geometric law defines the apparatus to the stated precision. The moving condensate has kinetic inductance Lk(R)=kinetic_scale*R^2; its flow kinetic energy is Lk*I^2/2. The unknown positive kinetic_scale lies in [4,16]. The integer phase winding n is prepared before each measurement and remains fixed during any radial displacement. There are no phase slips or quasiparticle currents. All allowed states remain in the same superconducting branch. The total linked magnetic flux is the applied flux plus Lg*I. The London fluxoid, the sum of this magnetic flux and Lk*I, is n times the flux quantum.

A large external source maintains a spatially uniform perpendicular magnetic field B while the ring deforms; its applied flux is pi*R^2*B. The source currents remain fixed during a force measurement, and their exchanged electrical energy is included in the energy accounting. Magnetic and mechanical transients have settled before readout. No resistance or radiation loss is included in this ideal quasistatic model.

The support measures the net current-dependent generalized radial force exerted by the ring, after subtracting the force in the same geometry with no circulating current. A positive force does positive work when R increases: mechanical work delivered by the ring is F*dR. The subtraction removes ordinary material elastic and condensation contributions. The current readout is the signed circulating current, positive when its self-flux is positive.

Each experiment contains `radius` in [0.7,1.6], `magnetic_field` in [-0.35,0.35], integer `winding` in [-3,3], and `observable`, either `current` or `radial_force`. Inputs are exact. The one common kinetic_scale applies to every preparation.

`data/calibration.json` contains current measurements at varied radius, field and winding. Each record has `input`, measured `value`, and known independent Gaussian standard deviation `sigma`. Sigma is a fixed instrument resolution, independent of the parameter and noiseless response.

Implement `Model.fit(records)`, returning self and storing `kinetic_scale`. `predict(experiments)` returns a finite NumPy array of shape (len(experiments),) in input order. Any implementation satisfying the interface may be used.
