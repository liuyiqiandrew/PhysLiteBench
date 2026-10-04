# Force on a dipolar optical particle

An ideal subwavelength particle in vacuum has co-located isotropic electric and magnetic dipole responses. Use rationalized units with vacuum permittivity, permeability and light speed all equal to 1. The field convention is Re[E exp(-i omega t)], and omega=k. A monochromatic plane wave propagates along +z with electric phasor E=xhat exp(i k z) and magnetic phasor H=yhat exp(i k z); their amplitudes are 1. The particle is held fixed and the measured observable is its time-averaged mechanical force along +z, in these units.

Treat the dipole approximation as exact. There are no higher multipoles, nearby surfaces, other particles, absorption, magnetoelectric constitutive coupling, or motion. The electric and magnetic dipoles respond coherently to the same incident wave. The induced moments are p=alpha_e E and m=alpha_m H. Their lossless dynamic polarizabilities, including radiation reaction, are

alpha_e = a_e / (1 - i k^3 a_e/(6 pi)),
alpha_m = a_m / (1 - i k^3 a_m/(6 pi)).

Here a_e = polarizability_scale * electric_weight and a_m = polarizability_scale * magnetic_weight. The unknown shared polarizability_scale is in [1,3]. Each experiment specifies k in [0.5,1.6] and electric_weight, magnetic_weight in [0,1.5]. The weights are known dimensionless control settings of the ideal dipole responses; the common scale has units of length cubed. These equations and the dipole approximation define the apparatus over the entire allowed band. There is no separate unknown particle radius or material dispersion.

Calibration uses both purely electric and purely magnetic preparations, with varying wavelength and response weight. Every record gives the measured force, independent Gaussian standard deviation sigma, and input object. Subsequent measurements can use any allowed pair of response weights. Fit one common scale across the full calibration.

Implement Model.fit(records), returning self, and Model.predict(experiments), returning a finite NumPy array of shape (len(experiments),). Store the fitted value in Model.polarizability_scale. Read records from data/calibration.json; each has input, value and sigma fields.
