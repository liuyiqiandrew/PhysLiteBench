# Tracer pulse in a tube

A dilute passive point tracer moves in an effectively infinite straight circular tube. A steady incompressible Newtonian liquid has laminar no-slip flow, with known cross-sectional mean axial velocity `mean_flow` and radius `radius`. Fluid inertia does not alter the fully developed pressure-driven profile. The tracer has constant isotropic molecular diffusivity D, unknown between 0.005 and 0.06 mm²/s, and no inertia, adsorption, reaction, or influence on the liquid. The circular wall reflects the tracer with zero normal solute flux. There are no axial boundaries in the measured region.

Each shot begins with a Gaussian axial pulse centered at z=0, with known standard deviation `initial_width`, independently uniform in cross-sectional area. The flow is already stationary at preparation. Measure after `time` seconds from this preparation. The detector integrates over the entire cross section and reports either `mean_position` (mm) or `axial_variance` (mm²) about that mean. It collects every tracer molecule, without velocity-dependent weighting or spatial censoring.

Inputs include `time` in [0,10] s, `radius` in [0.2,0.6] mm, `mean_flow` in [-1.2,1.2] mm/s, `initial_width` in [0.05,0.2] mm, and `observable`. Calibration uses zero flow and records axial variance. Measurement errors are independent Gaussian with the listed standard deviations. Initial widths and all apparatus settings are exact; one D applies throughout.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Store D in `Model.diffusivity`, in mm²/s. Records in `data/calibration.json` contain `input`, `value`, and `sigma`.
