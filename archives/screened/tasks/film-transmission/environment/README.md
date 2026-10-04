# Dielectric film spectra

A uniform planar dielectric film separates two semi-infinite optical media. All three media are linear, isotropic, lossless, nonmagnetic and nondispersive over the measured band. Their refractive indices are 1 for the entrance, 2.1 for the film, and the specified `exit_index` for the exit medium. Interfaces are flat and ideal. The common film `thickness` is unknown between 0.05 and 0.12 micrometers.

Illuminate from the entrance medium with a coherent monochromatic TE plane wave: the electric field is parallel to the interfaces and perpendicular to the plane of incidence. `wavelength` is the vacuum wavelength in micrometers, and `angle` is the incidence angle in the entrance medium, in radians from the interface normal. The incident electric-field amplitude is 1. Treat interference coherently throughout the film. There is no absorption, roughness, incoherent averaging, or finite detector aperture.

The observable is `reflectance` or `transmittance`, the reflected or transmitted time-averaged optical power crossing planes parallel to the film, divided by the incident power crossing such a plane. Detectors cover the entire illuminated footprint, in their respective homogeneous half-spaces. All allowed waves propagate normally away from the interface as well as along it; no total internal reflection occurs for these inputs.

Every experiment includes `wavelength` in [0.5,1.1], `angle` in [0,1], `exit_index` in [1,1.9], and `observable`. Calibration uses exit index 1 and records transmittance while varying wavelength and angle. Measurement errors are independent Gaussian with the listed standard deviations. One thickness applies to all settings.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Store thickness in `Model.thickness`, in micrometers. Records in `data/calibration.json` contain `input`, `value`, and `sigma`.
