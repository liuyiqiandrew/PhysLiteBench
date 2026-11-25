# Pulse and block measurements

A homogeneous isotropic block of known total inertial mass M and thickness `length = L` is freely suspended in vacuum and initially at rest. Its exact linear optical response in its rest frame is

    epsilon_r(omega) = mu_r(omega) = 1 + f / [1 - (omega + i0)^2].

This specifies the retarded lossless material response. The common oscillator strength f is unknown in [0.2, 0.4]. The known material resonance sets the frequency unit Omega0; units have c=Omega0=epsilon0=mu0=1. Length is in c/Omega0 and time in 1/Omega0. The plane faces have ordinary Maxwell continuity conditions and no coatings.

The block includes all electric, magnetic and mechanical constituents in M. It is initially unexcited, is mechanically stable, and has translationally invariant internal forces. No external mechanical force or reservoir acts during the measurement. The measured position is the center of mass of the complete body. Transverse dimensions and illumination are in the plane-wave limit, with edge effects negligible.

A linearly polarized pulse enters normally from the left, with no incoming field from the right. Its asymptotic incident positive-frequency electric-field amplitude is a real constant times

    [1 - ((omega-center)/width)^2]^4

for |omega-center| < width, and zero outside this interval. The real field includes the conjugate negative-frequency amplitudes. `center` is in [1.5, 1.7], `width` is in [0.04, 0.10], and L is in [0.5, 2.0]. The pulse energy E is its incident time-integrated vacuum energy flux through the illuminated area.

Two measurements are reported:

- `mode = "transit"`: the transmitted temporal energy-flux centroid at the unperturbed right face minus the incident temporal energy-flux centroid at the unperturbed left face, in time units.
- `mode = "displacement"`: M*c^2 times the final center-of-mass displacement divided by E, in length units. Positive displacement is in the incident propagation direction.

These are limiting weak-pulse measurements. At fixed pulse shape, f, L and M, first take E/(M*c^2) to zero. For displacement, extract the coefficient linear in E before the late-time limit after the pulse has left. The optical response at this order is evaluated for the resting undeformed block. Preparation and readout use asymptotic incoming and outgoing pulses.

## Data and interface

`data/calibration.json` contains independent measurements with keys `input`, `value`, and `sigma`. Each input has `mode`, `center`, `width`, and `length`. Every listed sigma is the fixed instrument standard deviation 0.001 in time units. Repeated settings are independent measurements.

Implement `Model.fit(records)` to estimate the common f, save it as `self.oscillator_strength`, and return `self`. `Model.predict(experiments)` must return a finite NumPy array of shape `(len(experiments),)` in the listed order and units. Prediction inputs use the same schema and allowed ranges.
