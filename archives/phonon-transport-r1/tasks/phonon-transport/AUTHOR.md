# Phonon transport, revision 1

The apparatus is a linearized gray Callaway relaxation model. The public
description specifies each collision class by its local maximum-entropy
target and conserved quantities. It does not prescribe the incorrect combined
isotropization operator supplied in the starter.

For an isotropic linear dispersion, a phonon's momentum divided by its energy
is its propagation direction divided by v. Frequency-independent relaxation
rates therefore allow the energy-integrated distribution to close. In linear
response the resistive target is the angular energy average M0. The normal
target is M0+3*mu*M1: it must reproduce both M0 and M1, and the factor three
follows from the sphere average of mu^2 being 1/3. There is no phonon-number
constraint. Thus

    partial_t f + v*mu*partial_x f
      = -nu_R*(f-M0) - nu_N*(f-M0-3*mu*M1).

The angular collision eigenvalues are zero for l=0, -nu_R for l=1, and
-(nu_R+nu_N) for l>=2. The shortcut uses zero for l=0 and -(nu_R+nu_N) for every
other harmonic. It is a stable, energy-conserving isotropic relaxation model
with the complete angular streaming operator. All controls, times, rates and
detector moments are used. Its error is treating momentum-conserving normal
collisions as resistive scattering.

Calibration prepares spatially uniform P2 anisotropy and measures M2. Streaming
vanishes and both models give initial[2]/5 times
exp(-(nu_R+nu_N)*t), exactly for every allowed nu_R. All three known normal
rates (.2,.5,1 s^-1) are nonzero during calibration. Time dependence identifies
the common resistive rate; the true value is .3 s^-1. Calibration contains
144 records with independent noise of .006 times the maximum noiseless signal.
The committed-data seed is 13119 and validation-noise seed is 23119.

Spatial streaming couples P0, P1 and P2 in the hidden experiments. The detector
measures an energy grating, a heat-current sine component, and transfer from
initial anisotropy into an energy grating. The oracle uses 64 orthonormal
Legendre modes, with multiplication by mu represented by its symmetric
three-diagonal matrix. Public coefficients are ordinary Legendre coefficients;
input and output factors sqrt(2*l+1) convert them correctly. Fourier amplitudes
use cosine-i*sine, so sine readouts are minus the imaginary part.

The independent verifier evolves 80 angular ordinates. Its collision matrices
are constructed directly from angular energy and momentum projections, rather
than copying the spectral diagonal. Validation checks 120 ordinates and 96
Legendre modes, the full allowed rate/time/wavenumber corners, homogeneous
mode decay, both collision conservation laws, weighted collision dissipation,
agreement when the normal rate is zero, and the collisionless sinc limit.
All discrepancies are below 5e-15 in the specified coefficient units.

Grading uses group RMS error divided by the group's RMS physical signal, with
a .04 limit. The fitted oracle's largest hidden error is .00254; the completed
shortcut gives 1.344, .660 and .718. All 256 noisy calibration fits pass both
chi-square and parameter checks, with maximum parameter error 1.52%. Hidden
checks at the observed parameter extrema give oracle error <=.00643 and
shortcut error >=.6599. These are parameter-extrema sensitivity checks, not an
exhaustive hidden Monte Carlo claim. Local isolated pytest controls give oracle 7 passed and shortcut 4 passed/3
hidden failures. Agent difficulty is not yet measured.

Run `python scripts/validate_phonon_transport.py` to validate the committed
data. Explicit `--generate` is required to regenerate both public and private
copies. Reports are written to `jobs/phonon-transport-validation/summary.json`.
The completed shortcut is `scripts/phonon_transport_baseline.py`. The hint is
appended only by the runner's instruction-only hint condition. Docker copies
only public environment files; no reference, true parameter or hint is in the
ordinary agent image.

The displaced Bose target and role of momentum conservation are discussed in
[Phonon hydrodynamics in two-dimensional materials](https://www.nature.com/articles/ncomms7400).
The present task is explicitly three-dimensional and gray, with its own
stated ideal relaxation model.
