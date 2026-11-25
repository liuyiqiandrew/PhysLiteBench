# Transport of a weak radiation pattern

Mutually incoherent narrow beams of photons propagate through a uniform stationary absorbing medium. The beams all have the same photon energy and speed c=1 in the stated units. The spatial coordinate x is periodic with period 2*pi; the preparation and all measurements are uniform in y and z. Each beam has a specified unit direction n_b in three dimensions. Between removal events photons follow straight paths with that direction. Absorption removes each photon with the same constant rate kappa per unit time, independently of direction and position. There is no scattering, emission, refraction, interaction between photons, or feedback of the radiation on the medium.

At time zero the contribution of beam b to the radiation energy density is

    I_b(x,0;epsilon) = weight_b * [1 + epsilon*modulation_b*cos(wavenumber*x)].

The positive known weights sum to one, setting the energy-density unit. Modulations are known real numbers in [-1,1]. The controlled epsilon is taken to zero at fixed listed settings. For every |epsilon|<1 this preparation has nonnegative intensity. Each measurement uses a new preparation and an otherwise identical unmodulated reference preparation with epsilon=0. The beam directions and weights remain the labels of the initial preparation.

A local radiometer measures total radiation energy density E at `position` and `time`. Its reported weak-response signal is

    limit as epsilon -> 0 of
    [E(position,time;epsilon) - E(position,time;0)] / epsilon.

This is a local energy-density difference in the stated unit. It is signed and includes the contributions from all beams. No spatial integration or normalization is part of the readout.

The single unknown removal rate kappa lies in [.12,.45] and is common to every experiment. Each input dictionary contains `directions` (two to six unit three-vectors), `weights` and `modulations` (matching lists), `wavenumber` (an integer 0,1,2 or3), `time` in [0,3.2], and `position` in [0,2*pi]. Directions, weights, modulations and all timing and length settings are exact.

`data/calibration.json` contains independent records with `input`, measured `value`, and Gaussian standard deviation `sigma`. Instrument uncertainty is fixed independently of kappa.

Implement `Model.fit(records)`, returning self and storing kappa as `absorption`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Run `python -m pytest -q test_public.py`.
