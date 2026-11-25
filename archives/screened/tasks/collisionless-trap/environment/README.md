# Slowly deformed trap

A dilute classical gas consists of noninteracting particles moving in a plane.
Use reduced units with particle mass and Boltzmann's constant equal to one.
Each particle has Hamiltonian

    H = (px^2+py^2)/2 + x.T @ K @ x/2,
    K = R(theta) @ diag(omega1^2,omega2^2) @ R(theta).T,

where R(theta) is the usual counterclockwise planar rotation. The trap center
remains fixed. Initially theta=0, the two frequencies are initial_frequencies,
and the gas is in canonical equilibrium with a bath at unknown temperature
T in [0.4,1.6]. The bath is then detached. There are no interparticle collisions,
random forces, damping, or other mechanisms for energy exchange between particles.

Change the trap quasistatically. For a ramp duration tau and u=t/tau between
zero and one, set s=3*u^2-2*u^3. Each normal frequency interpolates linearly in
s from its initial to final value, and theta=s*rotation. Frequencies remain
positive and nondegenerate throughout, with omega1<omega2. Use the strict
adiabatic limit: tau is arbitrarily long compared with both oscillation periods
and inverse frequency gaps. Every run starts with a fresh sample from the same
initial bath temperature.

After the ramp, hold the final trap fixed. A detector reports the ensemble
position variance along (cos(view_angle),sin(view_angle)), averaged over many
final trap oscillations. The gas mean position is zero. All returned variances
are in the squared length unit associated with the reduced units.

## Interface

Implement Model.fit(records), returning self and setting temperature, and
Model.predict(experiments), returning a finite NumPy array of shape
(len(experiments),). You may edit the whole implementation.

Each experiment provides initial_frequencies and final_frequencies, two-element
lists in increasing order; rotation and view_angle in radians. Frequencies
are between 0.5 and 4, their separation is at least 0.4, and both angles lie in
[-1.5,1.5]. The same fitted initial temperature applies to all preparations.

data/calibration.json contains records with input, measured value, and sigma,
the independent Gaussian measurement standard deviation. Settings are exact.
Run python -m pytest -q test_public.py.
