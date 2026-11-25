# Transport through a floating metallic contact

A phase-coherent spinless electron scatterer connects three ideal single-channel leads, labeled 0, 1 and 2. In units with electron charge magnitude e=1 and Planck constant h=1, its energy-independent scattering matrix throughout the relevant transport window is

S = I - 2 u u^T, where u = (cos(alpha) cos(beta), sin(alpha) cos(beta), sin(beta)).

Incoming and outgoing amplitudes have unit particle-flux normalization. The real symmetric matrix is unitary. The unknown common mixing angle alpha is between 0.25 and 0.7 radians. Each experiment sets beta between 0.15 and 0.8 radians. There are no electron interactions inside the scatterer, no spin degeneracy, and no additional channels or external electromagnetic environment.

Lead 0 ends in an ideal voltage-clamped reservoir at chemical potential `bias` between -1.5 and 1.5. Lead 1 ends in an ideal voltage-clamped reservoir at chemical potential zero. Lead 2 ends in a macroscopic metallic island with finite capacitance C. It has no external particle connection. Fast internal energy relaxation keeps its electrons in a Fermi distribution; an ideal heat-only thermostat maintains temperature zero, including under bias. That thermostat exchanges energy but neither charge nor particles. The island starts in its stationary state. Its mean electrochemical potential is not externally prescribed.

Use the macroscopic semiclassical limit: the island's charging energy and
Coulomb blockade are negligible. Scattering events transfer charge to or from
the island, and its instantaneous electrochemical potential responds through
its capacitance. Internal thermalization is much faster than island charge
relaxation; the latter is much slower than scattering. The instantaneous
occupation remains a Fermi step at the fluctuating island potential.
The voltage-clamped reservoirs have no potential fluctuations. Take the
stationary long counting-time limit at finite C first, with counting times
much longer than C/G22, where G22 is the dc island response. Report the leading
macroscopic-capacitance term as charging energy tends to zero. C does not enter
these limiting coefficients.

Define Q_a(t) as the signed number of electrons transferred from reservoir a
into the scatterer during time t. Positive current is particle flow from a
reservoir into the scatterer. The observable `current` is lim E[Q_a(t)]/t.
The observable `noise` is lim Cov[Q_a(t),Q_b(t)]/t. The observable
`third_cumulant` is lim E[(Q_a(t)-E[Q_a(t)])^3]/t. Means are stationary;
there is no spectral factor of two in these definitions. Odd cumulants retain
the sign of transferred particles. Each input includes `bias`, `beta`,
`observable`, `lead` (0 or 1), and `other` (0 or 1, used for noise only).

Calibration records dc currents and current covariances with independent
Gaussian errors specified by sigma. One alpha applies to every record and
subsequent measurement. Implement Model.fit(records), returning self, and
Model.predict(experiments), returning a finite NumPy array with one value per
input. Store the fitted alpha as Model.mixing_angle. Records in
data/calibration.json have input, value and sigma fields.
