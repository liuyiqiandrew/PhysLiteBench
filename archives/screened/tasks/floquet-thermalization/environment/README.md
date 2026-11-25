# Population and bath calorimetry of a modulated two-level system

Use hbar=kB=1. In the fixed basis |e>,|g>, the system Hamiltonian is
H(t)=[1+A*cos(Omega*t)]*sigma_z/2, with sigma_z|e>=|e> and
sigma_z|g>=-|g>. The prescribed classical drive starts at phase zero and can
supply or receive work. The label |e> always denotes this fixed basis state,
including when the instantaneous level splitting changes sign.

Two independent stationary bosonic reservoirs couple through sigma_x tensor
B_b. Their unsymmetrized correlation spectra are

    G_b(omega) = integral dt exp(i*omega*t) <B_b(t)*B_b(0)>
      = coupling * strength_b * omega * exp(-abs(omega)/cutoff_b)
        / (1-exp(-omega/temperature_b)).

Use the continuous value coupling*strength_b*temperature_b at omega=0.
Positive omega is energy deposited in a reservoir by the undriven system.
The reservoir parameters, fixed in every experiment, are

    bath 0: temperature=.2, cutoff=1.5, strength=1
    bath 1: temperature=.65, cutoff=.9, strength=.7.

Cross-correlations between the reservoirs vanish. Each remains in its own
thermal state. The common unknown coupling is between .002 and .008. There
are no other relaxation or dephasing channels and no backaction on the drive.

Use the leading weak-reservoir-coupling dynamics: take coupling to zero at
fixed nonzero Omega, fixed A and fixed spectrum shapes, observing
relaxation-scale times. All dissipative rates are much smaller than Omega.
Do not impose an adiabatic-drive approximation. Reservoir energy shifts affect
only phases. Corrections beyond this controlled limit, including preparation
transients on the reservoir correlation time, are outside the requested model.

Prepare a diagonal system density matrix with the given initial_population,
independently of the equilibrium reservoirs, at t=0. Every input specifies
amplitude in [0,2.5], frequency in [.6,2], initial_population in [0,1], and
cycles, an integer from 1 to 200. Elapsed time is t=2*pi*cycles/frequency.
The observable field selects one measurement:

- population: the probability of |e> at the end of this interval.
- heat: the mean energy gained by the selected reservoir over the entire
  interval, divided by elapsed time. A calorimeter measures that reservoir's
  energy change, with positive heat meaning energy delivered to the reservoir.

Every input includes bath (0 or 1); it selects the calorimeter for heat and is
unused for population. Heat has units of energy/time. It is an interval average,
not the instantaneous heat current at the final time. There are no couplings
between the calorimeters and the prescribed drive, other than energy exchange
through the system. The interaction energy makes no leading contribution in
the stated weak-coupling limit.

Calibration uses amplitude zero and measures both population and each bath's
heat over varied intervals and initial preparations. Errors are independent
Gaussian with the listed sigma. All controls and bath parameters are exact.
Fit one common coupling across all records. Implement Model.fit(records),
returning self, and Model.predict(experiments), returning a finite NumPy array
of shape (len(experiments),). Store the estimate as Model.coupling. Records in
data/calibration.json contain input, value and sigma. Run python -m pytest -q test_public.py.
