# Dissociation records

A rigid vessel of total volume 1.0 L is divided into two chambers by a free,
massless, frictionless piston. The piston is impermeable to all molecules.
Pressure equilibration is much faster than reaction. Each chamber is maintained
at the same fixed temperature by an external bath throughout the experiment.
The gas mixtures are ideal. There are no external forces on the piston, dead
volumes, heat sources other than the bath, or leaks.

In each chamber the reversible elementary reaction A <=> 2 B occurs. In mmol/L,
its equilibrium constant c_B^2/c_A is Kc=0.75 mmol/L. The forward reaction event
rate per unit chamber volume is k*c_A, and the reverse event rate is
(k/Kc)*c_B^2. The common k is unknown, between 0.03 and 0.3 per second. The
chambers are well mixed. Concentrations always refer to the instantaneous
chamber volume. Each preparation starts in mechanical equilibrium; the stated
initial concentrations have equal c_A+c_B in the two chambers.

## Interface

Implement Model in model.py. fit(records) returns self and sets rate to k in
per second. predict(experiments) returns a finite NumPy array of shape
(len(experiments),), reporting the gas concentration of B in
mmol/L. You may edit the entire implementation.

An experiment provides initial, a 2-by-2 array of initial concentrations
(chamber 0 then 1, species A then B); left_fraction, the initial fraction of
the vessel volume occupied by chamber 0; time in seconds from preparation;
and chamber, 0 or 1, whose B concentration is measured. Initial concentrations
are between 0.02 and 2 mmol/L, left_fraction is between 0.2 and 0.8, and time
is between 0 and 80 seconds. Use the same fitted k for all preparations.

data/calibration.json contains input, measured value, and known independent
Gaussian standard deviation sigma. Experimental settings are exact.
Run python -m pytest -q test_public.py.
