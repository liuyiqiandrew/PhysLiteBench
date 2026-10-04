# Ion motion in a cell connected through a capacitor

Two large parallel metal plates confine a dilute binary electrolyte in a planar
slab 0<x<L, L=1 micrometer. The plates are blocking: neither ion can cross them
or undergo an electrode reaction. Both monovalent ion species have the same
unknown diffusivity D, between 0.6e-9 and 1.6e-9 m²/s. The positive and negative
ions have charges +e and -e. Temperature is fixed at 300 K, the permittivity is
7e-10 F/m, and each species has mean concentration c0=0.015 mol/m³. Use
R=8.314462618 J/(mol K) and F=96485.33212 C/mol.

Use one-dimensional dilute-solution electrochemical transport and continuum
electrostatics. Ionic mobility obeys the Einstein relation. Neglect fluid flow,
steric interactions, ion pairing, image forces, compact surface layers, and
changes of material properties. Local electric-charge density comes only from
the two mobile ion species. Metal surfaces are equipotentials; the right plate
is grounded. The surrounding circuit has no stray capacitance or resistance.

The left plate is connected to an ideal voltage source through an ideal
capacitor. Its known capacitance per electrode area is series_capacitance.
Initially both the source voltage and the series capacitor's charge are zero.
At time zero the source jumps to the specified voltage and is then held there.
The electrolyte's total ionic charge is zero. If Q is the positive charge per
area on the left electrode, the capacitor carries the same series charge and
its voltage drop is Q/series_capacitance. Electrical adjustment of the metal
and source is instantaneous compared with ionic motion. The plates exchange
no ions; there is no ion reservoir.
An observation at time zero is immediately after the source step and electrical
adjustment, before the ions move.

Initially both concentrations are
c_plus(x)=c_minus(x)=c0*[1+first*cos(pi*x/L)+second*cos(2*pi*x/L)].
The two amplitudes obey abs(first)+abs(second)<=0.8. Each experiment starts a
fresh preparation. The material diffusivity is the same in all experiments.

## Interface

Implement `Model.fit(records)`, returning `self` and setting `diffusivity`, and
`Model.predict(experiments)`, returning a finite NumPy array of shape
`(len(experiments),)`. You may edit the whole implementation within this API.

Inputs contain `first`, `second`, `voltage` (-0.1 to 0.1 V),
`series_capacitance` (0.001 to 0.01 F/m²), `time` (0 to 0.01 s), and
`observable`:

- `positive_first`, `negative_first`: (1/(c0*L)) integral c_species(x)*cos(pi*x/L) dx.
- `positive_second`, `negative_second`: the corresponding cos(2*pi*x/L) moment.
- `electrode_charge`: Q/(F*c0*L), positive for a positively charged left plate.

All returned quantities are dimensionless. Calibration records contain `input`,
measured scalar `value`, and independent Gaussian standard deviation `sigma`.
Experimental settings are exact. Run `python -m pytest -q test_public.py` to
check the interface and calibration.
