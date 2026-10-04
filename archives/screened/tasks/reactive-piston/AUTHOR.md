# Reactive piston, revision 1

The same reversible A↔2B kinetics acts in two ideal-gas chambers at fixed temperature and common pressure. Symmetric concentrations keep initial volume fractions fixed, even when those fractions differ. They identify k from time-dependent B concentration. The completed shortcut exactly solves fixed-volume nonlinear mass-action kinetics and conserves each chamber's A-equivalent atom count. All calibration preparations lie in its correct special case.

In asymmetric preparations, the reaction changes total particle counts differently. The piston moves to keep pressures equal; each volume is its molecule-number fraction times total vessel volume. This changes both concentration readouts and the concentration-dependent reverse reaction rate. Only B concentration is observed, but the apparatus fixes the unobserved volumes and A amounts.

The oracle evolves two reaction extents and eliminates volumes algebraically. The independent reference evolves four species amounts and two volumes with the differentiated common-pressure condition. The external bath holds both temperatures fixed; piston work is internal to the rigid vessel and introduces no unknown heat capacity. The elementary-event convention removes the reverse-rate factor-of-two ambiguity.

Measurement sigma is 0.0012 mmol/L; hidden relative RMS limit is 1.2%. Run scripts/validate_reactive_piston.py; --generate alone changes public/private calibration copies. This is a new task; the original thermal-bodies task and all its historical artifacts are preserved.
