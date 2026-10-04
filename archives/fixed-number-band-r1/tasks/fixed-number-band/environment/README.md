# Differential calorimetry of a two-band solid

An ideal solid has two flat bands, each containing M spinless one-particle states. Their energies are -W/2 and +W/2. The particles are noninteracting fermions, and each one-particle state admits at most one particle. The gap W is independent of temperature and has the same unknown value in every experiment. Geometry and one-particle energies are held fixed during measurement.

Each sample contains exactly N particles. A weak thermal contact permits complete equilibration, including transitions between bands, but cannot exchange particles with the sample. Equilibration is taken before the contact energy is neglected. There is no other constraint on either band's population.

A differential calorimeter measures the reversible heat supplied to the sample per temperature increment. Contributions from the empty support and the thermal contact are subtracted. Divide the reading by the total number 2M of available one-particle states and report its thermodynamic limit as M,N increase with the stated filling n=N/(2M) fixed. The temperature change is quasistatic, and the sample remains closed to particle exchange.

Use energy unit E0, temperature unit E0/kB, and heat-capacity unit kB per available one-particle state. The common unknown width W/E0 lies in [.8,1.2]. Each experiment contains exact controls:

- `temperature`: T in [.1,.6].
- `filling`: n in [.1,.9]. Different experiments may use samples prepared with different fillings.

Records in `data/calibration.json` contain `input`, `value` and `sigma`. Measurement errors are independent Gaussian with fixed standard deviation sigma=.0003 in the stated heat-capacity units.

Implement `Model.fit(records)`, returning `self` and storing the estimated gap as `Model.width`. Implement `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)` containing the calorimeter readings. Run `python -m pytest -q test_public.py` for the public check.
