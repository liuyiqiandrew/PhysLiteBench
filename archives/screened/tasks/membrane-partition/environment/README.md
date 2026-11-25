# Salt partition records

A membrane separates two closed, well stirred liquid compartments, a gel and a
bath, at fixed temperature. Their liquid volumes remain fixed. The gel contains
immobile, permanently ionized negative groups with a specified concentration Q
in mmol of elementary charge per liter of gel liquid. The bath contains no
immobile charge. The permeating small ions are A+ (valence +1), B2+ (valence +2),
and C- (valence -1). There is one chemical species C-. All small ions cross the
membrane freely. No molecule binds to the fixed groups, and there are no reactions.

Use ideal dilute solution thermodynamics. Every small ion has the same
non-electrical standard chemical potential increase Delta_mu on transfer from
bath to gel. The common partition factor P=exp(-Delta_mu/(R*T)) is unknown,
between 0.2 and 2.0. R is the molar gas constant. Both bulk compartments are
electrically neutral; charge in the thin membrane double layers is negligible
in the amount balances. The membrane passes no solvent, its volume is negligible,
and there is no imposed voltage or external reservoir. Measurements are taken
after complete equilibrium.

## Interface

Implement Model in model.py. fit(records) returns self and sets partition to
P. predict(experiments) returns a finite NumPy array of shape
(len(experiments),), reporting mmol/L. You may edit the entire implementation.

Each input provides gel_volume and bath_volume in liters, fixed_charge Q in
mmol/L, and amount_a and amount_b, the total mmol of the respective cations in
both compartments together. The total mmol of C- is amount_a+2*amount_b minus
fixed_charge*gel_volume; allowed inputs make this positive. The field species
is 0 for A+, 1 for B2+, or 2 for C-, and compartment is gel or bath.
Volumes are between 0.2 and 2 L, Q between 0 and 2 mmol/L, and cation amounts
between 0 and 8 mmol. An absent cation has zero amount. Use one P for every
preparation and report the requested dissolved ion concentration.

data/calibration.json contains records with input, measured value, and sigma,
the known independent Gaussian measurement standard deviation in mmol/L.
Experimental settings are exact. Run python -m pytest -q test_public.py.
