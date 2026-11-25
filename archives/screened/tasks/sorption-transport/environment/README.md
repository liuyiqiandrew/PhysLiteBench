# Transport through sorbing chambers

Twelve identical, well stirred liquid chambers form a line. Neighboring chambers
exchange dissolved A and B through identical thin membranes; there is no flow.
If their dissolved concentrations are c_left and c_right, the net amount of A
transferred to the right per unit time and per chamber liquid volume is
k*(c_left-c_right). For B the coefficient is 3*k. The two ends are closed.
The common coefficient k is unknown, between 0.1 and 1.2 per second.

Each chamber contains the same immobile solid with 4.0 mmol of equivalent
binding sites per liter of liquid. A site is empty or binds one molecule, A or B.
Binding is reversible and much faster than interchamber exchange. At the fixed
temperature, the ratios [SA]/([S]*[A]) and [SB]/([S]*[B]) are respectively
2.0 and 1.0 L/mmol. Brackets denote concentrations per liquid volume. The liquid
is ideal and dilute. There are no other interactions, reactions, or site classes.
Only dissolved molecules pass through the membranes. Chamber liquid volumes
and solid site populations are constant.

The initial dissolved concentrations in every chamber are specified after local
binding equilibrium has been established. The apparatus is then isolated and
exchange starts at time zero. Measurements report dissolved concentrations.

## Interface

Implement Model in model.py. fit(records) returns self and sets rate to k in
per second. predict(experiments) returns a finite NumPy array of shape
(len(experiments),) in mmol/L. You may edit the entire implementation.

Each experiment contains initial, a 12-by-2 list of dissolved concentrations in
chamber order (A then B, each between 0 and 2 mmol/L); time, elapsed seconds
between 0 and 120; chamber, an integer from 0 to 11; and species, 0 for A or
1 for B. Predict all allowed experiments with the same fitted rate.

data/calibration.json contains records with input, measured value, and sigma,
the known independent Gaussian measurement standard deviation in mmol/L.
Experimental settings are exact. Run python -m pytest -q test_public.py.
