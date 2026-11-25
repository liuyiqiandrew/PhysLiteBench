# Adsorption on a uniform surface

A solid surface has a fixed population of equivalent adsorption sites in
contact with an ideal dilute solution containing species A and B. Each site can
be empty or hold one molecule of either species. The adsorbed molecules do not
interact with molecules on other sites. Binding is reversible, with one
adsorption/desorption equilibrium constant for each species at the fixed
experimental temperature. These constants, K_A and K_B, are unknown and have
units L/mmol. A pure-species equilibrium obeys the Langmuir isotherm.

The solution is a large, well stirred reservoir: its dissolved concentrations
remain constant during adsorption, and the adsorbed amount does not deplete it.
There are no reactions, multiple layers, distinct classes of sites, or changes
to the surface. Measurements are taken after equilibrium is reached.

The apparatus supports two protocols:

- `equilibrium`: equilibrate the surface with concentrations c_A and c_B and
  measure the fraction of all surface sites occupied by the requested species.
- `displacement`: first equilibrate with only the requested species at its
  stated concentration. Then add the other species to its stated concentration,
  holding the first concentration fixed, and equilibrate again. Measure the
  decrease in the requested species' occupancy as a fraction of all surface
  sites. This is the initial occupancy minus final occupancy, not the fraction
  of initially occupied sites lost.

## Interface

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`affinity_a` and `affinity_b`, each between 0.2 and 3.0 L/mmol.
`predict(experiments)` must return a finite NumPy array with shape
`(len(experiments),)` of dimensionless fractions. You may edit the entire
implementation within this API.

Each experiment is a dictionary with `protocol` (`"equilibrium"` or
`"displacement"`), `species` (`"A"` or `"B"`), `concentration_a` and
`concentration_b` (both in mmol/L, between 0 and 1.2). For `displacement`, both
concentrations describe the final reservoir. Predict all allowed experiments
using the same two fitted equilibrium constants.

`data/calibration.json` is a list of records. Each has an `input` experiment,
measured `value`, and known independent Gaussian measurement standard deviation
`sigma`. Experimental settings are exact.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
