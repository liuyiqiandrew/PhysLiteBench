# Finite-layer phoresis: checkpoint

This is an incomplete staged candidate, frozen for the requested repository checkpoint. It is not ready for model evaluation. No canonical task or shared documentation was changed.

The public apparatus, fit-stub starter, completed planar-hydrodynamic closure, physical oracle, independent solute finite-element/Stokes reference, standard harness and scientific validator are written. The independent source/public physics review passed. The preserved 48-case prototype compares the reciprocal force kernel with a direct force-free Stokes boundary-value calculation to 2.42e-12. A separate current-reference spot check covers the 16 hidden cases and agrees with the oracle to 2.33e-11; it used about 0.21 seconds. These are scientific checks, not model trials.

Pending: intentionally generate calibration data; run the complete 256-noise and domain validator; run both local controls; write AUTHOR; attach final peer review and frozen source provenance. The calibration directories are currently empty. The validator was written but has not been executed. Existing prototype files and reports remain preserved.

The physical approximation is the locally planar hydrodynamic transfer kernel, applied after solving the full curved solute concentration field. Flat-wall flow calibration is exactly shared and identifies viscosity. Finite-width sphere motion requires the curved force-free Stokes response, including fluid traction and the direct reaction of the solute interaction. The independent source reviewer checked these conventions and the total solution-pressure convention.

Archive scan covered92 distinct AUTHOR task families and found no prior diffusiophoresis or diffusio-osmosis task. Related screened electroviscous and ionic tasks concern different charge/current constraints. The continuum force-transfer mechanism is distinct; no model-difficulty outcome is claimed.

Primary background: Marbach, Yoshida and Bocquet, Local and global force balance for diffusiophoretic transport, Journal of Fluid Mechanics (2020), https://pmc.ncbi.nlm.nih.gov/articles/PMC7145454/ . The source and reference are independently derived for the specific compact potential in this stage.
