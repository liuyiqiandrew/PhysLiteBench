# Ternary gas: revision 1

The task distinguishes a conservative mixture-averaged approximation from
pairwise momentum balance. With ideal chemical potentials, negligible inertia,
and equal molecular masses, the molar diffusive fluxes j obey

    -grad(x_i) = sum(j != i) (x_j*j_i - x_i*j_j)/D_ij,
    sum_i j_i = 0.

Constant total molar density and sealed ends enforce the latter condition.
In a binary preparation these reduce exactly to Fick diffusion with its binary
coefficient. The starter solves each species' mixture-averaged Fick flux and
subtracts a composition-weighted total flux. This preserves total density and
individual closed-cell inventories, and is exact on every binary calibration.
It does not satisfy all pairwise friction balances in a ternary preparation.

The physical mechanism is established in the original
[Duncan and Toor experiment](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.690080112).
The reduced equations and closed boundary condition are also given in the
[Boudin, Grec, and Salvarani analysis](https://mate.unipv.it/~salvaran/BoudinGrecSalvarani.pdf).
The task uses an idealized apparatus and generated measurements, not their data.

Calibration uses the three binary pairs, D=.035 mm²/s, independent Gaussian
uncertainty .002 in every mole fraction, and seed9400. Its continuum reference
uses an eightfold refined mesh. The finite-volume oracle and shortcut have
identical binary predictions. The fitted D is .03514893, reduced chi-square
.98546. The small discretization bias remains far below the 3% parameter limit.

Hidden cases use three smooth ternary profiles. The oracle solves a constrained
3×3 friction system at each face. The independent reference eliminates the
third flux analytically, evolves two conserved fractions, and uses a fourfold
refined mesh. Eightfold refinement changes fractions by at most8.71e-6. The
predictions conserve each species to2.8e-16, remain positive, and decrease the
ideal mixing free energy. Hidden errors are normalized by the RMS departure
from the conserved uniform composition, rather than by the constant background.
The threshold is .04; the fitted oracle gives .00095–.00138 and the shortcut
.0834–.1880.

The256-noise check (seed19400) validates calibration and parameter recovery on
every draw. All pass; maximum parameter error is .646%. Hidden predictions at
both fitted-parameter extrema give oracle error below .00235 and shortcut
error above .0826. This is an extrema sensitivity check, not a full hidden
Monte Carlo evaluation. `scripts/validate_ternary_gas.py --generate` regenerates
both copies of calibration; ordinary validation leaves data unchanged.

An independent teammate reviewed the physical uniqueness, matrix signs,
binary equivalence, reference elimination, and closed-cell boundary weights.
Agent evaluation gives 0/3 unhinted and 3/3 hinted; all three unhinted failures retain the mixture-averaged physical closure. Full results are retained separately.

## Reviewed evaluation

The frozen revision scores 0/3 unhinted and 3/3 hinted on GPT-5.6 Luna high, Codex CLI 0.154.0 and Harbor 0.21.0, with 600/60-second agent/verifier limits. All three ordinary failures pass calibration and parameter recovery and are reviewed physical-model failures. Fixed-parameter repairs recover the correct hidden predictions. See `results/candidates.json` and `results/hardening-status.json` for job IDs, per-trial reviews and matched-source checks. These are development-batch outcomes, not a future failure-rate guarantee.
