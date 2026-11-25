# Blocking ion cell with a series capacitor, revision 1

Neutral-salt relaxation does not observe the electric charge mode. Equal
monovalent diffusivities and equal initial concentrations make charge density
identically zero at zero source voltage, for every calibration time. The
neutral cosine moments identify D exactly under both controls. Hidden voltage
steps require the local electrostatic constraint as well as global conservation.

Let x be scaled by L, concentrations by c0, potential by Vt=RT/F, and electrode
charge q=Q/(F*c0*L). Define kappa=F*c0*L^2/(epsilon*Vt), r=epsilon/(Cs*L),
and p=integral (1-x)*(c_plus-c_minus) dx. Integrating Gauss's law from the
positive left electrode and imposing the series-capacitor drop gives

    q = (v/kappa-p)/(1+r),
    Ehat(x) = kappa*[q+integral_0^x (c_plus-c_minus) dy],
    v_gap = kappa*(q+p),   v = v_gap+kappa*r*q.

Here v is the applied source voltage in thermal-voltage units. Positive left
voltage gives a field toward +x and pushes positive ions to the right. The
right electrode is grounded; total electrolyte charge stays zero, so its
opposite electrode charge is consistent. Fresh zero-voltage preparation fixes
the series-charge offset. Time-zero predictions are after instantaneous metal
adjustment, with unchanged ionic concentrations.

The completed shortcut evolves both ion populations with conservative blocking
fluxes and uses the correct global q relation. It replaces Ehat(x) by its
spatial mean kappa*(q+p). Thus each ion count and the source/capacitor voltage
relation pass, while local Gauss's law fails once charge separates. It is not
merely an uncharged diffusion solver or a charge-nonconserving implementation.

The oracle uses Scharfetter-Gummel fluxes with self-consistent integral fields
on 96 cells. The independent reference uses physical SI units, a separate
cell-centered Poisson matrix including the electrode/capacitor boundary row,
and central conservative drift-diffusion fluxes on 160 cells. It reconstructs
Q from the electrode surface field. Both are checked against doubled spatial
resolution. Initial neutral-mode calibration uses its exact diffusion solution.
The moderate Debye length is about eight percent of the gap; the grid resolves
the screening layers. No electroneutral bulk approximation is made.

All measured moments and electrode charge are dimensionless as defined in the
public interface. Each hidden preparation combines four observables and uses
norm(error)/norm(truth), with threshold 0.04. The reference norm is nonzero.
Calibration seed 9312 uses D=1.1e-9 m²/s, 160 independent readings, and sigma
0.00148380. The common fitted D is 1.1025593e-9, reduced chi-square 0.86310.
Oracle hidden scores are 0.00064–0.00105; shortcut scores are 0.221–0.583.

Refining the independent reference changes predictions by at most 0.000409
relative, and oracle/reference disagreement is below 0.001449. Both controls
conserve each ion count to 4e-16 and satisfy the global dimensionless voltage
relation to 7e-15. Oracle local Gauss residual is below 3e-13. Concentrations
remain positive; the fixed-voltage free energy decreases to numerical accuracy.

Noise seed 19312 supplies 256 calibration realizations. All calibration checks
pass, and maximum D error is 0.838%. Hidden predictions at the observed fitted
parameter extrema give oracle scores below 0.002843 and shortcut scores above
0.22049. These are 256 calibration fits plus two extrema sensitivity checks,
not 256 complete hidden evaluations or an exhaustive noise bound.

Run `scripts/validate_blocking_ion_cell.py`; the report is
`jobs/blocking-ion-cell-validation/summary.json`. Ordinary validation reads
frozen public/private calibration; only `--generate` replaces both copies.
The completed shortcut is `scripts/blocking_ion_cell_baseline.py`. A teammate
reviewed circuit charge/sign conventions, electrostatic uniqueness, initial
preparation, both flux formulations, and the independent boundary matrix.
Agent difficulty remains to be measured in frozen Harbor batches.
