# Electroviscous flow, revision 2

Revision 1 used a one-pass streaming-potential estimate and omitted the induced flow's contribution to total current. Two of three plain agents corrected that feedback. Its source, original hidden preparations, validator and control remain in `archives/screened/tasks/electroviscous-flow` and `archives/screened/scripts`, with the original jobs and reviewed results preserved.

Revision 2 supplies correct self-consistent convective-current feedback. The remaining approximation uses reservoir ionic conductivity throughout the slit, while retaining the full nonlinear transverse Poisson–Boltzmann charge distribution and its electroosmotic flow. This is a bulk-conduction approximation that neglects excess diffuse-layer migration current. It is valid when that excess is small; it is not an algebraic or coding mistake.

Write spatial averages as brackets. The pressure velocity is u_p=g(h²-y²)/(2 eta), and the response per axial electric field is u_E=epsilon[phi(y)-phi(h)]/eta. Let I_p=<rho*u_p>, A=<rho*u_E>, G=<sigma(y)>, and G_bulk=2 F² c0 D/(RT). Both controls solve the current constraint of their respective conductive law:

    oracle:   E=-I_p/(G+A)
    shortcut: E=-I_p/(G_bulk+A).

The local conductivity required by the specified ideal ions is sigma(y)=F² D[c_plus(y)+c_minus(y)]/(RT), where c_plus=c0 exp(-phi/Vt) and c_minus=c0 exp(phi/Vt). The shortcut includes all charge convection but replaces the sum of mobile populations in axial migration by its reservoir value.

The shortcut remains passive and reciprocal. Its open-circuit flow is positive, its approximate total current vanishes, and pressure work equals viscous dissipation plus G_bulk*E². Thus a negative conductance or obvious violation of its own current constraint cannot diagnose it. Its wrong migration law leaves a residual when evaluated against the physical local conductivity. Neutral-wall calibration is exactly shared and still identifies viscosity.

The public apparatus, calibration bytes, oracle, independent reference equations, and .025 prediction tolerance are unchanged. The starter, hint, revision metadata and hidden charged preparations changed. The original r1 charged preparations gave only 1.68–1.90% error for this new closure, so they were not diagnostic at the retained tolerance. The new cases are strictly within the original public ranges:

| Group | Half-gap (micrometres) | Reservoir concentration (mol/m³) | Wall charge (C/m²) |
|---|---:|---:|---:|
| positive_charge | .22 | .0065 | .00032 |
| negative_charge | .38 | .007 | -.00034 |
| overlapping_layers | .24 | .008 | .00032 |

Each group retains signed pressure gradients and mean/point readouts. Diffuse-layer ohmic conductance is 1.30–1.58 times the reservoir value. The completed shortcut's hidden relative RMS errors are .2443/.1048/.1634, while the oracle errors are below .000182. Both recover eta=.00119963271 with reduced chi-square .95823. Calibration seeds remain 9314 and 19314. All 256 additional calibration fits pass; hidden checks at their fitted-parameter extrema give oracle error below .00156 and shortcut error above .1039. This is a calibration Monte Carlo plus extrema sensitivity, not 256 full hidden runs.

The independent reference solves the joint electrostatic, momentum and total-current boundary-value problem, without the oracle's eliminated electric-field denominator or analytic velocity response. Oracle/reference agreement is 3.83e-13 relative; refinement changes predictions by 4.12e-13. Charge neutrality, reciprocal hydraulic coupling, the positive gradient-integral identity for A, and correct pressure-power balance all agree below 4e-13. The shortcut closes its approximate current to 2.01e-16 and satisfies its dissipation identity to 2.35e-13. Its hidden mean flow remains at least .254 of pressure-only flow. All allowed geometry/viscosity corners retain positive correct and approximate flow.

Primary physical context: [Levine, Marriott and Robinson, Theory of electrokinetic flow in a narrow parallel-plate channel](https://doi.org/10.1039/F29757100001), which treats nonlinear double-layer overlap and coupled electrokinetic flow. The task states its own ideal constitutive assumptions and fixed-charge boundary conditions.

Reproduce with `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_electroviscous_flow.py --noise-trials 256`. Use `--generate` only for intentional calibration replacement. The shareable report is `results/electroviscous-flow-r2-validation.json`. Local isolated controls give oracle 7 passed and shortcut 4 passed / 3 intended hidden failures. They use the actual public and private tests; final peer review and Harbor results are recorded separately.
