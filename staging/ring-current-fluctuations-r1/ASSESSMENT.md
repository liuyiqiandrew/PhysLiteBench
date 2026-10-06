# Ring current fluctuations r1: local scientific assessment

The package passes local scientific validation. It is ready for independent
package review and frozen Docker controls; model difficulty is unmeasured.

The observed-data fit is `D=0.9370167715770505`, with reduced chi-square
`0.9547336033` for both completed controls. The oracle's three prediction-group
errors are below `4.95e-5`. The completed homogeneous-profile shortcut has errors
`0.2307086` and `0.4235179` on the two fixed strong-field groups while passing the
exact-limit group. Local pytest gives seven passes for the oracle and five
passes plus exactly those two prediction failures for the shortcut.

All 256 fixed-seed noisy calibration fits passed the unchanged thresholds.
Fitted `D` ranges from `0.9364648462` to `0.9375204031`; the largest reduced
chi-square is `1.2770602251`. The largest oracle prediction error is `0.0005868921`;
the smallest shortcut error on either distinguishing group is `0.2300126397`.
These are all-realization checks, not checks only at fitted extrema.

The self-contained independent solver reproduced all six raw grid-256 gold
values exactly in the pinned local environment, retaining 36 raw/refined
profiles at grids 128/256. The earlier 37-case domain study found maximum
elliptic/reference disagreement `3.3577e-5`, grid refinement `1.0078e-4` and
doubled-mesh change `2.5187e-5`, against fixed `0.002` gates. It retained three
raw optimizer iteration-limit failures, which remained ineligible for selection.

The physical source differs from the reviewed prototype only by enforcing the
analytical zero-current endpoint in scalar root finding and its reported
residual. The validator checks this exact source change and near-center values
on seven fields, including between the domain-study fields. Other oracle and
independent-reference functions match the reviewed prototypes by AST identity.

The first bare-Python validation startup failed before data generation because
NumPy was absent. That attempt is retained as an environment failure. The first
scientific validation completed successfully in `14.89` seconds. Its complete
report is `results/validation-20261006T100320969111Z/report.json`; the top-level
`results/validation.json` is a copy. Pytest cache-permission warnings are retained
in the logs and did not affect the test outcomes.

The gold follows the published half-filled stationary Bethe/MFT branch. The
exact finite-ring conditioned ensemble remains translation invariant; the
nonuniform macroscopic profile has an arbitrary center. Neither the sampled
domain study nor agreement of the two numerical formulations proves unrestricted
global optimality or a uniform bound over the continuous input domain.

No seeds, measurement errors, prediction cases or grading thresholds were changed
after outcomes. The original failed numerical screen remains preserved alongside
its separately frozen polishing diagnostic. No Docker controls or model trials
have yet been performed for this package.
