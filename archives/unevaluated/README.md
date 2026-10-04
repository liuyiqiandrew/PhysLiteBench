# Unevaluated backups

`tasks/dielectric-flow` is a scientifically validated revision 1 backup. It has not been evaluated with Harbor or a solving agent and makes no difficulty claim. It was archived when the retained benchmark reached its requested ten zero-score tasks.

Completed: full oracle and completed shortcut, independent finite-volume Maxwell-stress reference, equal public/private calibration, 256 noisy calibration fits, observed-fit-extrema hidden checks, current conservation, incompressibility, Maxwell-force identity, mechanical work, field reversal, translation, and grid refinement. Local public/private verifier controls give oracle 7 passed and shortcut 4 passed with exactly 3 intended hidden failures. The final source peer review passed. Author notes are complete.

The science and local-test report is `results/dielectric-flow-validation.json`; the peer review is `results/dielectric-flow-physics-review.json`. No scientific checks remain pending. Docker/Harbor controls and solving-agent evaluations were deliberately not run, so no measured-difficulty claim is made.

The validator reads the archived task and baseline directly. From the repository root with the pinned NumPy 2.3.3/SciPy 1.16.3 environment, run:

    OPENBLAS_NUM_THREADS=1 python archives/unevaluated/scripts/validate_dielectric_flow.py --noise-trials 256 --output /tmp/dielectric-flow-validation.json

Ordinary validation does not regenerate data. The manifest records every preserved file except itself.
