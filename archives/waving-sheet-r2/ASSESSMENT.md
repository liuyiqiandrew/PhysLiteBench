# Waving-sheet revision 2

Status: scientifically validated, independently reviewed and frozen for root's
evaluation. No Docker or model evaluation of this revision has run.

The material is Oldroyd-B with known solvent fraction .25 and relaxation time
1 s. The completed source retains exact harmonic response, moving-wall
kinematics and all r1 mean fluid convection. It extends linear laboratory
polymer stress to quadratic pumping; the physical model includes nonlinear
polymer transport. AUTHOR states the limitation and the related polymer history.

Calibration contains 288 fixed-sigma readings at 48 distinct harmonic settings.
All 256 noise draws pass with the oracle and reject the source's three pumping
groups. Source anchors pass. The independent BVP agrees within 4.50e-9; local
controls give oracle 7/7 and source 4 passes with 3 intended hidden failures.
The .04 prediction gate, public frequency range and neutral instruction remain
unchanged. The reference uses independent boundary equations with the shared
constitutive tensor definition; no stronger algebraic independence is claimed.

The written final peer is
`results/physics-review-waving-sheet-r2.json`. Source and report provenance is
`results/waving-sheet-r2-source-provenance.json`. Prototype and feasibility
evidence are preserved, including the rejected corotational variant and the
initial numerical tail failure. R1's two passes, one physical failure and three
unscored interruptions remain in its archive and are linked in the history.

The top manifest freezes this package only. Root owns model launches, canonical
promotion, empirical classification and shared benchmark summaries.
