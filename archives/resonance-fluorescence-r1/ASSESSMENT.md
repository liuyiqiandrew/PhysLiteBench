# Resonance fluorescence, revision 1

Frozen after scientific validation, isolated completed controls and independent
final peer approval. No model evaluations have been run. Root alone schedules
Docker controls and the frozen unhinted batch. Final review:
`results/physics-review-resonance-fluorescence-r1.json`.

The completed source solves the exact driven emitter dynamics and stationary
population fluctuations, then treats those fluctuations as nondestructive
photon-intensity noise. Mean-count calibration matches exactly and identifies
efficiency. The stated broadband fluorescence counter instead measures actual
emission events. Its finite-gate variance requires different conditioning.

All 256 noise draws pass calibration and parameter checks for both models.
The oracle passes every hidden group; source errors remain above .855 at the
.04 gate. The 64-case independent direct count-moment check agrees to 1.96e-12;
count-resolved, tilted-generator, thinning and limiting checks pass. Local
controls give 7/7 versus 4 passes and the three intended variance failures,
in .47 and .45 seconds. Signals are nonzero and source variances positive.

The stage contains 18 source/control files, three scientific/local/history
reports, four prototype files, provenance and the final peer. Calibration has
24 unique settings repeated six times with fixed sigma .0004. The exact neutral
instruction and 600/60-second limits are retained. A private validator tuple
conversion bug was fixed before the final report, without physics, data or
grading changes; the history records it. Canonical and shared files were not
changed. Familiar photon antibunching may make this easy; empirical difficulty
remains untested.
