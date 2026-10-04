# Finite-layer phoresis, revision 1

Scientific and local controls are complete. The public apparatus, starter, completed shortcut, oracle and independent finite-element/Stokes reference are ready for final frozen-hash peer attachment. No model evaluation has been launched. The canonical tasks and shared docs are unchanged.

All 256 noisy fits pass with the physical oracle and reject the planar-closure control on all three spherical groups. Reference error is at most 2.65e-10; direct force balance closes to 7.3e-13. The local oracle passes 7/7; the shortcut passes calibration, parameter and wall checks and fails its three spherical groups. Both local runs take less than 0.7 seconds.

See tasks/finite-layer-phoresis/AUTHOR.md for the mechanism, independent derivation, source overlap assessment and failure-classification rules. Scientific and local reports are under results/. The previously pushed incomplete checkpoint is preserved byte-for-byte under checkpoint/; its historical status is not the present completion status.

Root must verify final provenance and peer attachment before any Docker/model evaluations. No difficulty outcome is claimed.
