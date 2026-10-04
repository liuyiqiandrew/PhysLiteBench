# Geometric rotor revision 2 archive

The neutral-instruction conditional screen produced raw 0/1, with no infrastructure exception. The runner stopped after the initial failure, as required. This result is an implementation-induced regression and is excluded from the intended physical-model-failure count.

The agent's exploratory code constructed the complete projected derivative Gram matrix, including the internal doublet connection and correct endpoint parity. It repaired an initial duplicate FFT normalization, but then diagonalized H/inertia instead of H/(2*inertia). The missing one-half caused a false calibration mismatch; the agent subsequently chose to preserve the supplied calibrated scalar-metric predictor.

At the exact fitted inertia 1.0699072779593213, changing only the exploratory return from eigh(H) to eigh(H/2) reduces calibration chi-square from 1,686,145.62 to 1.07114. The corrected diagnostic restores every hidden group, with maximum normalized error below 7.87e-5. The uncorrected final submission has errors 0.6530, 0.6683 and 0.5918. Both internal-column choices in the diagnostic give the same corrected response; this is not an unresolved physical-subspace ambiguity.

This archive preserves the frozen task, two owned scripts, full staged bundle, all preliminary prototypes and reports, source/science/local/peer provenance, the complete trial review, five exploratory projection probes, the exact last diagnostic and fixed-inertia repair script/report, the native/public model trajectory, final submission and verifier results, both Docker controls and the matrix index. Original jobs, global ledgers/reports/prototypes, and staging remain in place.

Frozen AUTHOR prose predates evaluation. This README and the archived trial ledger record the final raw 0/1 outcome and its excluded implementation cause. Canonical task and scripts were removed only after archive and source hashes were verified. Revision 1 remains independently preserved with its 3/3 result.
