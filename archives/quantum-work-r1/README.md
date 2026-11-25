# Quantum-work revision 1 archive

The neutral-instruction conditional screen gave an initial pass and one pass among the two followups: 2/3 overall, with no infrastructure exceptions. The two successful submissions correctly implemented the distribution of two recorded energy measurements.

The failed followup initially identified the same correct physics. Its diagnostic forgot to transpose the transition matrix when weighting initial-state rows, producing a false calibration mismatch; it then reverted to the supplied operator-moment observable. This is an implementation-induced regression, excluded from the intended physical-failure count. The preserved repair diagnostic changes calibration chi-square from8193.58 to1.10293 by that one transpose at fixed fitted scale. Correcting only the final measurement statistics restores all hidden groups, with calibration unchanged to roundoff.

This archive preserves revision1 task sources, calibration, completed shortcut, validator, prototypes, science and local controls, provenance, peer reviews, all three model reviews, fixed-parameter diagnostic, and public/native trial evidence. Docker oracle and shortcut evidence are included. Original jobs and global review ledgers remain unchanged. No revision2 staged files or reports are part of this archive.

Frozen AUTHOR prose predates evaluation; this README and the archived trial ledger record the final outcome.
