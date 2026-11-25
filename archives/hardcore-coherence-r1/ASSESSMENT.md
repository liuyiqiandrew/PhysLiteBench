# Hard-core coherence staging status

Scientific validation and local controls are complete. This stage has no model evaluations yet. The neutral instruction, .04 hidden gate and 600/60-second limits are unchanged.

The supplied forward model propagates all occupied orbitals exactly and gives exact hard-core boson densities, but uses fermionic off-diagonal coherence for a bosonic release image. The private oracle restores the physical one-body operator; the verifier independently evolves the fixed-number bosonic Hamiltonian. Public preparation, open boundaries, sudden interaction-free release and envelope normalization define the measurement without giving its solution.

All 256 noisy fits pass the oracle and reject the completed shortcut. Local tests give oracle10/10 and shortcut7passes/3 intended failures. These are scientific controls, not model success-rate evidence. Details and immutable source hashes are in results/hardcore-coherence-r1-validation.json, results/hardcore-coherence-r1-local-controls.json and results/hardcore-coherence-r1-source-provenance.json. The original two-file prototype is preserved under prototype/.

The archive comparison in results/archive-overlap-audit.json documents related statistics tasks and the different physical operator here. The root's separate hard-core ring prototype is an alternate revision of this family. Neither is counted as qualified without reviewed fresh model outcomes.
