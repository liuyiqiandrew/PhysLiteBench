# Coarse pointer r1: independent physics review

Verdict: approve the scientific package for a frozen evaluation, subject to the separately run Docker controls. No blocking issue was found. I read the task, public and private data, oracle, completed shortcut, validator, saved validation report, and private hint. I changed no task files, regenerated no data, and ran no model trials. The twenty reviewed files are hashed below.

## Apparatus and exact response

The public preparation specifies the normalized pure Gaussian wavefunction, so `w` is unambiguously its position standard deviation. Position and displacement are in millimeters; `[q,p]=i` with momentum in inverse millimeters makes the prescribed impulse dimensionless. The initial product preparation, fresh shots, lack of free evolution or feedback, ideal pointer readout, discarded pointer, analyzer ket, and joint probability per prepared shot are explicit. No energy scale, phase reference, hidden normalization, or postselection rule must be inferred.

The impulse translates the two pointer components to `psi(q-d)` and `psi(q+d)`. Integrating over `q>=h` gives accepted population multipliers

```
a_plus  = Phi(( d-h)/w),
a_minus = Phi((-d-h)/w),
```

and coherence multiplier

```
c = exp[-d^2/(2*w^2)] Phi(-h/w).
```

The accepted qubit state for the initial `|+x>` therefore has diagonal entries `a_plus/2,a_minus/2` and real off-diagonal entry `c/2`. Projecting onto the stated analyzer gives exactly the oracle's expression, including its factors of two and `cos(phi)` sign. Acceptance is `(a_plus+a_minus)/2`, independent of analyzer angles. The rejection map has complementary populations and coherence `exp[-d^2/(2*w^2)] Phi(h/w)`; both branches together preserve trace and give the expected unconditional dephasing.

The specification selects a measurement instrument, not merely its effects. A binary projector on this same continuous pointer followed by tracing it would give the same integrated map; discarding a classical record does not restore qubit coherence. AUTHOR states this correctly.

## Completed shortcut and calibration

The shortcut is a valid binary square-root instrument with Kraus operators `diag(sqrt(a_plus),sqrt(a_minus))` and `diag(sqrt(1-a_plus),sqrt(1-a_minus))`. Their adjoint products sum to identity. The accepted coherence multiplier is `sqrt(a_plus*a_minus)`, while both diagonal entries remain exact. Cauchy-Schwarz bounds the physical overlap by this multiplier. Thus the baseline differs through a physical state-update assumption, with valid probabilities over the complete input domain.

Calibration contains 216 readings: twelve repeats of eighteen displacement/threshold/observable settings. Every threshold exceeds the displacement. Both relevant normal-CDF tails then increase strictly with positive width, proving global identification on `[.6,1.0]`. The joint calibration analyzer has `theta=0`, so its probability depends only on `a_plus/2`. The entire calibration family is exactly shared by both instruments for every possible width; calibration does not merely happen to agree at the nominal value.

I checked all eighteen settings and their analytic derivatives on 101 widths across the interval. The minimum derivative was `0.0365172927/mm`. Public and private calibration bytes match. Both controls refit to `w=0.8303658404231437`, reproducing the saved reduced chi-square `1.0954324385787944` and every saved hidden-group metric.

## Reference and independent checks

The private reference directly projects the outgoing two-component pointer wavefunction onto the analyzer and integrates its squared modulus. It does not call the oracle CDF/overlap formula or the shortcut. Acceptance instead integrates the spin-summed pointer density. This is an independent application of the microscopic Born rule, appropriate to this small exactly specified apparatus.

The validator checks adaptive-integration refinement and a 4097-point position grid. Its finite grid extends to `d+10*w`; omitted probability is bounded by the average of the two Gaussian tails beyond that point, at most the upper standard-normal tail at ten. This is negligible beside the recorded `2.553e-13` grid error. The saved 144 domain cases include displacement, threshold, width, and analyzer corners. Adaptive integral agreement is `4.441e-16`; complementary-analyzer and unconditional-overlap identities hold within `1.111e-16`. Positive accepted and rejected maps are checked separately.

For additional independent checks, I used 128 fresh random preparations across the public domain:

- Integrating all four entries of the translated-pointer Gram matrix before applying any analyzer reproduced the exact accepted map within `3.331e-16`; none had a negative eigenvalue.
- A direct controlled-rotation dilation with an ancillary qubit reproduced the completed shortcut's accepted joint probabilities within `3.331e-16`. The two binary Kraus operators satisfied completeness within `2.221e-16`.
- Reloading the frozen truth reproduced oracle group errors of `0.02052%`, `0.01807%`, and `0.02658%` on the diagnostic groups, versus shortcut errors of `40.380%`, `27.604%`, and `32.138%`.

These checks complement the earlier concept review's 96 direct wavefunction integrals. They establish both controls as complete quantum models and localize their difference to the intended instrument choice.

## Grading, separation, and fairness

The saved 256 independent calibration-noise trials all pass the parameter and calibration gates. The largest noisy oracle group error is `0.04585%`; the smallest diagnostic shortcut error is `27.602%`. The 4% relative Euclidean group threshold is comfortably between these, and much larger than quadrature error. Forty-one noiseless fits recover width within `1.654e-8` absolute, including the permitted endpoints.

The hidden tests include common anchors and three nonzero transverse-analyzer groups, including positive and negative interference contributions. True probabilities in the diagnostic groups range from `0.105255` to `0.752337`; the shortcut predictions are also nonzero. Separation does not rely on dark outcomes, tiny denominators, or a narrow numerical tolerance. Displacement, threshold, and analyzer values all remain within the public bounds.

The Docker context copies only the public apparatus, starter, public test, and calibration data. The private hint, explanation, reference, true width, frozen predictions, and oracle are outside that context. The hint describes a physical derivation and is not supplied in a plain run. The verifier compares frozen and declared hidden input lists, checks calibration and the inferred parameter, and writes each group metric. The solution installer replaces only the model. The original runtime limits and dependency pins are retained.

The source implements the approved distinct instrument task, rather than the archived quantum-work observable distinction or qubit-control initial-state approximation. No model-difficulty claim follows from these controls. Trial review should identify retention of the square-root state-update assumption separately from Gaussian-width, angle, normalization, numerical, or coding mistakes.

## Reviewed source hashes

SHA-256 values identify the reviewed bytes; generated Python caches are excluded.

| File | SHA-256 |
|---|---|
| `tasks/coarse-pointer/AUTHOR.md` | `b87260a7dd501487d90eed65f4a681894ab83daba5dcad607fb93b586aa34627` |
| `tasks/coarse-pointer/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/coarse-pointer/environment/README.md` | `3cbcb6259703549ec13a1914cf2233cc8002b976a10ba2069cf077fe974d728c` |
| `tasks/coarse-pointer/environment/data/calibration.json` | `7158b10229ddbecf195e6e6288e96280a75873b7ef310303cbd2f5cc25b57e8a` |
| `tasks/coarse-pointer/environment/model.py` | `161cd769d2fc9bd5a83faf38f3a454c6eb075560c7005c8a9a32e97af350148b` |
| `tasks/coarse-pointer/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/coarse-pointer/hint.md` | `927c53411b91f0fa15adc731da62cf48349b7385ae4c07bbfd512bc9910e6e3b` |
| `tasks/coarse-pointer/instruction.md` | `dc6563dbc1400306eb2dc697c7fb1a83de7cb298d93bd18f4fbd7eb938e744b0` |
| `tasks/coarse-pointer/solution/model.py` | `a4718fd40cd59e2f995aaa1d16b688e97aa9830d2b66b7db4d41bcc14a533d8f` |
| `tasks/coarse-pointer/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/coarse-pointer/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/coarse-pointer/tests/data/calibration.json` | `7158b10229ddbecf195e6e6288e96280a75873b7ef310303cbd2f5cc25b57e8a` |
| `tasks/coarse-pointer/tests/data/hidden_truth.json` | `080221d84f03dddc0455648120d04ad933ca319df21be5de06e5723cbdfe9331` |
| `tasks/coarse-pointer/tests/metadata.json` | `b9a20f300e1f93cb9193564b3af8439692e8525b3f207e92d4ba1644b7e7b661` |
| `tasks/coarse-pointer/tests/reference.py` | `ad9f36055f6e677584df0c0476803e1e0ddf6a3c356f02e7ddaf4315585385f5` |
| `tasks/coarse-pointer/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/coarse-pointer/tests/test_hidden.py` | `e5e87b72bbbbe2f6710c9d225c97f4ea4ef93b6b6bf647bba8aa7530ef146e68` |
| `scripts/coarse_pointer_baseline.py` | `1a1330ce65bc9861f9dc8dbae92692b99534592925ffc341d30ebf409f23ae1f` |
| `scripts/validate_coarse_pointer.py` | `8be9627b5e302bcd13a0273d68194dc6aaaa50fec3d4ab7925b90d93608c5ead` |
| `results/expansion-coarse-pointer-r1-validation.json` | `620caf15dd5aedad7cced65d20f2bc2cfac3b4465dd110ba6ab319f80b4ff20b` |
