# Disordered wave chain r1: independent physics review

Verdict: approve the scientific package for a frozen evaluation, subject to the separately run Docker controls. No blocking physics, specification, reference, or grading issue was found. This review makes no claim about model difficulty. I read all 16 task files, the completed shortcut, validator, and saved validation report. I changed no task or control source and ran no model trials.

## Apparatus and calibration

The public wave equation, derivative jump, identical surrounding medium, one-sided incident wave, and power readout fix the scattering convention. The derivative jump is `2*b*psi`, with `b` and `k` both in inverse millimeters. Thus the amplitude is `t=k/(k+i*b)` and the single-barrier power is `T1=1/(1+(b/k)^2)`. Identical media make the power ratio `|t|^2`; no additional velocity or admittance factor is missing.

The gap distribution `L=(20*pi+u)/k` produces exactly uniform propagation phase modulo `2*pi`, positive ordered separations, and no extra parameter. The ensemble is prepared at each requested wave number. Unlimited incident coherence, fixed barriers throughout repeated reflections, and absence of phase noise remove the main operational ambiguity: phases change between realizations, not between visits to a gap. The arithmetic mean is explicitly distinguished from amplitude and logarithmic averages. Counts zero through twelve, their edge cases, wave numbers, unknown interval, API, and additive Gaussian measurement noise are stated.

Let `r=(b/k)^2`. Both models give `1/(1+N*r)` for all calibration preparations `N=1,2`, throughout the parameter interval. Each such mean strictly decreases with positive `b`, so calibration identifies the strength globally. The positive parameter restriction removes the sign ambiguity of power-only data. The 192 public and private readings are byte-identical. I reproduced both fits at `b=0.8298243791550319`; the saved reduced chi-square is `1.0449762218775427`.

## Physical law and reference

The scalar resistance `R=(1-T)/T` of a coherent partial chain satisfies

```
R' = R+r+2Rr + 2 sqrt[R(1+R)r(1+r)] cos(theta).
```

An independent uniform new gap makes `theta` uniform even though the partial chain has a nontrivial reflection phase. Integrating the last phase gives `E[T'|T]=T/(1+r*T)`. This yields the exact two-barrier calibration law and the diagnostic identity `E[T3]=1/[(1+r)*sqrt(1+4r)]`. Since the conditional mean is a strictly concave function of `T`, replacing a fluctuating partial chain by its mean loses information from the third barrier onward.

The oracle's conical-function angular representation, integration weights, factor `2*pi`, spectral power `N`, and radial endpoint are correct. The primary paper's averaging operator and its mean-transmission formula agree with these conventions: [Ng and Englert, equations 10, 33, and 34](https://arxiv.org/pdf/1212.1951). The tail estimate `4*(Q+1/pi)*exp(-pi*Q)` follows from the conical-function magnitude bound and is `4.5361e-18` at `Q=14`. This bounds truncation, not quadrature or roundoff error; AUTHOR correctly makes that distinction.

The private reference instead iterates the finite radial transition kernel on `x=acosh(2/T-1)` and the observable `sech(x/2)^2`. The shrinking intervals contain the complete reachable support, and the zero derivative at the radial origin matches its even extension. It does not call the spectral oracle. The validator also multiplies the original complex delta-barrier and propagation matrices on independent static samples. Those stochastic checks test the microscopic mapping and are correctly excluded from grading truth. The exact small-count identities, deterministic cross-representation checks, refinement, and direct wave checks provide sufficient independence.

The shortcut is a completed, flux-conserving intensity model including repeated reflections, with transmission `1/(1+N*r)`. It is positive and bounded. Its erroneous physical assumption is replacing coherent propagation through fixed random gaps by dephased intensity propagation before averaging. It is not an incomplete transfer calculation or an invalid distributional closure.

## Independent checks and numerical margins

In addition to reading the saved report, I performed these checks without regenerating any data:

- Raw spectral integration at counts zero through three, bypassing the oracle's small-count shortcuts, matched the four closed identities within `1.71e-14` over five strength/wave-number pairs including domain corners.
- For counts `3,7,12` at those five pairs, doubling only the radial grid changed the kernel by at most `1.137e-12` absolute; doubling only angular order changed it by at most `1.066e-14`.
- Independent complex transmission/reflection amplitude composition on 20,480 configurations agreed with direct delta-transfer products within `1.869e-13` in power. Power conservation held within `1.056e-12`, including the strongest twelve-barrier case.
- At all counts `0..12` for the same five parameter pairs, the spectral mean was positive, decreased with count, and did not exceed the incoherent mean.
- Reloading the checked-in data reproduced every saved oracle and shortcut group error. The oracle's largest error was `0.06355%`; shortcut errors were `17.599%`, `92.730%`, and `29.666%` on the three diagnostic groups.

The author's validator additionally records 88 domain cases, 41 noiseless parameter-recovery cases, 256 calibration-noise realizations, and six microscopic ensembles of 262,144 samples. Its largest oracle/kernel absolute difference is `1.249e-12`; its largest spectral refinement difference is `1.744e-14`. Across noise draws, the worst oracle group error is `0.15694%` and the smallest diagnostic shortcut error is `17.4898%`. These leave ample separation around the 4% relative Euclidean group threshold. The 3% parameter gate and chi-square threshold are also satisfied with substantial margins. Hidden means remain nonzero; no zero-valued target is responsible for failure.

## Separation and evaluation scope

The Docker context copies only public apparatus, starter, public test, and calibration data. Private truth, true parameter, author explanation, reference, and solution are outside it. The private verifier checks frozen input lists against the reference's declared lists, validates calibration and the fitted parameter, and records the four group errors. The oracle installer changes only the model. Existing 600/60-second limits and dependency pins are retained.

The closest reviewed archives address power normalization in a film, dynamic correlations in a coherent array, or the distinction between residence and scattering-delay times. Their physical omissions differ from static phase disorder versus loss of coherence. The preceding concept review records the novelty assessment.

A trial retaining intensity addition because phase averaging supposedly removes all coherent returns has the intended physical error. A trial constructing the correct coherent wave problem but failing sampling, quadrature, or implementation must be classified separately. No trial outcome has been used to choose these source-review conclusions.

## Reviewed source hashes

SHA-256 values below identify the reviewed bytes. Generated Python cache files are excluded.

| File | SHA-256 |
|---|---|
| `tasks/disordered-wave-chain/AUTHOR.md` | `ab78a89c3f65c5f82a250b4dd106d6988e0ffbbe149131d8926de73cf231c2c6` |
| `tasks/disordered-wave-chain/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/disordered-wave-chain/environment/README.md` | `4a8feef790a2582e0acb2f6a8e57a43f940967a1fbf1aac1380e64bdd11894ce` |
| `tasks/disordered-wave-chain/environment/data/calibration.json` | `8a1deaa168b74b4ec32bf348e08a14c8dd85f70c0cd7cd37accf38f2acc4f1ba` |
| `tasks/disordered-wave-chain/environment/model.py` | `80df6278d48dfdd4450cd22ba0a5e192c1e1786574618d28445ee94da529e232` |
| `tasks/disordered-wave-chain/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/disordered-wave-chain/instruction.md` | `578843551f0393ac6f918f4493a9cc0e884ec1ee467b7f55205b1581e8716065` |
| `tasks/disordered-wave-chain/solution/model.py` | `4bf5d262dddf79660d0d56e9d176609ed9e15272d39cd16aede2c68b5c67643a` |
| `tasks/disordered-wave-chain/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/disordered-wave-chain/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/disordered-wave-chain/tests/data/calibration.json` | `8a1deaa168b74b4ec32bf348e08a14c8dd85f70c0cd7cd37accf38f2acc4f1ba` |
| `tasks/disordered-wave-chain/tests/data/hidden_truth.json` | `e269dc3894a73e61d7a4eadb06c443a767b64ccf19bf00551dbd980e528b39b7` |
| `tasks/disordered-wave-chain/tests/metadata.json` | `b9a20f300e1f93cb9193564b3af8439692e8525b3f207e92d4ba1644b7e7b661` |
| `tasks/disordered-wave-chain/tests/reference.py` | `a4f246ff9b63ab09d1787d20cd42bb272ce6b8140da8166baa4c9448b27a22aa` |
| `tasks/disordered-wave-chain/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/disordered-wave-chain/tests/test_hidden.py` | `537b4665a6db383c37d97d017449903847980bc171b5a10050ccd6c3983dc2f6` |
| `scripts/disordered_wave_chain_baseline.py` | `57adfcd756b6815abe0b167057742e2ca23ebc9534b3916f245e01a099f1690c` |
| `scripts/validate_disordered_wave_chain.py` | `137a89a3691278b01351c00a8a26d2e2b521d66ac826dc276c1c3b48533b3cf3` |
| `results/expansion-disordered-wave-chain-r1-validation.json` | `c5842ab181ce59d6803997e34b491efa29e767139d9c652924c2c5d861cdf496` |
