# Quantum work, revision 2 — staged and unevaluated

This is a scientific revision of the same operational quantum-work task. It changes the preparation to an exact uniform classical mixture, calibrates variance and nonzero third cumulant, and tests the fourth cumulant. It does not create a new task family. The running revision1 and all its evidence are left unchanged; promotion requires its separate archive.

The source computes the exact initial state and both correctly ordered finite-pulse unitaries. It assigns the energy-change operator W=U†H0U−H0 its ordinary spectral statistics. That is a valid single-observable measurement model, with correct matrix powers, centering, and cumulant subtraction. It does not implement the two projective energy records specified by the apparatus. The error concerns which physical measurement is modeled, not eigenvector normalization or a mistaken fourth-cumulant formula.

## Apparatus and calibration equivalence

Each independent repetition chooses one of the three H0 eigenstates with probability1/3, then records energy before and after the isolated cyclic pulse sequence. This is a finite exact preparation, without an infinite-temperature limit or unknown population. The public files define the measured difference and its population cumulants, supply all Hamiltonians and pulse controls, and give no transition-probability formula or numerical-method mandate. The unchanged neutral instruction permits replacing any implementation while preserving the API. The private hint is outside the Docker image.

Write A=H0 and B=U†H0U. With rho=I/3, trace cyclicity makes the first three moments, and therefore the first three cumulants, of the two constructions equal for every scale and every pulse sequence. For example,

    3*m3_records = Tr(B³−3B²A+3BA²−A³)
                 = Tr[(B−A)³].

This strengthens the calibration: noncommuting drives, nonzero third cumulants and the exact unitary are already supported. The fourth-order difference is

    kappa4_records−kappa4_operator
      = 2*Tr(B²A²−BABA)/3
      = Tr([A,B]†[A,B])/3.

The correction is nonnegative. Means vanish in this cyclic maximally mixed preparation, but third cumulants need not vanish: a three-level unitary can have asymmetric transition probabilities. All six selected calibration protocols have a nonzero third cumulant throughout the sampled allowed scale interval.

The operational distinction follows the primary treatment of quantum work by Talkner, Lutz and Hänggi, [Fluctuation theorems: Work is not an observable](https://arxiv.org/abs/cond-mat/0703189), Physical Review E75,050102(R)(2007). The fourth-order trace identity above is derived directly for this task's uniform preparation.

## Independent checks

The oracle enumerates initial/final transition probabilities using matrix exponentials. The private reference independently propagates all basis columns with an adaptive Schrödinger solver, evaluates Tr[exp(zH0)U exp(−zH0)rho U†], and extracts four raw moments by analytic coefficient integration on a complex contour before constructing cumulants. It never enumerates work outcomes. The reference uses radius.18 and64nodes; radius.24 with96nodes changes answers by at most1.15e−12. Off-grid discrepancies are at most2.33e−12; all hidden reference values agree within4.28e−12. A cold evaluation of all hidden inputs takes.028seconds.

Forty-eight off-grid tests check the first-three identity, the fourth commutator correction, unitarity and common-energy-shift invariance. Their largest identity error is1.69e−14. Zero-drive and zero-duration outputs vanish to reference roundoff. Public input ranges are checked. Every scored fourth cumulant remains negative in both constructions, and the smallest true magnitude over a201-point full-scale scan is2.2563. Separation therefore does not depend on a sign mismatch or a nearly zero denominator. Minimum group errors over that scan are.1724,.2247,.2151.

There are288 independent calibration records:24 repeats of six pulse sequences, each reporting variance and third cumulant. Sigma=.0003 is fixed independently of parameter or response. True energy_scale=1.06; calibration seed860041 and noise seed860047. Both controls give the same calibration predictions for all scales. On a201-point scan of[.8,1.2], each calibration variance derivative is positive, with minimum.22726; the smallest absolute third cumulant is.0080477. Five full objective profiles, including both endpoints, decrease to a unique minimum and then increase. Noiseless parameter recovery is better than1.92e−10 relative.

The separate prototype additionally checks256 noisy fits with true scales spanning the entire interval rather than only the nominal value. Its maximum parameter error is.0233%, maximum reduced chi-square1.259 and maximum hidden prediction error.0897%.

## Controls and limits

Grading remains reduced calibration chi-square<1.5, parameter error<3%, and hidden normalized group RMS error<.04. The actual data fit gives scale1.0600209411 and chi-square.778213. Oracle hidden errors are at most.00005590; completed-shortcut fourth-order errors are.401091,.293643,.307387. The off-calibration lower-cumulant anchor passes both models.

All256 nominal noise realizations pass calibration, parameter recovery and all oracle predictions; the shortcut fails every fourth-order group in every realization. Maximum chi-square is1.19204, relative parameter error.00016910, oracle hidden error.00047845, and minimum shortcut error.29287. Local isolated pytest gives oracle8/8 and shortcut5passes/3intended physics failures in under one second each. Agent/verifier limits remain600/60seconds. Scientific validation does not establish Luna difficulty; no model evaluation has run on this staged revision.

Evidence: results/quantum-work-r2-validation.json, results/quantum-work-mixed-r2-prototype.json and results/quantum-work-r2-source-provenance.json under the staged root. The completed shortcut is scripts/quantum_work_baseline.py. Reproduce from the repository root with `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/quantum-work-r2/scripts/validate_quantum_work.py`. Add `--generate` only to regenerate both calibration copies deliberately. Paths are relative to the validator's root and remain valid when promoted.

Independent physics peer approval is in results/physics-review-quantum-work-r2.json under the staged root. Its24 off-grid checks give first-three equivalence within9.6e−16, fourth commutator identity within2.1e−15 and ODE/counting reference agreement within1.43e−12.

## Evaluated outcome and archive

After revision1 was archived, revision2 was promoted unchanged and evaluated with the neutral instruction. The initial trial quantum-work__E5koVZi passed; both conditional follow-ups quantum-work__au5xPa2 and quantum-work__szLrJXR also passed. All three final implementations replace the single energy-change observable with the measured joint initial/final energy-record distribution and compute the fourth cumulant correctly. Each fit is1.0600209411; the largest hidden group error is below5.59e−5. All native verifiers report8/8 passes without exceptions. Some explanatory comments overstate which lower cumulants differ, but all three physical prediction models are correct.

The honest aggregate is3/3 genuine physical successes, including the initial trial. Oracle and shortcut controls gave1/1 and0/1 respectively. The task misses the requested difficulty target and is archived under archives/quantum-work-r2 with full source, staging evidence, reports and all native trials. The earlier revision1 result2/3 and its implementation failure remain separately preserved in archives/quantum-work-r1. No scientific or agent-visible file changed after the revision2 freeze; only this author outcome paragraph was appended.
