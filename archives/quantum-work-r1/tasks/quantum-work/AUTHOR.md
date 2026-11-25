# Quantum work, revision 1

The task tests the statistics of an operational energy-record protocol. The supplied predictor computes the exact Gibbs state and correctly ordered unitary for both finite pulses. It forms the Hermitian energy-change operator U†H0U-H0 and takes its spectral moments. This is a consistent single-observable construction, but its third cumulant is generally different from the difference between the two energies actually recorded by the apparatus.

The public README specifies fresh Gibbs preparation, an initial ideal projective energy measurement, the full finite pulse sequence, a final projective measurement, and subtraction of the recorded values. There are no intermediate measurements, additional switching impulses, or postselection. It supplies no joint-probability formula or correction. The exact neutral instruction allows arbitrary implementation changes. The hint is outside the public image. All sigma values are fixed instrument uncertainties.

## Physics

Let A=H0, B=U†H0U, and rho=exp(-A/T)/Z. The correct joint probability of initial energy E_n and final energy E_m is p_n*|U_mn|². The recorded value is E_m-E_n. Because rho commutes with A, its mean and variance exactly match those of B-A for every allowed pulse sequence, including noncommuting pulses. Thus calibration validates substantial, correct dynamics and is exactly blind to the measurement interpretation.

The third-cumulant difference is

    kappa3_records-kappa3_operator
      = Tr[rho*(B*A*B-B²*A)]
      = sum_(i<j) (p_i-p_j)*(E_j-E_i)*|B_ij|².

The correction is nonnegative for Gibbs weights ordered by energy. The first two raw moments coincide, so the same correction applies to the third raw moment and third cumulant. This distinction is separate from bosonic transfer-noise, equilibrium bath heat, and detector spectral-weight tasks: here the exact unitary and initial state are already correct, while the two measurements cannot be replaced by one spectral observable.

Primary source: Talkner, Lutz and Hänggi, [Fluctuation theorems: Work is not an observable](https://doi.org/10.1103/PhysRevE.75.050102), Physical Review E 75,050102(R) (2007), [author manuscript](https://arxiv.org/abs/cond-mat/0703189). The task uses its operational energy-measurement setting; no formula from the paper appears in the public instructions.

## Independent verification

The oracle uses joint initial/final measurement probabilities and matrix-exponential pulse propagation. The private reference independently integrates the Schrödinger equation for the three basis columns, forms the analytic generating function Tr[exp(zH0)*U*exp(-zH0)*rho*U†], and extracts moments by a 32-point Cauchy contour before converting to cumulants. It does not enumerate joint outcomes. The contour radius is .08; changing radius to .12 and resolution to 48 points changes results by at most 3.07e-13. Off-grid oracle/reference differences are below 3.64e-13; cold evaluation of all private inputs takes .017 seconds.

Forty-eight independent control samples check unitarity, first-two-cumulant equivalence, the third-order commutator and thermal-pair identity, the cyclic Jarzynski equality, nonnegative mean work, and invariance under a common energy shift. Zero-drive work vanishes. Independent peer review adds 32 controls and checks the same physical identity without relying on the oracle outcome sum. The supplied operator has positive variance and correct first two moments; its third error is a measurement-model error rather than a matrix or centering defect.

The initial prototype included a duration sweep whose third cumulant approached zero. Those points were excluded before task generation. The frozen duration sweep uses phase .9 and substantial positive skew. Across all frozen third-order inputs, the smallest absolute third cumulant is .11187 and the smallest standardized skew is .50066. The task does not rely on cancellation or a tightened grading tolerance. The initial screen is preserved in results/quantum-work-prototype-initial.json.

## Calibration and controls

There are 144 records: six noncommuting pulse sequences, three temperatures, mean and variance readouts, and four independent repeats. The common energy scale is 1.06, fixed sigma is .0003, calibration seed 860031 and noise seed 860037. Public and private data copies are identical. A 121-point objective profile covers the full [.8,1.2] parameter interval at three separated true scales, including .801 and 1.199. Each profile has one minimum, with exact noiseless recovery better than 4.04e-9 relative.

The standard limits are reduced calibration chi-square<1.5, parameter error<3%, and hidden group normalized RMS error<.04. Nominal fitted scale is 1.059952204282482 and reduced chi-square 1.102926. Oracle hidden errors are at most .000139. Shortcut third-order errors are .992951,1.147061 and1.062491; its off-calibration mean/variance anchor also passes.

All 256 noise trials pass calibration and parameter checks, every oracle passes all hidden groups, and every shortcut fails the three third-order groups. Worst oracle error is .000747; minimum shortcut error is .992935. Local isolated pytest: oracle 8/8; shortcut 5 passes and 3 intended hidden failures. Docker controls and model screens are run separately after freeze; no Luna difficulty result is claimed by these scientific checks.

Reports: [validation](../../results/quantum-work-validation.json), [local controls](../../results/quantum-work-local-controls.json), [source provenance](../../results/quantum-work-source-provenance.json), [physics peer](../../results/constraint-hardening-physics-review.json), [independent peer checks](../../results/physics-review-quantum-work-checks.json), and [root peer](../../results/root-hardening-physics-review.json).

Reproduce with `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_quantum_work.py`. Use `--generate` only when intentionally regenerating both frozen calibration copies.
