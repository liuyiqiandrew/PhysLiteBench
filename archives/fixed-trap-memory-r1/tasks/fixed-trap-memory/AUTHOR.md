# Fixed-trap memory, revision 1

The public apparatus is a finite physical hopping channel with independently fabricated site types. A site keeps its escape rate for the complete passage experiment. A fresh channel is prepared for each repetition, and the reported variance pools both fabrication and thermal trajectories. The supplied implementation is the exact stochastic model obtained by resampling the destination site's rate class at every arrival. Both models have the same full mean passage-time calibration. Their distinction is the persistence of a physical trap when the particle returns to it.

No agent difficulty is claimed before evaluation. This stage has completed scientific checks and local oracle/shortcut controls. The neutral instruction permits replacement of all prediction code and helpers. The fitted scale starts at None; the completed shortcut is separate from the unfinished public fit.

## Derivation and physical closure

Transient sites are0,...,N-1; siteN absorbs. At0 the sole departure to1 has the site's total escape rate, without rejected outward attempts. At an interior site, departure chooses right/left with probabilities p and1-p independent of the rate class. A typeA site has residence mean1/gamma and typeB has c/gamma. Each actual visit uses an independent exponential holding clock conditional on this fixed specimen. The preparation at0 is independent of the fabricated types. These facts determine the first-passage ensemble without requiring a thermodynamic-limit or long-time approximation.

Let P be the transient jump matrix, G=(I-P)^(-1), M_i the number of visits to site i before absorption, and M=sum_i M_i. Starting at0,

    E[M] = (G 1)_0,
    E[M^2] = ((2G-I) G 1)_0,
    E[sum_i M_i(M_i-1)] = 2 sum_i G_0i (G_ii-1).

The embedded path is independent of the escape-rate disorder. Write mu=E[tau] and mu2=E[tau^2] for the distribution of the site's mean residence time. Both models give E[T]=mu E[M]. The renewal model's raw second moment is

    E[T^2]_renewal = mu^2 E[M^2] + (2mu2-mu^2) E[M].

For a fixed specimen, repeated visits share the same random residence mean. Conditional on the embedded path, averaging independent exponential clocks and specimen disorder gives variance `mu2 sum_i M_i + (mu2-mu^2) sum_i M_i^2`. The conditional mean is `mu sum_i M_i`. The resulting physical second moment exceeds the renewal value by

    (mu2-mu^2) E[sum_i M_i(M_i-1)].

The first moments coincide, so this also equals the difference of centered variances. It is not a change to the waiting-time distribution for a single visit or a variance-subtraction convention.

The source's2N-state continuous-time generator tracks the current site and rate class. At a jump, its destination class is independently sampled with the fabrication probabilities. It preserves the current rate between jumps, including the full mixture of exponential holding times, and samples the initial class correctly. Thus it is a positive, absorbing Markov process with correct finite geometry and bias. Every input participates. The source solves both hitting-time moment equations correctly; the approximation is a physically different renewal of the environment.

The independent reference enumerates the2^N binary specimens, builds each specimen's explicit physical backward generator, and solves `A m1=1` and `A m2=2m1`. It averages the raw first and second moments over specimen probabilities before subtracting the square of the pooled mean. It contains no return-count correction or Green-matrix identity. The prototype also separately checks the law of total variance, distinguishing the mean thermal conditional variance from the variance of specimen-specific means.

## Calibration, identifiability and validation

There are22 distinct calibration settings:18 means spanning size, directional bias, residence ratio and type probability, each measured12 times; and four homogeneous-channel variances, each measured18 times. This gives288 independent records. Both physical controls agree on all means for arbitrary disorder and on all statistics for zero disorder. Calibration therefore exercises both requested statistics without exposing repeated-site disorder correlations.

Every mean is a positive known coefficient times gamma^-1; every calibrated variance is a positive known coefficient times gamma^-2. Gamma is globally identified over[.6,1.4]. At33 parameter values, noiseless source fits recover gamma within2.76e-8; minimum nominal absolute calibration sensitivity is4.5451. Numerical source/oracle calibration differences are at most1.14e-13, and independent-reference bias is below5.69e-13 instrument sigma.

The instrument uncertainty is fixed at.1 t0 for means and.5 t0 squared for variances. It depends only on the physical units of the requested statistic and does not encode the unknown scale or the clean response. The true scale1.07, data seed192011 and noise-validation seed192013 are private. Both controls fit1.069990945 with reduced chi-square.963887.

The three diagnostic groups have positive physical variances between425 and13748 in t0 squared. The nominal source normalized RMS errors are.437839,.431017,.538864, while oracle errors are1.70e-5. The exact-anchor group combines disordered means, directed motion with no revisits, and a homogeneous variance; both controls pass it. Group RMS normalization does not divide by a near-zero individual prediction.

`results/fixed-trap-memory-r1-validation.json` records:

- All256 noise draws pass calibration and parameter checks for both controls. All physical hidden predictions pass. The maximum physical hidden error is.0004801, and the minimum source diagnostic error is.430844 against the unchanged.04 prediction gate.
-96 supported corners and32 random cases agree with independent specimen enumeration within1.88e-13 relative. Moments are positive and finite. The physical variance is at least the renewal variance in the checked domain.
- Directed no-return motion, zero contrast and pure-type populations make both models agree. Scaling all escape rates rescales mean and variance by the expected first and second powers.
-33 full-range parameter recoveries preserve the diagnostic gaps, whose minimum is.431026. Scale dependence is exact, so the fractional physical distinction is independent of the fitted rate scale.

The prototype preserves12 diagnostic cases,54 limit checks and33 full-range fits. Its smallest individual variance discrepancy is13.09%, so the group errors are not produced by an isolated vanishing signal. The source is independently checked both as a2N-state generator and through the renewal moment identity.

The local isolated verifier reports oracle7/7 in.61seconds, and shortcut4passes/3intended failures in.58seconds. Scientific validation took3.50seconds. Standard600-second agent and60-second verifier limits remain. Grading checks numeric calibration, parameter recovery, exact output shape/finiteness and physical predictions; it places no restriction on algorithms or implementation structure.

## Relation to earlier work

The prototype archive audit scanned255 AUTHOR paths and read the four closest families. Single-file memory concerns a conserved particle rank and preparation-dependent joint asymptotic covariance, not fixed spatial escape-rate disorder. Exclusion current concerns conservation between simultaneous counters; terminal current noise concerns the location of the electrical readout; finite-band reservoir concerns unitary quench populations. They share broad memory or measurement themes but do not contain this return-sensitive disorder mechanism. No previous result is discarded or reclassified.

Primary background is Akimoto and Saito, [Exact Results for First-Passage-Time Statistics in Biased Quenched Trap Models](https://arxiv.org/abs/1901.00624), Phys.Rev.E99,052127(2019), and Luo and Tang, [Sample-dependent first-passage time distribution in a disordered medium](https://arxiv.org/abs/1507.07409), Phys.Rev.E92,042137(2015). These studies distinguish persistent random traps from renewal CTRWs and discuss passage-time fluctuations. Their geometries and disorder laws differ from this task. The finite reflecting-channel, binary-specimen and pooled-variance identities here are derived directly and verified by enumeration rather than borrowed from an asymptotic result.

Reproduce author checks with:

    uv run --no-project --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/fixed-trap-memory-r1/scripts/validate_fixed_trap_memory.py

Use `--generate` only for deliberate replacement of both calibration copies. The separate local report is `results/fixed-trap-memory-r1-local-controls.json`. Root owns frozen model trials. Retaining the renewal closure would be a physical-model failure; constructing the fixed-specimen process but mishandling its raw moments or matrix convention would require a separate implementation classification.
