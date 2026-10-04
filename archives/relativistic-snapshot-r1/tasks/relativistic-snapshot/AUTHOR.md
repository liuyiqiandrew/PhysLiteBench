# Relativistic snapshot, revision 1

Author-only. This candidate tests which particles belong to a relativistic spatial snapshot. The source computes the canonical rest distribution, mass shell and both Lorentz transformations correctly. It averages a sample selected on a rest-frame simultaneous slice. The apparatus selects particles in a fixed laboratory volume at laboratory simultaneity, then expresses those recorded four-momenta in a separate analysis frame. These are different sampling ensembles. No empirical difficulty is claimed before frozen model trials.

## Apparatus and calibration

The ideal homogeneous gas is classical, noninteracting and number conserving in three dimensions. The public rest measure is explicitly canonical d^3p, temperature is explicitly comoving, and measured energy includes rest energy. Spatial selection is at one laboratory time, with equal efficiency per particle and no boundary effects. The analysis frame is only postprocessing of the selected records. Thus neither a relativistic temperature convention nor a detector flux weighting is left unspecified.

The sole unknown is mass m in [.8,1.2], with private truth1.07. Calibration uses four temperatures and four nonzero gas velocities with analysis velocity equal to gas velocity, plus two temperatures at zero gas velocity with three distinct analysis velocities. These22 controls are each repeated12 times (264 independent records). Both closures agree exactly throughout both calibration families for every allowed mass; neither known velocity control is an unused placeholder. Hidden observations combine unequal gas and analysis velocities. The public description does not call attention to these calibration identities.

Measurement sigma=.004 E0 is a fixed instrument uncertainty independent of mass or the clean response. Seeds are161051 for checked-in data and161053 for256 independent calibration-noise realizations. Generation requires `--generate` and writes identical public and private copies. The standard normalized RMS hidden limit is.04; parameter relative error is.03 and reduced calibration chi-square is below1.5.

## Derivation and completed control

Write b for gas velocity, a for analysis velocity, gamma(v)=(1-v^2)^(-1/2), E=sqrt(m^2+p^2) and e0=<E> in the gas rest frame. The exact ideal-gas identities are

    e0 = m K1(m/T)/K2(m/T) + 3T,
    <p_x^2/E> = T.

The completed source integrates the full positive canonical rest momentum density using radial rapidity and angular quadrature. It transforms each momentum into the laboratory and then into the analysis frame, returning the ordinary rest-sample average

    source = gamma(a)*gamma(b)*(1-a*b)*e0.

This is the mathematically correct energy average for a fixed population of labels sampled on a simultaneous rest slice. It is a coherent alternative physical sampling model, not a defective Lorentz transformation, missing factor, failed numerical solver or fitting error.

On a simultaneous laboratory slice the invariant scalar phase density is exp[-gamma(b)*(E_lab-b*p_x_lab)/T], integrated with ordinary laboratory d^3p_lab. Transforming that integral back to rest momentum introduces the normalized particle-selection weight

    d^3p_lab/[gamma(b)*d^3p] = 1+b*p_x/E.

The oracle retains the complete source integration and changes this selection measure only. Equivalently, the number current and stress-energy tensor give

    <E_lab> = gamma(b)*(e0+b^2*T),
    <p_x_lab> = gamma(b)*b*(e0+T),
    correct = gamma(a)*gamma(b)*[(1-a*b)*e0+b*(b-a)*T].

The extra term vanishes at a=b or b=0. In particular a nonzero boosted calibration with a=b still returns e0. Calibration therefore does not determine the missing sampling physics.

Identifiability is global, not only a sampled optimizer result. Differentiating the canonical expectation gives

    de0/dm = m*<1/E> + (m/T)*(<E>*<1/E>-1) > 0.

Cauchy-Schwarz makes the second term nonnegative and the first strictly positive. Both calibration families multiply e0 by a positive known factor. Hence distinct masses cannot have identical noiseless calibration predictions. The sampled derivative lower bound is.61899, and the Fisher information at the true mass is1.39636e7 in the stated noise units.

## Independent reference and validation

The verifier integrates directly in laboratory momentum coordinates, using the scalar phase density above and the measured energy `(E_lab-a*p_x_lab)/sqrt(1-a^2)`. It neither copies the oracle's rest-coordinate selection weight nor substitutes the final tensor formula. Its radial rapidity and angular orders are192 and128; refinement to288 and192 changes corner results by at most1.08e-14 relative. Oracle/reference corner disagreement is at most8.38e-14. The tensor/current result is a third analytic check. An adaptive radial integral independently checks rest equipartition and energy. Reversing both velocities leaves the output unchanged, all energies stay positive and above rest mass, and nonrelativistic/ultrarelativistic limits converge correctly.

The prototype also constructs a microscopic reference: draw600000 uniformly positioned rest-slice particles with thermal momenta, intersect each inertial worldline with laboratory time zero, select a fixed lab interval, and average the transformed recorded energy. At the off-grid control the selected mean differs from the exact result by0.70 standard errors. Averaging all rest labels instead gives the shortcut. This Monte Carlo is supplementary evidence; the deterministic verifier does not depend on its sampling error.

The checked-in mass fit is1.0706135034 with reduced chi-square.867924. Both controls agree on calibration to4.45e-16; numerical calibration bias is below4.5e-12 sigma. Nominal shortcut group errors are.169140,.182777,.149812. All256 calibration-noise realizations pass both calibration and parameter checks, and every oracle hidden group passes with worst error.000294. The smallest shortcut diagnostic error is.149763. A41-point mass sweep recovers the noiseless parameter to3.14e-8, checks exact calibration equivalence, and retains at least.14519 diagnostic separation across the full mass interval. Matched-frame and zero-gas-velocity hidden anchors pass both controls.

The actual local harness passes7/7 for the oracle in.65s. The completed source passes calibration, parameter and anchor checks and fails exactly three diagnostic groups (4pass/3fail) in.52s. Local pytest emits a harmless cache-directory warning because its common root resolves outside the writable directory; test outcomes and calculations are unaffected. The standard600-second agent and60-second verifier limits are unchanged.

## Prior-family audit and provenance

The prototype audit scans251 archived/current/staged AUTHOR files and finds no prior relativistic gas snapshot apparatus. The archived effusive-beam task also concerns sampling, but its missing physics is aperture-crossing flux weighting in nonrelativistic kinetics. Circular-detector concerns noninertial quantum vacuum response. Single-file-memory concerns preparation-dependent two-time tracer statistics. This candidate shares general ensemble interpretation with those topics while testing a different physical hypersurface and a different calibration nullspace. No model outcome from those families is repurposed as evidence for this candidate.

The neutral instruction allows replacing every helper and prediction function. Docker copies only public README, starter, tests and calibration. No answer formula, sampling-weight correction, prescribed algorithm or class restriction is supplied publicly beyond the required API. The mass starts as None. Hidden verification checks numeric outputs, shape, finiteness, calibration and mass recovery, with no implementation inspection.

Primary sources: [Chacon-Acosta, Dagdug and Morales-Tecotl, Phys. Rev. E81,021126 (2010)](https://arxiv.org/abs/0910.1625) develops covariant Juttner statistical moments; [Dunkel and Hanggi, Physics Reports471,1–73 (2009), Appendix C](https://math.mit.edu/~dunkel/Papers/2009DuHa_PhysRep.pdf) gives observer-simultaneous energy and momentum; [Dunkel, Hanggi and Hilbert, Nature Physics5,741–747 (2009)](https://arxiv.org/abs/0902.4651) discusses the physical role of measurement hypersurfaces. Our formulas and independent integrations are derived above for the exact stated ideal model.

Reproduce deterministic validation from the repository root:

    uv run --no-project --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/relativistic-snapshot-r1/scripts/validate_relativistic_snapshot.py

Reports are `results/relativistic-snapshot-r1-validation.json`, `results/relativistic-snapshot-r1-local-controls.json` and `results/relativistic-snapshot-r1-source-provenance.json` within the stage. The prototype, its report and archive audit are preserved under `prototype/`. Independent final peer review is attached separately. No model-agent trial has been launched by this package's author.
