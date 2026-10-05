# Cellular tracer dispersion r1

This is an unevaluated replacement candidate, resumed only after the documented rotating-reservoir family review found no justified immediate r3. No model or Docker runs have occurred. No result from another family is replaced by these author checks. The task has the unchanged neutral instruction, permits full implementation replacement, and uses the standard 600/60-second limits, 3% parameter, reduced-chi-square1.5 and 4% prediction gates.

## Apparatus and supplied approximation

The measured quantity is the long-time variance rate of unwrapped laboratory displacement for every independent tracer. The prescribed field is steady, periodic and divergence-free. Initial phase is uniform, and there is no selection, absorbing boundary or conditioning. The complete trajectory law is public. Molecular diffusivity D in [.8,1.2] is the only unknown; the detector direction and all four first/second harmonic amplitudes are known.

The completed source is the self-consistent Gaussian sampling model developed in the preserved prototype. Define A=(a²+c²/4)/2 and B=(b²+d²/4)/2. An auxiliary phase Y undergoes independent Brownian diffusion with tensor diag(Kx,Ky) on the cell. A second process integrates the entire velocity u(Y), with independent molecular displacement noise of diffusivity D. Its asymptotic diffusion is

    Kx = D + A/Ky,     Ky = D + B/Kx.

The source solves the positive fixed point exactly. Subtraction gives Kx−Ky=(A−B)/D; the remaining quadratic determines Ky. The zero-amplitude limits are handled by the same stable root, and Kx,Ky≥D throughout the public domain. This is a definite positive stochastic approximation, using every harmonic and coupling the assumed sampling rates. It substitutes Gaussian cell-phase sampling for the actual advected trajectory. It is not the original weak prototype that used bare D for both sampling rates, nor does it omit a velocity component or provide an unfinished forward solver. Only fitting is left unfinished in the public file; the separate baseline completes fitting without changing any forward helper.

For a one-leg shear, the physical transverse coordinate is molecular Brownian motion. The approximation then gives the exact nonzero asymptotic enhancement. This equivalence concerns the measured diffusion coefficients; independent auxiliary/output noises do not make the two full tracer processes identical.

Both tensors are diagonal for the specified odd separable fields. Reflection x→−x maps the velocity to its full reversal and flips the tensor's off-diagonal entry. Reversing a steady incompressible drift exchanges the generator with its adjoint; the symmetric long-time diffusion tensor is unchanged. Together these facts force the off-diagonal entry to zero. The oracle and reference retain the full tensor rather than assume that cancellation numerically. Across152 domain cases the largest off-diagonal entry is3.62e−15. Detector angles still vary the two generally unequal diagonal contributions.

## Physical oracle and independent reference

The oracle uses the full backward generator L=DΔ+u·∇ and mean-zero correctors −Lχ_i=u_i. Its effective tensor is equivalently

    K_ij = D delta_ij + sym <u_i chi_j>
         = D delta_ij + D <grad chi_i dot grad chi_j>.

The implemented Fourier solve uses the second form, which exposes positivity. Every advection harmonic and derivative is present. The uniform phase measure follows from incompressibility. Positive D makes the cell equation coercive on zero-mean functions, so the full stated control box is well posed. No bound on a Peclet number beyond the public controls is required.

The private reference independently constructs a spatial continuous-time nearest-neighbor process. Its rates are D/h²±u_i/(2h), each jump carrying its actual signed displacement±h even across a cell boundary. First and second derivatives of the displacement-counting generator yield the variance rate. It imports no spectral corrector or oracle coefficient. Uniform stationarity follows because u_x is independent of x and u_y of y. The finite grid is an approximation to the stated continuum experiment, not a change to that experiment.

The rate positivity guarantee holds on the entire public box. With |u_i|≤8.8 and D≥.8, every rate at n≥48 is at least h⁻²(.8−4.4h)>0, because h≤π/24<2/11. At n48 this conservative lower bound is13.075. Refinement improves it. The reference uses n48/96 Richardson extrapolation, with independent n96/192 refinement in the author checks.

Every one of the24 calibration settings and13 hidden settings is compared with both spatial resolutions. Maximum oracle/reference discrepancy is2.511e−5; maximum refinement change is2.352e−5. Calibration numerical bias is at most.000604 instrument sigma. The independent grid error is far below the 4% prediction gate. Sixty-four further domain cases, including all32 corners and32 interior fields, have the same independent refinement. Their maximum error is3.064e−5, decreasing to1.926e−6 at the finer extrapolation. All152 domain cases compare spectral cutoffs20 and28; the maximum change is2.843e−14.

Three additional finite-grid calculations propagate the first two displacement moments from uniform phase to times8/D and16/D. Their late-interval slopes agree with the stationary counting moments within1.04e−7 relative. A separate principal counting-eigenvalue curvature agrees within5.83e−8. These are deterministic finite-state dynamics with explicit grid and long-time checks, not Monte Carlo. No particle-sampling uncertainty is hidden in the report.

## Calibration, margins and domain qualifications

There are24 nonzero shear settings: both orientations, four amplitudes and three detector angles, repeated independently12 times for288 records. Sigma is the fixed instrument value.003, independent of D and noiseless response. Generation uses the independent spatial reference and seed193081. Public and private data copies are byte-identical. The private D is.93; public constructors use None.

For x-shear at a fixed detector angle, K_e=D+C/D with C=cos²(angle)(a²+c²/4)/2. Two included different amplitudes give a difference(C2−C1)/D, strictly decreasing for positive D. This proves global parameter identification, even though one individual D+C/D curve can be nonmonotone. Both orientations enrich the calibration. Across41 noiseless D values including endpoints, both fits recover the parameter within7.31e−9. Exact continuum calibration equivalence is within8.89e−16.

On the frozen noisy data both fits give D=.9300860867 and reduced chi-square.935216. Oracle hidden-group errors are at most5.36e−5. The source errors are.374863,.370327,.375478 in the three diagnostic groups; the zero/shear/weak anchor group passes at5.29e−5. These are ordinary positive spreading coefficients, not near-zero targets.

All256 fixed-seed noise trials pass calibration and parameter gates for both models. Every oracle prediction and source anchor passes; every source diagnostic group fails. The largest parameter error is.000404, largest oracle prediction error.000235, smallest source diagnostic error.370263, and largest chi-square1.22675. All individual rows are retained.

The41-point full-D sweep of the fixed diagnostic groups has a smallest sampled gap.243835. This is not a uniform discrepancy claim over the full amplitude box. Zero, shear and weak flows are valid public inputs and deliberately do not strongly distinguish the two models. The152 domain rows include all32 signed corners,96 fixed-seed interior samples and24 zero/weak/intermediate fields. Some source differences are negative;102 rows underpredict at least one axis. Those results and the original weak amplitude4 findings remain preserved. No general upper bound is attributed to the self-consistent approximation. Positivity and well-posedness have analytic full-box arguments; numerical error and discrepancy statements retain their actual sampled scope.

Actual local controls finish in less than one second each: oracle7/7; shortcut4 passed and only the3 diagnostic predictions failed. The first validator execution completed all scientific checks and these correct controls, then stopped on an author bookkeeping assertion expecting8 rather than7 tests. Its complete report, actual control outputs and executed validator bytes are preserved. Only the expected test-summary counts were corrected; the completed report was finalized by a separate stdlib audit without repeating science. No data, grader or physical source changed because of this assertion.

## Preservation, overlap and source access

The bounded prototype is byte-exact under prototype/, including its original author manifest, Materials peer, initial Brownian sampler, stronger fixed-point source, all weak/strong outcomes and recorded source access. All24 files from the stopped partial package are byte-exact under development/stopped-partial-checkpoint/. The provisional outline's second-harmonic bound±.7 conflicted with the.75 shear calibration. The package explicitly uses±.8 and validates that enlarged domain; no earlier file or result was changed. Source and report provenance are recorded separately after final authoring.

The closest actual archive comparisons remain those in prototype/overlap-and-sources.json. Tube-dispersion passed3/3 plain and3/3 hinted with a source that discarded all shear enhancement. Driven-diffusion passed3/3 in both conditions and used an inappropriate equilibrium response relation. Survivor-transport r1 passed2/3 and concerned conditioning on survival. The present source retains exact nonzero shear coefficients and samples all velocity harmonics, but replaces advected phase residence with an auxiliary Gaussian law. The shared tracer/homogenization theme is acknowledged; this is not a claim of unrelated mathematics. It does not reuse the single-file fixed-preparation covariance or a Langevin elimination limit.

Primary background is Majda and Kramer, *Simplified models for turbulent diffusion*, Physics Reports314(1999), institutional PDF https://mhd.ens.fr/IHP09/Young/Biblio/MajdaKramer.pdf. The preserved access record covers sections2.1 and2.2.3 and nearby finite-time discussion, not the entire337-page paper. Materials independently read the cell formulas and cellular-flow discussion. This supports homogenization and the role of streamline sampling, not a claim that this exact auxiliary fixed-point closure is derived in that paper. The Phythian–Curtis publisher abstract is background only; its full paper was not accessed. Failed NYU mirror URLs remain recorded.

Empirical difficulty is unknown. A solver may directly derive the correct cell generator from the public trajectory law. If that physics is chosen but Fourier signs or numerical convergence fail, the trial must be classified as mathematical or numerical rather than retention of the intended physical approximation. No evaluated outcome is claimed here.

Reproduce the full author check with the staged validator using pinned NumPy2.3.3, SciPy1.16.3 and pytest8.4.2. Run `scripts/validate_cellular_tracer_dispersion.py` from this stage; `--generate` deliberately regenerates both data copies with the fixed seed. The validator includes isolated local controls, not Docker or model calls.
