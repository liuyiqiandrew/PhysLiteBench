# Cellular tracer: bounded feasibility

Two scientific executions completed successfully. There is no task package, calibration file, fitted truth, Docker run or model evaluation. The initial Brownian-sampling source and its full report are preserved in check.py/report.json. The stronger source refinement and every compared case are in closure_followup.py/closure-followup-report.json. No failed or weak parameter case was removed.

## Physical approximation and calibration

The exact point-tracer law is dX=u(X)dt+sqrt(2D)dW on an unbounded plane with periodic velocity. The initial phase is uniform. The long-time unwrapped covariance defines a positive effective diffusion tensor. The liquid velocity is prescribed and incompressible; there is no additional fluid solver, tracer inertia or absorption.

The initial source samples all harmonics of u on a molecular Brownian trajectory rather than the transported position. It exactly retains the entire shear velocity covariance, giving Kx=D+a0/D and Ky=D+b0/D, where a0=(a²+c²/4)/2 and b0=(b²+d²/4)/2. This is the effective diffusivity of a definite one-way sampled process; it is much stronger than replacing the flow by its mean, but its very large strong-flow errors make it a weaker proposed benchmark source.

The proposed refinement instead lets the auxiliary sampling phase have its own asymptotic diffusion tensor diag(Kx,Ky). Integrating its full velocity correlation and adding independent molecular displacement gives

    Kx = D + a0/Ky,
    Ky = D + b0/Kx.

These equations have a unique positive solution with Kx,Ky>=D. Their difference gives Kx−Ky=(a0−b0)/D; one positive quadratic root determines both. Equivalently, a Brownian auxiliary phase with that tensor samples the complete velocity field, its time integral is added to an independent bare Brownian displacement, and the output tensor equals the assumed sampling tensor. Thus the source is a completed self-consistent stochastic closure, not a missing fit, incorrect derivative or arbitrary removal of a velocity component. It includes feedback between both flow components statistically. It treats diffusion valid over long distances as the law of phase sampling at the velocity-cell scale, losing deterministic streamline memory.

If b=d=0, the actual transverse coordinate is precisely molecular Brownian motion. Both sources then give the exact longitudinal Taylor enhancement for every a,c,D; the orthogonal case is equivalent. Calibration can use several nonzero shears in both directions. With c=.25a, each longitudinal reading is D+C(a)/D. The difference at two amplitudes is [C(a2)−C(a1)]/D, strictly identifying D>0 even if an individual curve has two possible parameters. Thirty-three noiseless recoveries spanning [.8,1.2] differ by at most2.22e−16. No measured covariance or diffusion parameter is exposed as an input.

## Independent physical calculation

The spectral calculation uses the backward generator L=D Delta+u dot grad. Its zero-mean correctors obey −L chi_i=u_i. The tensor D I+sym< u_i chi_j > also equals D I+D<grad chi_i dot grad chi_j>; the latter identity checks signs and positivity. Fourier multiplication by the full velocity is retained. Cutoffs18 and24 agree within5.33e−15 on the36 original cases.

The reference independently constructs a positive continuous-time jump process on a periodic spatial grid. Its nearest-neighbor rates are D/h²±u_i/(2h). Every rate is positive on the reference cases. Each jump carries its actual signed displacement ±h, including wraps. Derivatives of this counting generator give the stationary variance rate. No Fourier velocity coefficients or spectral corrector are imported. Uniform stationary probability and row balance are checked. Grids48 and96 with Richardson extrapolation agree with the continuum result to7.04e−6 relative across12 selected cases. Additional192 grids reduce checked errors to4.47e−7. This is convergence evidence, not exact finite-grid equivalence.

A further32-grid tilted principal-eigenvalue calculation agrees with its own stationary counting moments. Independent finite-time propagation of first and second displacement moments gives late interval slopes within1.04e−7 of the same jump process's stationary limit on three cases. Times8/D and16/D are recorded, so the finite-time intercept is not mistaken for diffusivity. These are exact finite-state dynamics with discretization/time-limit checks, not particle Monte Carlo; there is no unreported sampling uncertainty or claim of a microscopic continuum trajectory simulation.

## Recorded margins and limitations

The first report keeps36 fields/parameters: zero flow, equal amplitudes .1,.5,1,2,4,6,8, unequal amplitudes and additional harmonics, each at D=.8,1,1.2. The followup retains their signed self-consistent errors. At weak and intermediate amplitudes the refined approximation can underestimate diffusion and can be within4%; those cases remain visible. In particular, amplitude4 has a weakest sampled difference only4.50%, so it should not support a robust graded margin without further work.

The source-refinement sweep adds168 cases:21 diffusivities, equal amplitudes4,5,6,8, with and without second harmonics. For amplitudes5–8 the minimum sampled relative error is11.54%; true readings are3.052–4.733 and source readings3.947–6.302 in L0²/t0. At a=b=6,D1 the physical diffusivity is3.73908 and the source is4.77200. Neither is zero or negative. These are sampled bounds, not a continuous-domain proof or a noise study. Some unequal fields have much larger directional errors, but no selection of those is needed for this initial margin.

Fixed-point residuals are at most1.78e−15. Thirty zero/shear anchors agree with the exact physical solver within8.89e−16. Physical tensors are positive; flow reversal and axis exchange pass. The simple initial source is an upper bound on checked cases, while the refined source has no such general bound and is not claimed to. The first and second scientific runs took4.72s and2.47s. No runtime or mathematical difficulty is being used as a proposed failure mechanism.

A final package would still need full proposed-domain checks, fixed instrument uncertainty, 256 noisy fits, an independently checked source implementation, local controls and a new public/input peer. The provisional input outline gives all physical facts and units without prescribing the cell equation or exposing the approximation. It does not omit cross advection from the apparatus.

## Overlap, sources and likely difficulty

Actual archived source and outcome records were read. Tube-dispersion scored3/3 with a source that keeps only bare molecular variance; this proposal retains and calibrates the full nonzero shear enhancement. Driven-diffusion scored3/3 and concerns a nonequilibrium Einstein relation. Survivor transport scored2/3 and uses selected absorbing paths, whereas this experiment counts every trajectory. The broad transport/coarse-graining themes are shared, and no claim of an unrelated mathematical principle is made. Details and hashes are in overlap-and-sources.json.

[Majda and Kramer](https://mhd.ens.fr/IHP09/Young/Biblio/MajdaKramer.pdf), sections2.1–2.2, support the cell construction and qualitative distinction between shear and cellular transport. That paper is not cited as proof of the proposed self-consistency formula. A separate publisher abstract on Gaussian self-consistent diffusion is background only; full text was not read. The surrogate above is justified directly as its own positive stochastic process.

Empirical difficulty is unknown. An agent may immediately see that the known velocity must be sampled at the actual tracer position, then solve the cell problem. The refined source reduces the superficial missing-component diagnosis but does not guarantee a physical reasoning failure. A trial deriving the correct cell generator and then failing on Fourier signs or grid convergence would be mathematical/numerical, not the desired target. Root has authorized this bounded prototype only.
