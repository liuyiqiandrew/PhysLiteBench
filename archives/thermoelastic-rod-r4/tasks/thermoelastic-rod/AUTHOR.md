# Thermoelastic rod r4

This is an unmeasured same-family revision of the canonical r3 rod, whose fresh retained batch scored 1/3. No r3 artifact or outcome is replaced. The complete three-trial public/native review and prior history are preserved in `prototype/trial-and-history-review.json` at the stage root.

All three fresh r3 solvers actually derived the correct uniform-total-stress mechanics. The two failures corrected their exploratory code but compared only the calibration subspace and retained the starter. The passer also checked contact/even preparations and installed the correct mechanical inversion. Changing the viscosity profile would not test a new assumption of that passing solver. Revision 4 instead adds a thermally coupled Maxwell internal strain; both completed controls keep the correct force balance, clamp constraint and internal-strain kinetics.

## Physical distinction

For the public Helmholtz free energy, the stress and entropy departure are

```
sigma = (E0+E1)*epsilon - E1*z - (beta0+beta1)*theta
s = ce*theta/T0 + (beta0+beta1)*epsilon - beta1*z.
```

The heat equation at first perturbative order is

```
ce*theta_dot + T0*(beta0+beta1)*epsilon_dot - T0*beta1*z_dot = k*theta_xx.
```

The internal production eta*z_dot² and mechanical power products are quadratic and do not enter this first-order temperature equation. They remain the dissipative terms in the quadratic availability balance. The preparation heaters establish a relaxed Maxwell arm before release:

```
epsilon_initial = [beta0*theta_initial - mean(beta0*theta_initial)]/E0
z_initial = epsilon_initial-beta1*theta_initial/E1.
```

The completed source evaluates the caloric state at locally relaxed z while keeping finite-rate z in the mechanical equations. Minimizing psi over z gives

```
s_rel = (ce/T0+beta1²/E1)*theta + beta0*epsilon.
```

The source solves the corresponding heat balance exactly at its spatial resolution. It is a partial caloric-equilibration approximation, not an exact alternative thermodynamic material with unchanged stress. It preserves its own relaxed caloric energy, but violates the actual material's transient stress/entropy reciprocity and physical linear energy. The bounded prototype records up to 1.0913 J transient physical-energy drift. The discrepancy is exactly beta1 times the Maxwell-arm stress divided by E1 in the entropy. This qualification is intentional and must not be recast as a coding defect.

Both controls retain the actual chi-dependent mechanical stress, uniform total stress, zero mean strain, exact Maxwell relaxation and finite thermal body. Their initial and final equilibrated states coincide. At chi=0 their full equations coincide for every preparation, despite nonzero beta0 and active viscoelastic mechanics. The oracle changes only the thermal capacity and internal-strain storage blocks in the completed source. The public fit remains unfinished with `conductivity=None`.

## Apparatus and input

The public description supplies the full free energy and internal kinetics, fixed-clamp preparation, thermal contact, insulation, linearization, units and temperature measurements. It does not supply the entropy correction or a numerical algorithm. The ordinary instruction is unchanged and explicitly permits replacement of every helper. The image copies only the public README, starter, tests and data. True conductivity and author explanations stay private.

Calibration has 15 distinct settings: chi0, an insulated initial first cosine mode with amplitude −.8,.4,.8K, and times .2,1,4,12,40s. Each is repeated 16 times, giving 240 readings. The fixed instrument standard deviation is .0005 K, independent of the response, material setting and unknown. Public and private data are identical. Explicit regeneration uses the 320-cell independent entropy reference; the 80-cell oracle's deterministic calibration difference is 1.54e−5 K, below .031 sigma. It is recorded separately from random noise.

The three diagnostic groups measure ordinary mean rod temperature at chi=1 and times 1,5,20,80 s. The original absolute .025 K RMSE gate, 3% parameter gate, chi² 1.5 gate, 600 s agent and 60 s verifier limits remain unchanged. The diagnostic preparations were selected in the bounded author study, whose full weak-case record is preserved. Zero/small-chi settings are valid and retained in scientific checks; not every allowed experiment is expected to distinguish the controls.

## Independent reference and checks

The forward model eliminates strain into a temperature/internal-strain generator on 80 finite-volume cells. The verifier evolves cell entropy and internal strain, directly solves the common stress using constitutive equations, and computes heat transfer from face fluxes on 160 cells. It does not use the forward capacity/storage blocks. It shares the finite-volume spatial family, so this is independent constitutive elimination, not a separate spatial method. The bounded prototype additionally compared entropy-coordinate BDF time integration with the eigensystem propagation, agreeing within 1.64e−10 K.

Final scientific validation checks 45 structured parameter systems and 12 random interiors, with 72 temperature readouts per system, including zero time, all four observables, signed preparations, all parameter endpoints, chi0/.01 and contact endpoints. Maximum oracle/reference difference is .000402 K. Every graded input is independently refined at conductivity 80,145,220: the reference 160→320 change is at most 1.72e−5 K and oracle 80→reference 320 difference 8.60e−5 K.

The physical generator conserves linear energy to 9.1e−14 in the tested generator residuals and satisfies the exact quadratic dissipation identity to 3.8e−15 relative error. Its availability matrix is positive. Both controls satisfy common-stress and clamp constraints, relaxed initial z rate, and sampled spectral stability. The source conserves its own caloric-energy weights; its physical-energy residual is intentionally nonzero at chi>0. Source thermal capacity is positive because beta1=chi*beta0 makes its nonlocal addition positive semidefinite. Full continuous-domain source stability is not claimed from sampling alone.

An 81-point calibration grid has strictly monotone signed-channel profiles; 17 noiseless endpoint/interior fits recover conductivity within 1.76e−8 relative error, with at most one sampled objective minimum. This is a numerical global-identification check, not an analytic theorem about arbitrary noisy objectives. Across 41 conductivities the minimum diagnostic RMSE is .0297255 K, above the unchanged.025 K gate; the narrowest margin is disclosed.

All 256 independent noise draws pass the oracle calibration, parameter and prediction checks; all 256 completed-source draws pass calibration/parameter and fail all three diagnostic groups. Maximum oracle error is .000145794 K; minimum source error is .034632 K. Maximum calibration reduced chi² is 1.28195 and maximum relative parameter error .0018082. The controls have identical calibration equations, so each draw's same fitted parameter is used for the completed source; actual separate source fits at the first and last draws verify that equivalence.

Fresh local public+private pytest runs give oracle 7/7 in .99 s and source 4 passed/3 intended prediction failures in 1.00 s. The broader author sweep took 72.44 s; it is not the verifier runtime. There have been no r4 Docker controls or model trials. These results establish scientific validity and separation, not empirical difficulty. Solvers that differentiate the free energy may repair this directly.

## Evidence and reproduction

Run `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 python scripts/validate_thermoelastic_rod.py` from the stage root. Add `--regenerate` only for an intentional calibration update. The first full validation succeeded; its output log is preserved. Scientific, local-control and input-audit reports, source provenance and the complete immutable bounded prototype are included in the stage. Local controls ran the actual pytest files in separate fresh temporary app/test copies without modifying either source.

The existing r1/r2/r3 history is retained in the copied review. A second-order mean-heating alternative was not implemented because it changes perturbative order and overlaps other viscous-heating work. Profile-only r3 changes were rejected based on the actual passing implementation. Future causal trial review must distinguish retaining the relaxed caloric assumption from recognizing the entropy of z but making an algebra, time-integration or implementation error.

Primary background: A. Berezovski, [Constitutive Modeling with Single and Dual Internal Variables](https://www.tud.ttu.ee/web/Arkadi.Berezovski/texts/2023-entropy-25-00721.pdf), Entropy 25, 721 (2023), sections 3.2 and 5.2, supports the thermodynamic internal-variable framework. The exact access scope and failed/limited discovery accesses are recorded in the prototype. The particular quadratic material and reduced caloric approximation are explicit author constructions, not quoted empirical constitutive laws from that article.
