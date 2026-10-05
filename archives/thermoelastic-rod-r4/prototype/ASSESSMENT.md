# Thermoelastic rod: bounded hardening review

Status: author science complete for a bounded prototype; independent peer pending. No task harness, data generation, model evaluation, or canonical source change has been made.

The current rod revision scored 1/3 in the fresh retained batch. All three solvers actually derived the correct uniform-total-stress mechanics. The two failures compared their alternative only in the calibration subspace, found agreement, and retained the supplied closure. Their exploratory array and normalization errors had been corrected before that decision. The passer also compared contact and even-mode preparations, then installed the physical mechanics. The complete public-command, final-source, input, and outcome evidence is in `trial-and-history-review.json`. Changing the viscosity profile would not test a new assumption in that passing method.

The proposed revision changes the material from Kelvin–Voigt to a standard linear solid with a thermally coupled Maxwell arm. Both completed predictors retain uniform total stress, the clamp constraint, and the exact internal-strain relaxation. They differ in the caloric treatment of the relaxing arm. This is a same-family revision, not a new independent theoretical principle.

## Material and preparation

Let theta=T−T0, epsilon be axial strain, and z be the Maxwell-arm viscous strain. The Helmholtz free energy per reference volume is

```
psi = E0*epsilon^2/2 + E1*(epsilon-z)^2/2
      - theta*[beta0*epsilon + beta1*(epsilon-z)]
      - ce*theta^2/(2*T0).
sigma = (E0+E1)*epsilon - E1*z - (beta0+beta1)*theta.
eta*z_dot = E1*(epsilon-z) - beta1*theta.
```

E0=E1=2e9 Pa; beta0=E0*0.002*[1+0.65*cos(2*pi*x/L)] Pa/K; beta1=chi*beta0 with known chi in [0,1]. The drag is eta=E1*(12 s)*[1+0.97*cos(2*pi*x/L)] Pa s. The original L=.05 m, area=1e-4 m², T0=300 K, ce=1e6 J/(m³ K), conductivity range [80,220] W/(m K), and finite body heat capacity 6 J/K are retained.

The effective one-dimensional material has already undergone transverse relaxation. Fixed end separation imposes mean epsilon=0; negligible axial inertia imposes spatially uniform total sigma. Fourier heat conduction, a massless left-end contact of conductance H in [0,1.2] W/K, and insulated remaining surfaces complete the boundary conditions. No internal-variable boundary condition is needed because z has no spatial-gradient energy or transport.

Each prescribed initial temperature profile is maintained by preparation heaters while the clamps stay fixed, until z_dot=0. The heaters are removed and the specified thermal contact is made at t=0. Thus

```
epsilon_initial = [beta0*theta_initial - mean(beta0*theta_initial)]/E0
z_initial = epsilon_initial - beta1*theta_initial/E1.
```

This is the same initial state for both predictors. The perturbation model retains terms first order in temperature departure and strain about T0. The internal dissipation eta*z_dot² and mechanical work products are second order and do not supply a first-order heating term. This does not remove the first-order reversible thermal coupling. Initial temperatures obey the previous 1 K bound; the prototype uses the old mean/cosine temperature measurements and time range 0–80 s.

## Physical model and completed approximation

The physical entropy departure and heat equation are

```
s = ce*theta/T0 + beta0*epsilon + beta1*(epsilon-z)
ce*theta_dot + T0*(beta0+beta1)*epsilon_dot - T0*beta1*z_dot = k*theta_xx.
```

The completed approximation evaluates the caloric state at local Maxwell-arm relaxation while retaining the actual finite-rate mechanical state:

```
s_rel = (ce/T0 + beta1²/E1)*theta + beta0*epsilon
(ce + T0*beta1²/E1)*theta_dot + T0*beta0*epsilon_dot = k*theta_xx.
```

This is the entropy obtained by first minimizing psi over z. It is a definite partial-equilibration approximation with positive effective heat capacity, rather than a missing numerical update. It uses chi in both the mechanical and caloric equations. It is not an exact alternative thermodynamic material with the unchanged stress law: the transient stress/entropy reciprocity of the supplied psi would forbid that interpretation. In particular,

```
s - s_rel = beta1*[E1*(epsilon-z)-beta1*theta]/E1.
```

The source conserves its own relaxed linear caloric energy. It does not conserve the physical psi-based linear energy at chi>0: the bounded diagnostics find up to 1.0913 J transient drift. That defect is a consequence of its assumed caloric equilibration and must remain disclosed. Both predictors share the equilibrated initial state and final equilibrium; sampled late temperatures agree within 2.8e-9 K. Neither final equilibrium nor calibration alone exposes the approximation.

At chi=0 the two equations coincide for every allowed experiment, while beta0 remains nonzero and Maxwell mechanical relaxation remains active. This gives nontrivial exact calibration. The proposed calibration uses an initial 0.8*cos(pi*x/L) K profile, insulated ends, and the first cosine temperature component at five times. All five predictions decrease on the sampled 81-point conductivity grid; nine fits, including endpoint truths, recover conductivity within 1.76e-8 relative error. This is bounded numerical evidence, not an analytic global-injectivity proof or a noise study. A full package would still need those checks.

## Preserved numerical evidence

`check.py` / `report.json` preserve all 54 first-screen combinations: chi=0,.01,.1,.25,.5,1; conductivity=80,145,220; and H=0,.4,1.1. Each includes three initial preparations, six times, and all four old temperature observables. The zero and small-chi cases remain present. Maximum all-observable RMSE at chi=1 is only .01627 K, below the old .025 K threshold; no claim is made that every readout distinguishes the models.

`followup.py` / `followup-report.json` preserve the subsequent bounded checks. Three ordinary mean-temperature groups at chi=1 have minimum RMSE .03871, .02972, and .03439 K over 15 conductivity values spanning the full interval, compared with the unchanged .025 K gate. Their minimum physical RMS signal is .08645 K. The times are 1,5,20,80 s. The insulated group uses initial (mean,first,second,bath)=(.2,0,.8,-.6); the two contact groups use (-.3,.3,.4,.8). These groups were selected after the broad screen, and are feasibility evidence, not predeclared model-test outcomes.

The primary calculation eliminates strain into temperature/internal-strain equations. The reference instead evolves cell entropy and internal strain, recovering temperature and the common stress directly from the local constitutive equations. It does not reuse the temperature-capacity blocks. Both share the finite-volume heat operator, so the reference is independent constitutive elimination, not independent spatial discretization. Their maximum first-screen difference is 4.32e-11 K. A separate entropy-coordinate BDF propagation agrees within 1.64e-10 K at an off-grid setting. For three refined cases, 80→160-cell changes are at most .000136 K and are about one quarter of the 40→80 changes.

The physical solver preserves linear total energy within 1.65e-11 J, mean strain within 5.7e-19, and total-stress uniformity within 1.3e-8 Pa in the initial screen. Its quadratic availability decreases at the sampled times. Both sampled generators have no positive growth rate beyond 3.5e-14 /s numerical roundoff. The physical availability is positive and obeys the conduction-plus-internal-relaxation dissipation balance; this supplies its analytic stability argument. Stability of the reduced caloric closure is currently a bounded numerical check, not a proof over the full continuous domain. Screen and follow-up runtimes were .84 and 2.47 s respectively on the author machine.

## Scope and next decision

This change tests a physical assumption absent from the passing r3 solution: whether the caloric state of a relaxing internal strain may be replaced by its relaxed state. It preserves the passers' correct mechanics. It may still be easy for a solver that differentiates the supplied free energy correctly. There is no empirical evidence of a new model failure yet.

The original failures, broad weak cases, all attempted prototype cases, and source-energy defect are retained. A possible second-order mean-heating variant was not implemented because it would add a different perturbation order and overlap ongoing viscous-heating work. A coefficient-only or profile-only r3 change was rejected because the passing mechanical inversion already handles those changes.

Root review and independent actual-file peer are required before any harness. Future packaging would need a complete public domain, exact calibration identifiability or a sufficient global check, fixed instrument noise, 256 noise draws, full-domain and per-scored-case refinement, and local controls. None of those future requirements is claimed complete here. Primary background and exact access scope are recorded separately in `source-access.json`; the particular quadratic material and approximation above are author constructions.
