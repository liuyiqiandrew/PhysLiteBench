# Hydrodynamic heating r4

This revision changes the contact material and measured subsystem. It preserves the correct mechanics and dashpot heat recovered by both r3 passers. Canonical r1 scored 1/3, archived r2 scored 2/3, and archived r3 scored 2/3 with two physical passes and one physical failure. Those results and their source/trajectory reviews remain unchanged. A larger contact network or additional phases would repeat the solved r3 law and were not used.

The r4 spring is locally equilibrated, with measured isothermal stiffness `k(T)=k0[1+(T-T0)/(2T0)]` and positive constant fixed-extension heat capacity C0. Both complete contacts are held at T0. The front calorimeter collects signed heat from the elastic element and dashpot; rear-contact and bulk heat are separate. The public apparatus gives force-temperature data and heat capacity, not an entropy or heat formula. This is a new constitutive and detector specification, not a reinterpretation of r3's temperature-independent energetic spring and dashpot-only detector.

## Mechanics and thermodynamics

All linear electromagnetic and mechanical response is retained from the r3 physical oracle. In the reduced units, `gamma=.06`, `nu=.04` and the unknown plasma frequency `p` lies in `[.85,1.15]`. With the incoming `exp(-i omega t)` clock, the contact impedance is `alpha/(1-i omega tau)`, and the fluid has both electromagnetic and viscous modes. Both faces satisfy the signed traction boundary condition and continuity of tangential electromagnetic fields. The known controls span the unchanged r3 box.

Write `g=.5`, `k0=kappa/tau`, extension e, and dashpot velocity vd. The measured force and heat capacity admit

```
psi(T,e) = k(T)e^2/2 + C0[(T-T0)-T ln(T/T0)]
s(T,e) = C0 ln(T/T0) - k'(T)e^2/2
U(T,e) = C0(T-T0) + [k(T)-T k'(T)]e^2/2.
```

Affine caloric terms have no effect on this experiment. At fixed T0, `U=(1-g)k0 e^2/2`. The traction into the contact is `k0 e=kappa vd`, and slip is `e_dot+vd`. The whole-contact first law and entropy balance independently give

```
q_out = traction*slip - U_dot
      = kappa vd^2 + g k0 e e_dot
      = kappa vd^2 - T0 s_dot.
```

Stiffness stays positive in the stated temperature neighborhood. C0 is positive, and entropy production is `kappa vd^2/T0`. Whole-contact heat can be instantaneously negative because elastic heat is reversible; it is not pointwise dissipative power. The positive absolute T0 and caloric baseline do not introduce unknown parameters in the isothermal observable.

For unit incoming field, let `a=alpha*J/(1-i omega tau)` be scaled traction. Dividing heat by incident mean power gives mean `|a|^2/(alpha p^2)` and harmonic

```
Q2 = (1-i g omega tau) a^2/(alpha p^2).
```

The supplied source returns the exact dashpot harmonic `a^2/(alpha p^2)`. It thus implements a complete reduced caloric attribution while retaining all fields, contact dynamics and r3 heat physics. It is not an exact thermodynamic alternative satisfying the stated force-temperature data. Its completed control is byte-identical to the archived r3 physical oracle. The public starter only removes fitting code and imports used by that code; the constructor remains None.

## Calibration, identification and signals

The reversible elastic contribution integrates to zero over every period. The complete mean map and fitting objective therefore agree exactly at every p and every allowed control. The six nonzero finite-memory calibration settings, repeated48 times, retain all288 r3 records byte-for-byte, including fixed sigma=.0002. Calibration seed193051 and the original noise seed193053 are private. No data were regenerated. The inherited mean generation used the independent r3 time reference; its small finite-tolerance difference from the exact mean is below numerical significance and is not instrument noise.

The actual fits independently give p=1.0301321259201883 and reduced chi-square1.047512 for both controls. A fresh1001-point parameter grid verifies exact source/oracle mean equality and increasing sampled curves, with minimum sampled secant derivative .05708755. Forty-one noiseless fits, including both endpoints, recover p within7.05e-9. Dense noiseless objectives have at most one sampled interior minimum. These are numerical full-range checks, not an interval proof or a claim about arbitrary noisy objectives.

The256 existing r3 oracle noise fits are transferred exactly, because both new controls have that identical objective. Their parameters, calibration chi-square and parameter errors are retained, and new whole-contact predictions are evaluated against the new reference. These are reused realizations, not256 fresh draws or independently refitted new records. All pass calibration and parameter checks. All oracle groups pass; the source fails all three harmonic groups and passes the mean anchor. Worst oracle error is .00094333, and minimum source diagnostic error is .300765. Actual-source diagnostic errors are .30108, .40257 and .36510 against the unchanged .04 gate.

The diagnostic RMS signals at true p are .04095, .03506 and .04715 of incident mean power. Minimum diagnostic separation over the41 sampled p values is .29256. Paired quadratures avoid dividing by an individual component near zero. Their exact relative source error is `|g omega tau|/sqrt(1+(g omega tau)^2)`, giving6.983%-66.897% over the public box; this does not claim a bound for a single component. All weak cases from the bounded study remain. The author-only g0 limit is exact, g=.01 and small tau can fall below4%, and tau0 is the memoryless limit. The tau=.003 and tau0 rows are analytic checks only.

The private parameter, hidden data, reports and thermodynamic diagnosis stay outside the public image. Calibration chi-square1.5, relative parameter error.03, grouped prediction error.04, and600/60-second model/verifier budgets are unchanged. The original unrestricted neutral instruction and public tests are unchanged.

## Independent reference and numerical evidence

The verifier solves the eight-real Maxwell/momentum boundary problem with collocation, independent of the modal root implementation. It then integrates a real extension ODE from zero for eight drive periods, measures a settled period, and projects mechanical input minus U-dot onto the clock harmonics. With rho=1 as a harmless normalization, `v=J/p`, `kappa=alpha`, and `k0=alpha/tau`; dividing by unit-field incident power multiplies heat by2. The separate harmonic projection adds its own factor2. The final complex heat correction is not inserted into this reference.

The reference shares the declared continuum and the inferred equilibrium caloric law; it is not a microscopic contact-force simulation or an independent measurement of that law. Its stored instantaneous entropy expression is a cross-check of first-law heat. No modal oracle is imported into the reference.

All16 graded settings and6 calibration settings are checked at p=.85,1.03,1.15. For each of these66 cases, field tolerance, temporal tolerance/step, settling8→12 periods, and measured periods1→2 are varied separately. Maximum oracle/reference discrepancy is2.96e-14; maximum individual refinement change is2.10e-14. All28 graded scalar outputs are recorded, including the mean anchors. The largest independent mean bias over the calibration/endpoints is5.31e-11 sigma.

All32 full-domain corners and32 random interiors additionally compare full quadrature vectors and refined references. Maximum reference error and refinement are2.89e-13. Mean optical absorption equals bulk heat plus both contact mean heats within1.45e-15. First-law/entropy waveform agreement is8.33e-17, and integrated U-dot agrees with the endpoint energy within4.36e-13. The minimum sampled quadrature-pair RMS is .00656. Negative instantaneous whole-contact heat is retained and correctly allowed.

Cold hidden reference evaluation took1.11 seconds. Fresh isolated local copies ran the actual public/private tests: oracle7/7 in1.729 seconds; source4 passes and3 intended diagnostic failures in1.730 seconds. Full author validation took46.04 seconds. This is not a Docker or model evaluation.

## History and scope

The exact12-payload bounded study and its original manifest are copied into prototype. Its pending-peer labels are preserved historical state; the later approved peer and root packaging decision are separate history files. The complete earlier r3 report, source, calibration and current three-trial results/reviews remain available. The initial provisional family assessment reviewed only the first two r3 passers and keeps that qualification.

The first package validator contained an unclosed parenthesis. Both syntax checking and its attempted invocation stopped before science; the exact failed script and failure record are in development/initial-validator-syntax-failure. Only the missing parenthesis changed. The corrected run completed all planned science and controls. No scientific failure or weak case was removed.

Thermoelastic-rod r4's three passers already derived reversible entropy coupling from an explicit free energy. Here that dependence must be inferred from measured force-temperature data and a whole-contact detector. This is a substantive same-family inference change, with shared thermodynamic principles and uncertain empirical difficulty. A later failure must be reviewed from its full public actions and final source: a correct entropy or first-law derivation followed by a sign, factor or numerical error is not a clean physical closure failure.

Reproduction uses `scripts/validate_hydrodynamic_heating.py` with NumPy2.3.3, SciPy1.16.3 and pytest8.4.2. It reads inherited calibration and refuses to overwrite an existing validation report; preserve old output before an intentional rerun. No data-generation, Docker or model command is executed. Final evidence, input audit, provenance and independent peer are under results.
