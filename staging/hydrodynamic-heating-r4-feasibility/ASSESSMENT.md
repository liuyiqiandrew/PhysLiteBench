# Thermoactive contact: bounded r4 feasibility

Status: the one bounded numerical run passed. Independent actual-file peer and root review are pending. No task package, model trial, Docker invocation, canonical edit or Git operation was made.

This is a same-family revision. The archived r3 finished with two physical passes and one physical failure. Its passers retained the full Maxwell/viscous response and correctly computed dashpot heat. The new completed source is that correct response, copied exactly from the r3 physical oracle. It does not revert to the failed mechanical-work readout. Prior artifacts and the earlier two-source assessment remain unchanged.

## Changed material and measurement

Keep the r3 slab, periodic forcing, mechanical contact and fitted plasma frequency. Replace its temperature-independent energetic spring by a locally equilibrated elastic element with measured isothermal traction

`t_c = k(T)*e`, where `k(T)=k0*[1+g*(T-T0)/T0]`, `g=1/2`, and `k0=kappa/tau`.

The material has a positive constant fixed-extension heat capacity per area C0. The force law holds throughout a stated neighborhood, for example `.9*T0 <= T <= 1.1*T0`, where stiffness stays between `.95*k0` and `1.05*k0`. The complete contact is clamped at T0 during measurement. The dashpot remains Newtonian and temperature independent. There is no additional internal relaxation. A whole-contact thermostat now collects heat from both elastic element and dashpot; bulk and rear-contact heat remain separate. The calorimeter records signed heat with the same clock and mean/2ω quadratures as r3.

These are explicit changes to material caloric data and detector scope. They cannot be silently applied to the old dashpot-only apparatus. Future public prose can give the measured force-temperature law and heat capacity without supplying an entropy or heat correction formula.

## Two thermodynamic routes

Integrating the measured force and heat capacity gives an admissible free energy per area

`psi(T,e)=k(T)*e²/2+C0*[(T-T0)-T*ln(T/T0)]`,

up to irrelevant affine caloric terms. Its entropy is `s=C0*ln(T/T0)-k'(T)*e²/2`. At fixed T0, entropy balance gives

`q_out = kappa*v_d²-T0*s_dot = kappa*v_d²+g*k0*e*e_dot`.

Independently, internal energy is `U=C0*(T-T0)+[k(T)-T*k'(T)]*e²/2`. At T0, `U=(1-g)*k0*e²/2`. Mechanical power enters as `t_c*v_slip`, so the first law gives

`q_out = t_c*v_slip-U_dot`,

with `v_slip=e_dot+v_d` and `t_c=k0*e=kappa*v_d`. It yields the same heat. Positive C0, k(T) and kappa give stable local caloric and mechanical response. At g=.5 the extension part of internal energy is positive. Entropy production is `kappa*v_d²/T0 >= 0`; whole-contact heat can be negative during part of a cycle because the elastic heat is reversible. Pointwise nonnegative heat would be an incorrect check for this detector.

For the `exp(-i*omega*t)` convention, if Qd2 is the normalized dashpot-heat harmonic, the entropy route gives `Q2=(1-i*g*omega*tau)*Qd2`. The mean is unchanged exactly for every control and p. The source returns the exact dashpot result for the whole-contact signal. This is a reduced caloric attribution, not an exact alternative material having the stated temperature-dependent force law.

## Independent numerical check

The oracle uses the preserved modal field solver and the entropy-derived harmonic. The reference uses the preserved independent eight-real collocation Maxwell/momentum BVP, then integrates real spring extension from zero for eight periods before measuring a settled period. With a harmless normalization rho=1, it uses `v=J/p`, `kappa=alpha`, `k0=alpha/tau`, and `e_dot=v-e/tau`. It computes mechanical input minus the derivative of the internal energy above. It does not insert the final complex heat factor. A separate instantaneous entropy expression checks the first-law waveform.

The first successful run took 7.24 seconds. All 32 endpoint combinations and 16 random interiors were retained. Maximum oracle/reference discrepancy is 2.89e-13; modal/collocation wall-current disagreement is 1.70e-13. The first-law and entropy waveforms agree within 8.33e-17, and integrated U-dot agrees with the endpoint energy within 4.36e-13.

Six representative domain cases separately refine field tolerance, temporal tolerance/step, settling from eight to twelve periods, and recording from one to two periods. Their maximum changes are 2.55e-14, 1.94e-13, 4.29e-15 and 4.13e-14, respectively. This is independent field and real-time heat evaluation within the same declared continuum/thermodynamic model, not a microscopic contact simulation or an entirely independent material law.

## Signals and calibration

For a matched pair of in-phase/quadrature outputs, the relative error is exactly

`|g*omega*tau|/sqrt(1+(g*omega*tau)²)`.

At fixed g=.5 and the unchanged frequency/tau box, this ranges from 6.983% to 66.897%. This bound is for paired quadratures, not a single component near a zero. The sampled minimum paired signal RMS is .00656 of incident mean optical power. The inherited three diagnostic groups have RMS signals .04095, .03506 and .04715, with source errors .30124, .40274 and .36525. The mean anchor remains shared. No control settings were selected after seeing a model outcome.

The existing 288 readings retain fixed sigma=.0002. Their complete mean prediction and fit objective equal the r3 physical oracle for every p. The actual fit is reproduced exactly. Existing 41 noiseless full-range recovery/profile checks therefore transfer, with the original qualification: sampled monotonicity is not an interval proof or a theorem about arbitrary noisy objectives. The independent first-law mean differs from the exact calibration mean by at most 4.99e-11 sigma.

The report reuses the preserved 256 r3 noisy oracle fitted parameters and rescored the new readouts against the new first-law reference. These are not fresh noise draws and no calibration was regenerated. All calibration/parameter outcomes remain the recorded passing outcomes; the new maximum oracle error is .00094333 and minimum diagnostic source error is .300765. The common fit reuse is justified by exact mean-map equality, not by substituting a favorable physical fit into a different source objective.

Weak cases remain: g=0 gives equality; g=.01 and tau=.03/.003 give gaps below4%; tau0 is the analytic memoryless limit. The ordinary g=.5,tau=.7 baseline is also retained in that limit collection. Real-time references were run for g0/.01/.5 at tau=.7 and g=.5,tau=.03. The tau=.003 and tau0 rows are analytic checks only, not independent time-integration claims.

## Scope and next gate

Thermoelastic-rod r4 obtained three physical passes after deriving reversible entropy coupling from an explicit free energy. This contact revision instead asks for the entropy dependence implied by measured isothermal force-temperature data and heat capacity. It is a real change in the presented inference and measured subsystem, with shared thermodynamic principles. Thermal-bodies and rotating-contact history also warn that bath attribution and simple energy balances can be easy. No empirical difficulty or prospective0/3 outcome is claimed.

No failed numerical attempt occurred in this bounded study. All inherited evidence, the one executed source/report, all weak cases and the earlier provisional deferral are preserved. The copied prior assessment retains its historical third-review-pending qualification; current r3 results are separately linked in provenance. A full package would still need a final public/input audit, all graded-case refinements, complete local controls, final source peer and root freeze. This directory is not an evaluation-ready task.
