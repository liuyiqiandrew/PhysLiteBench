# Quenched spin activity r1 physics review

Independent source review, 2026-10-11. **Approved for evaluation; no blocking physics, ambiguity, or numerical issue found.** This is approval of the apparatus and scientific controls, not evidence that the target model fails. The evaluation target is three fresh, frozen, unhinted `gpt-5.6-luna` High trials.

I inspected the public apparatus and starter, AUTHOR.md, completed oracle and shortcut, finite-size reference, validator, public/private tests, calibration and hidden data, and image/configuration files. I read the completed [validation report](expansion-quenched-spin-activity-r1-validation.json) and performed the additional microscopic and counting-function checks below. No task code was changed during this review.

## Apparatus and physical reduction

The public energy and excluded-spin flip rate are mutually consistent. For a spin with local field `h_i`, flipping it changes the energy by

    delta_E = 2*sigma_i*(J*(S-sigma_i)/N+h_i).

The displayed rate is `kappa/(1+exp(delta_E))`, which obeys finite-N Gibbs detailed balance. The two populations each contain exactly half the spins and retain their labels throughout a run. This is quenched field heterogeneity, without an additional disorder average or species exchange. At zero disorder the labels still define distinct counters.

The counters include both flip directions with positive increments. Their counting biases alter statistical weights only. The initial finite-N canonical state, uninterrupted dynamics, long-time-then-size limit, even-size sequence, normalization per spin per time, and possible output signs are all explicit. The unknown attempt rate is common to both species and has the stated inverse-time units. Hidden inputs introduce no new preparation, interaction, or measurement rule.

At each finite size, positive flip rates give one positive dominant tilted eigenvector. The canonical initial state has nonzero overlap with it. Consequently equilibrium coexistence does not leave an unspecified initial-phase selection in the requested long-time function.

## Variational oracle and finite reference

For within-population magnetizations `z_plus,z_minus`, put `M=(z_plus+z_minus)/2` and `f_g=J*M+h_g`. The flip rates per spin of population `g` are

    u_g = kappa*(1-z_g)*(1+tanh(f_g))/4
    d_g = kappa*(1+z_g)*(1-tanh(f_g))/4.

After detailed-balance symmetrization, the limiting diagonal is minus the total physical escape rate and each pair of opposite flips contributes `2*exp(-s_g)*sqrt(u_g*d_g)`. Weighting each species by one half yields precisely the oracle's potential

    kappa/4 * sum_g [exp(-s_g)*sqrt(1-z_g**2)/cosh(f_g)
                    -1+z_g*tanh(f_g)].

The thermodynamic answer is its global maximum. A maximum-row-sum bound gives the upper bound; positive vectors localized near a maximizing macroscopic density give the matching leading Rayleigh quotient. The two-species population factors, counting signs, and energy convention are correct. The angle substitution removes the square-root endpoint singularity without changing the maximization domain.

The reference independently constructs the exact two-count birth-death process. For total spin `S`, an up flip samples `(S+1)/N`, while a down flip samples `(S-1)/N`. Its multiplicities are the available down/up spins in the relevant fixed population. Symmetric off-diagonals use the geometric mean of forward and reverse count-state rates, with the species counting factor. The unchanged escape diagonal and division by `N` are correct. The largest eigenvalue of this lumped process equals that of the full spin process: permutation symmetry within each population preserves the unique positive dominant state.

As an additional independent check, I built the full 64-state process at `N=6` from explicit spin configurations and Hamiltonian energy differences, using `kappa/(1+exp(delta_E))`. Across six preparations, including coupling/bias extremes and opposite species biases, its dominant tilted eigenvalue per spin agrees with the count reference within `5.14e-16`. The microscopic detailed-balance residual is below `6.94e-18`. This checks count multiplicities and finite-size factors without reusing the reference's reduced rates.

The reference extrapolates sizes 80, 160, and 320; refinement uses 160, 320, and 640. On hidden groups, the largest change is `1.221e-6` relative, and the largest oracle discrepancy from the refined reference is `1.798e-7` relative. These are far below the 4% grading gate. The 36 random/boundary checks have maximum absolute discrepancy `3.57e-5`; finite-size extrapolation is less accurate at some domain points than on the graded groups, as expected near transition regions. The report appropriately distinguishes this domain result from hidden accuracy.

Global-search grids 49 and 97 agree to floating-point precision on all checked preparations. This is numerical evidence, not a formal guarantee that every continuous-domain maximum is enumerated. In combination with the independent finite-size comparisons and the low-dimensional bounded domain, it is sufficient for the current frozen cases. A missed maximum in a future agent submission must still be diagnosed as a numerical failure when its physical formulation is correct.

## Completed shortcut and calibration

The shortcut selects the global equilibrium free-energy minimum and then freezes the resulting local fields while calculating exact independent-spin counting functions. Where equilibrium minima are degenerate, it retains every phase and takes the maximum of their long-time counting functions. This is the generating function of a fixed mixture of auxiliary processes: each run selects a phase with nonzero weight and retains that phase's fields. Those weights do not affect the asymptotic function. The control is therefore a coherent physical approximation, including coexistence, rather than a counting-field-dependent root prescription.

The physical omission is feedback from trajectory conditioning to the collective field. Both the exact apparatus and the approximation retain the fixed field populations. The control does not fail because its two-state counting algebra is wrong.

At `coupling=0`, the real apparatus is exactly the auxiliary independent-spin process. Direct two-state eigenvalues verify both controls' calibration law. The law is linear in `kappa`, with nonzero known coefficients, so the scalar weighted least-squares objective has one minimum. Both fitted rates are `1.13010692172`; both reduced calibration chi-squares are `1.00368974`. Calibration values, standard uncertainties, and the exposed unknown contain no private fitted answer.

## Robustness, separation, and image boundary

Across 256 independent noise datasets, fitted rates range from `1.12918880` to `1.13095186`, within `0.08424%` of truth and well inside the 3% parameter gate. Maximum reduced chi-square is `1.34699`. The largest oracle hidden error is `0.08436%`, versus the 4% gate. The weakest shortcut diagnostic group remains at `12.46%`; the other group minima exceed `45.9%`, `74.3%`, and `98.4%`.

The five group RMS signals are `0.1957`, `0.1219`, `0.2843`, `0.07852`, and `0.1255`. None of the grading groups depends on a nearly zero denominator. Calibration/noise error and finite-size error are substantially smaller than the physical separation.

The validator checks zero bias and spin reflection with population exchange. My additional checks on 20 random apparatus settings found no violation of nonincreasing dependence on either activity bias or joint convexity in the two biases. Supplemental checks used the available NumPy 2.5.2/SciPy 1.18.0 environment; task images retain the pinned dependencies.

Public/private calibration files are byte-identical. The Docker image explicitly copies only the public README, model, test, and calibration directory. True parameters, hidden cases and truth, independent reference, complete solution, and author explanations remain outside it. The instruction permits replacing all implementation code. Frozen hidden values and unchanged 600/60-second limits keep verification practical.

The [cited primary paper](https://arxiv.org/abs/2108.00063) studies activity-biased trajectories of a fully connected random-field Ising model and supports the collective mechanism. The task's two separately biased populations and normalization are also checked directly above. The related qualified exclusion-ring task concerns a conserved spatial density profile; this apparatus concerns two nonconserved, permanently labelled magnetizations. The common trajectory-conditioning principle should remain acknowledged.

## Evaluation requirement

No empirical failure is established by this review. Retaining the equilibrium collective field, or replacing the fixed field populations by an annealed field process, can establish a physical failure when supported by the actual trajectory and final code. Correct collective physics followed by an incorrect factor, optimizer, sparse eigenvalue calculation, or size extrapolation cannot be counted automatically as the intended failure. Docker controls and all three fresh target-model outcomes must be retained and inspected.

Reviewed source SHA-256 values:

| File | SHA-256 |
|---|---|
| `AUTHOR.md` | `087c0eaba627d30b17d6f463166cea7aa8571db5528dd1d69391a46a37545a4b` |
| `environment/README.md` | `2ebbe907ba2daa70c29f9909efc945ac8f0e7298717794528318f9668ba1b65d` |
| `environment/model.py` | `d8c6cd4f7511c279e8fd0cc6474afcbe66d73ae8f6c8896c260bce5fbdcf0745` |
| `solution/model.py` | `beb8a5d1d6c6dd22af326e974baa0292c97f4cff847d8f1dd8615bbd8fbed5e5` |
| `tests/reference.py` | `d8bc35b5f5f8e0f46934dedc3f082468486e434ae681625c9d3fd57b18c66978` |
| `tests/test_hidden.py` | `7814f388084165fb1c371fd216c3e2f3b1525789fb0ca5689258ecf0b2969d42` |
| `scripts/quenched_spin_activity_baseline.py` | `a9c26aa06f73723d5621c5481155072061390394cac5b00eee31a4b75d3fd167` |
| `scripts/validate_quenched_spin_activity.py` | `6be17bebe460c115aaece730626890662d42f2d82976f246e916566308845276` |
| Public/private calibration | `7790a5ae53566168bf5ba385da5f1d3acced7e64dde9d5f28138608a0b2f5b78` |
| Hidden truth | `264694eee764f948cfb7dc89f515011278952312b8505e5b948d81c3201bcebe` |
| Validation report | `b2370c7f75c252623d64b63412cddbac3cb87aeda5a2598587689a38a157591d` |
