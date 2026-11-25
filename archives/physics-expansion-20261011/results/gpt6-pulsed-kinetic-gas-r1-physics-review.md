# Pulsed kinetic gas r1 physics review

Independent review, 2026-10-11. **Approved for evaluation; no blocking physics, interface, or data-separation issue found.** This approves the scientific controls, not a claim about Luna failures. I read the public apparatus, starter, complete shortcut, solution, independent reference, both tests, image configuration, runner configuration, and validator. I also inspected the completed [validation report](gpt6-pulsed-kinetic-gas-r1-validation.json) and ran additional analytical-limit and identifiability checks without editing the task.

## Apparatus and preparation

The public specification fixes the initial position and velocity law, the Langevin convention, pulse ordering and finite amplitudes, bath continuity, both detector quadratures, units, unknown interval, and every input range. In particular, a velocity impulse is added to the incoming velocity; no reset is authorized. The two pulses may be coincident or absent, and those cases have definite meanings. The hidden cases obey these definitions. There is no undeclared self-consistent plasma field or collision model.

The unknown rate is shared across shots, while each known thermal speed fixes both initial velocity variance and bath forcing strength. This is sufficient to derive the requested ensemble without assuming equilibrium after either kick. The statement that calibration uses single impulses is accurate: all calibration records have `a1=0`, with a nonzero second pulse.

## Derivation and signs

Let `A(t)=(1-exp(-gamma*t))/gamma`. Direct integration of the velocity equation gives thermal displacement increments `Y` before and `Z` after pulse two, with

    V(t) = Var(increment of length t)
         = 2*s**2*(gamma*t-1+exp(-gamma*t))/gamma**2
    D = Cov(Y,Z) = s**2*A(delay)*A(after).

The first pulse contributes `a1*A(delay+after)*sin(k1*x0+phase1)` to the final position, while its displacement at pulse two uses `A(delay)`. The second pulse therefore samples the position already changed by the first pulse and by `Y`.

The implementation expands `E[exp(+i*mode*x)]`, so its real and imaginary parts are exactly the public cosine and sine readouts. Expanding the second kick with integer `n` leaves coefficient `q=mode+n*k2` on the pre-pulse noise and requires `ell=-(mode+n*k2)/k1` to be an integer. The first-kick Bessel argument, both phase factors, and the Gaussian variance `q*q*V(delay)+mode*mode*V(after)+2*q*mode*D` then follow directly. The source's signs and divisibility mask are correct.

The parameter interval excludes zero friction, so the displayed formulas need no singular-limit branch. Delay zero gives `V=D=A=0` directly. The fixed Bessel cutoff is safe on this domain: the second-kick argument never exceeds about 9.16, while omitted orders start at 65. Since the remaining Bessel factor and Gaussian characteristic factor have magnitude at most one, the ordinary large-order Bessel bound makes this tail negligible uniformly in the stated inputs.

## Shortcut and calibration

The shortcut is a completed implementation of the same nonlinear impulse map with independent Gaussian thermal displacement increments. It retains each exact increment marginal and deterministic relaxation from the first kick, but discards the velocity information linking the intervals. This is a physical factorization assumption; it is not a truncation of nonlinear pulse mixing, a zero-output placeholder, or a broken fit.

When `a1=0`, the first-kick Bessel factor requires `ell=0`, hence `q=0`. Every dependence on `V(delay)` and `D` disappears. Oracle and shortcut are therefore identical throughout the full friction interval for all calibration preparations, not just at the true parameter. Their fitted friction is `0.1701303773` and reduced calibration chi-square is `0.9769763`.

The factorized model also incorrectly changes a first-only pulse prediction if a positive nominal second-pulse time is inserted. This is an additional consequence of discarding velocity memory, not an ambiguity in the public experiment. No diagnostic hidden group relies on that absent-pulse defect: the discriminating groups use two nonzero pulses. The hidden single-pulse anchor uses the same second-only preparation as calibration.

The short-delay calibration curves are strictly monotone on the validator's friction grid. As an additional check, I generated noiseless calibration at nine true rates spanning both bounds and inspected a 161-point objective profile for each: all decreased to one minimum and increased afterwards. Bounded recovery differed from those truths by at most `9.58e-9`, including endpoint optimization error. This supports identification beyond the one frozen dataset.

## Independent reference and checks

The reference directly integrates the Langevin trajectory over uniform initial position and a Gaussian displacement. It computes velocity-noise variance and cross covariance from forcing integrals, then analytically averages the remaining conditional Gaussian displacement. It has no Fourier-Bessel expansion and does not copy the oracle's closed variance/covariance formulas. Shared use of the exact deterministic impulse relaxation is necessary apparatus physics, not a duplicated prediction algorithm. The nonnegative conditional-variance clamp only handles roundoff in an analytically nonnegative quantity.

The completed validator checks 72 random cases at three friction values against refined independent integration. Maximum absolute discrepancy is `1.32e-16`. Hidden quadrature refinement changes any result by less than `2.8e-17`, and checked-in truth agrees with recomputation. These are sampled domain checks; they are not an exhaustive enumeration of continuous inputs.

I additionally checked first-only pulses against their one-pulse closed form, simultaneous equal-wavenumber pulses against one pulse with the summed complex amplitude, and spatial reflection of both detector quadratures. Across friction bounds, wavenumbers and detector modes 1–3, zero/maximum delay, and minimum/maximum readout delay, the largest analytical-limit discrepancy was `8.33e-17`; reflection was exact. Increasing the oracle cutoff from 64 to 128 changed none of these extreme cases. These supplemental checks used an already installed NumPy 2.5.2/SciPy 1.18.0 environment; the task retains its pinned image versions.

## Tolerances and separation

Prediction error is a group's Euclidean residual norm divided by the true norm. The group RMS signals are `0.2121`, `0.03953`, `0.03234`, and `0.05559`. Some individual components are small, but no group is normalized by a nearly zero response.

Across all 256 independent noise datasets, the largest fitted-parameter error is `0.6836%` against the `3%` gate, the largest calibration chi-square is `1.45937` against `1.5`, and the largest oracle group error is `0.3324%` against `4%`. The minimum shortcut error in the weakest discriminating group is `66.43%`; the other two exceed `111%` and `165%`. The separation comfortably exceeds calibration uncertainty and numerical error. Both complete controls have the same likelihood, so the shortcut's hidden failures cannot be attributed to fitting bias.

## Separation and evaluation conditions

The Docker context explicitly copies only public README, model, test, and calibration data. The true friction, hidden inputs and truth, reference, solution, and author material remain outside the ordinary image. Public and private calibration copies are byte-identical. The instruction explicitly permits replacing all implementation code while preserving the API. The 600-second agent and 60-second verifier limits are unchanged; hidden truth avoids repeated expensive reference integration during verification.

A later trajectory must show that the agent retained an inappropriate physical factorization or reset assumption before its failure counts toward the requested result. Recognizing the continuous velocity process and then getting a covariance integral, Fourier phase, Bessel index, or implementation wrong is a different diagnosis. Fresh Docker controls and three frozen `gpt-6-luna` high trials remain the parent task's responsibility.

Reviewed central source SHA-256 values:

| File | SHA-256 |
|---|---|
| `environment/README.md` | `e6400859500fbfdfa60b21087e7271ed51ac37918cd2b226555a77b57a4a3bdb` |
| `environment/model.py` | `6a7bd869b59ffb6edae54691984e06ca5149cdf0ea5b4807e0a1e994ab68b399` |
| `solution/model.py` | `4a9ab15a6bd9fd0bb6771a18aeb8c8f45706c74a67f466852bf700ce8f3bb699` |
| `tests/reference.py` | `e5d90d50cb0ffc1531209050d7dd41d36e122920c476a1ecd80ed505821308c4` |
| `tests/test_hidden.py` | `13ed13737e4f0c076062eb9453a59fd7bf904cd8befff2c80407c4b8cb78295d` |
| `scripts/pulsed_kinetic_gas_baseline.py` | `94e211b7510d4a46eeb529e5e76d5adb66f6ee0afd4f7df48d51c16931276c94` |
| `scripts/validate_pulsed_kinetic_gas.py` | `46c768c6f725d1b2358f148908b0570148833ca01b38bd92f9eb67e3e16835e2` |
| `environment/data/calibration.json` | `661f2d1522ce760f05925e2705a222ec4f075d04392c86dc24ac78275e935295` |
| `tests/data/hidden_truth.json` | `65fec4e6bdc029d96582553f810423aa77fd8cce182c9424e0c0c0707938b48f` |
| Validation report | `ae507594af5154b6bbd929baf5c46c22f12d404b77a4c18f6683d13e56149e32` |
