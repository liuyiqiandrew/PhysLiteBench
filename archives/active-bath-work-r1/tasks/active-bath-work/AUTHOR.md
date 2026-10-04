# Active bath work, revision 1

This backup tests a joint small-mass and short-memory limit. The position process has a correct overdamped description, but work measured over one inertial relaxation time retains the force–velocity correlation on that shrinking time scale. The public apparatus gives the entire finite-mass process, full stationary preparation, and measurement-before-limit convention. No stochastic calculus convention is hidden: the measured work is an ordinary time integral of continuous finite-mass variables.

## Oracle and completed shortcut

For general drag gamma, write M=(gamma I−B J)^−1. At fixed position-observation lag, the limiting position drift is −kM and its isotropic stationary covariance is

`C_x=[T+D/(gamma*(1+chi^2))]/k * I`.

Both supplied and physical models use this exact stationary covariance and full matrix exponential for every position readout. Both therefore match every calibration preparation for every allowed D, including nonzero signed field, signed chirality and changing memory ratio. The calibration's dependence on D is linear with a nonzero slope, so weighted least squares uniquely identifies D. The completed shortcut does not ignore memory ratio: it uses it in its physical work approximation.

At finite mass the force variance is `C_ff=D*gamma/(alpha*m) I`. The instantaneous mobility closure v=Mf predicts work over duration m/gamma equal to

`W_short=2 D gamma/[alpha*(gamma^2+B^2)]`.

This is the leading active contribution of the ordinary overdamped velocity law; the trap term contributes vanishing work on this interval. It is a coherent fast-momentum approximation, but its assumed ordering is invalid when the active force evolves on the same scale as momentum.

Let C=<v f^T>. Its leading stationary cross-moment equation is

`(gamma I−B J) C + (gamma/alpha) C (I+chi J) = D gamma/(alpha m) I`.

The measured work is `(m/gamma)*Tr(C)`, giving

`W=2 D gamma*(alpha+1)/[gamma^2*(alpha+1)^2+(gamma*chi−alpha*B)^2]`.

The task fixes gamma=1 and T=.7. The hidden dependence on the relative rotation rates produces both enhanced and suppressed work. The mean work is finite because force variance grows as 1/m while the physical measurement window shrinks as m/gamma. This is an energy measured in a specified interval, not an arbitrarily rescaled power. Trap stiffness affects finite-mass corrections but drops out of this leading work.

This mechanism is distinct from passive entropy-anomaly and stochastic-area tasks: a driven force has its own chiral dynamics and its memory shrinks together with inertia. The measured quantity is its ordinary mechanical work. Correct position diffusion, all position correlations, probability conservation and stationary positional energy do not determine this energetic observable.

## Independent reference and validation

The private reference constructs the full six-dimensional stationary covariance at three positive masses and computes the finite-mass observable before extrapolating. It uses scaled variables `(x,sqrt(m)*v,sqrt(m)*f)` to avoid diverging covariance entries. It does not contain the limiting work formula. For stationary work, it evaluates the force–velocity covariance directly; for position measurements, it propagates the full finite-mass covariance with a matrix exponential. Three-mass extrapolation cancels the first two corrections.

The validator independently reconstructs the unscaled physical equations and checks their agreement with the scaled reference. It checks mass refinement on every scored preparation and supported parameter corners, finite-mass covariance positivity, active-force variance, and mechanical energy balance. It also checks simultaneous reversal of field and chirality, the scalar unrotated relaxation result, and convergence to the instantaneous-mobility approximation as the force becomes slow relative to inertia. The last check uses alpha times the work to avoid comparing two trivially vanishing limits.

Calibration contains 192 independent position-correlation records with fixed absolute sigma .001. It is generated from the independent finite-mass reference, with seed 934171. The 256 noise-control seed is 934173. True D=.8; the public interval is [.4,1.2]. The uncertainty is fixed independently of D and the response. Public/private data copies match. Prediction error is RMS error divided by the RMS reference work within each hidden group; the tolerance is .03, far above parameter uncertainty and numerical error.

`results/active-bath-work-validation.json` records all scientific checks, noise realizations and actual local pytest runs. All 256 calibration and parameter checks pass. The oracle passes all 256 sets with maximum hidden error .000897; the completed shortcut fails all sets, with minimum hidden-group error .6784. Actual shortcut group errors are 5.297, .7756 and .6793. Independent finite-mass agreement is within 1.12e-9 on every scored preparation and tested corner, and the finite mechanical energy balance closes within 1.4e-15. The actual local oracle passes all seven tests; the completed shortcut passes interface/calibration/parameter checks and fails only the three hidden groups. `results/active-bath-work-source-provenance.json` records source hashes and the exact shared neutral instruction. Independent source review is in `results/constraint-hardening-physics-review.json`.

Reproduce with:

```
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_active_bath_work.py
```

Regeneration requires `--generate`. The completed shortcut is `scripts/active_bath_work_baseline.py`; the oracle is `solution/model.py`. The unchanged benchmark limits are 600 seconds for the agent and 60 seconds for verification. The neutral-r1 conditional screen completed with three genuine physical successes out of three: initial jVMqHXJ, followed by UCjBmas and mmUE3am. All three derived the fast velocity–force covariance and independently checked finite-mass dynamics. Their final hidden errors are all .00025156 or smaller. The initial success is included; this family does not meet the requested failure target and is archived. Native evidence remains under the jobs listed by jobs/matrix-neutral-r1-20261002-192539/index.json, with reviews in results/neutrality-brownian-trial-reviews.json.
