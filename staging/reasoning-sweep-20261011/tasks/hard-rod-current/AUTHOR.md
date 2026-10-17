# Hard-rod current, revision 2

Revision 1 passed one of three Luna High trials. Its two failures froze the
current-conditioned crossing rates; the successful solution recovered the
full monodisperse law. That revision, every outcome, and both causal repairs
are preserved in `archives/physics-expansion-20261011/`.

Revision 2 changes the apparatus to a binary length mixture with equal
velocity fractions. The completed shortcut already solves the exact
monodisperse current law, repairing revision 1. Its new assumption replaces
the length of the net crossed block by its ordinary mean before averaging
the exponential moment. Large currents favor atypically short crossed rods.
This is a distinct composition fluctuation, not a repeat of the old batch.

## Physical derivation

For mean length a and contrast d, individual lengths are a*(1-d) or a*(1+d).
Lengths stay attached to ordered rods while equal-mass collisions exchange
velocities. Define center compression x_n=y_n+A_n with
A_(n+1)-A_n=(a_n+a_(n+1))/2. The ordered y coordinates are the sorted free
trajectories of a marked Poisson gas with density c=rho/(1-rho*a).

The half-and-half ring preparation has fixed total occupied length. Taking
the thermodynamic limit first makes each finite ordered block fair binary
length draws, independent of the free exponential gaps and velocity marks.
The public exponential moment averages over these freshly drawn orders
before taking its logarithm. Lengths are fixed within each trajectory.
Conditioning the moment on one typical infinite order would give a different
answer. No length is transported with a passing free velocity label.

For positive current j and long-rod fraction f in the net crossed block,
the empirical mean length is b=a*(1-d+2*d*f). The compressed cut moves at
b*j. Free right/left crossings have rates c*(v-b*j)/2 and c*(v+b*j)/2.
Their Poisson flow cost must be combined with j*D(f||1/2), because about
j*t ordered labels are net crossed. The number of labels in this cost is
not the total number in a ballistic observation interval.

Minimizing over composition and taking the current Legendre transform gives

```
abs(lambda) = s+c*a*sinh(s)-logcosh(c*a*d*sinh(s)),  s>=0,
F(lambda) = c*v*(cosh(s)-1).
```

The optimum mean crossed length is a*(1-d*tanh(c*a*d*sinh(s))). The field
equation has derivative one plus c times this positive length times cosh(s),
so its root is unique. The saddle current lies inside the allowed free
crossing cone. Reflection gives negative fields. The oracle evaluates
cosh(s)-1 as 2*sinh(s/2)^2 and logcosh with logaddexp.

Mean current is zero and variance growth is rho*v*(1-rho*a), independent
of contrast. More precisely, with h=1-rho*a,

```
F(lambda) = rho*v*h*lambda²/2
            +rho³*v*a²*d²*h*abs(lambda)³/2 + O(lambda⁴).
```

The limiting SCGF is generally not three-times differentiable at zero.
Finite second differences have an O(step) correction. This is a physical
effect of averaging a random contiguous block whose size scales with
absolute current, not numerical noise.

The shortcut sets the composition to its global mean and uses
abs(lambda)=s+c*a*sinh(s), F=c*v*(cosh(s)-1). It is the full, valid
monodisperse gas at mean rod length. It agrees with every equal-length
SCGF and every mixture variance used in calibration, while omitting the
additional initial-composition cost under finite current conditioning.

## Independent reference

The verifier reference minimizes the explicit sum of two Poisson relative
entropies plus j*D(f||1/2) over f at each trial current, then maximizes
abs(lambda)*j-I(j). It enforces b*j<v and does not use the oracle field
equation. Refinement tightens both scalar optimization tolerances.

A second reference builds finite-time center-current tails. Label the last
initial free point left of zero by index zero and set A_0=0. The rank event
is J(t)>=k exactly when Q_t(-A_(1-k))>=k. For positive k, n=k-1 compression
steps contain n-1 full lengths and two half endpoint lengths (n=0 has zero
offset). Their distribution is a Binomial(n-1,1/2) interior and an independent
Binomial(2,1/2) half-weight endpoint sum. At each resulting cut z, Q is a
Skellam variable with means c*max(v*t-z,0)/2 and c*(v*t+z)/2. A noncentral
chi-square CDF evaluates its positive tail. Very rare tails use an
exponentially tilted sum of two Poisson probabilities, evaluated near their
modes. A Chernoff bound excludes only weighted contributions smaller than
exp(-80) of an already accumulated term. This avoids probability underflow
being magnified by the exponential current weight.

Summing these tails gives the moment of exp(lambda*max(J,0)), whose
positive-field SCGF equals F because F>0 and clipping the negative-current
part changes a moment by at most one. The finite sum retains every possible
positive center current. Time windows 16,32,64,128 check the limit for
four nonzero-field preparations. Small-field prototype cases converge more
slowly; no universal exact 1/t correction is claimed.

The initial noncentral-CDF-only version passed the shorter windows but its
t=128 high-field check missed the limit by .000305 because of extreme-tail
underflow. The tilted Poisson sum corrects this supplemental reference;
task data, oracle, calibration, and grading did not change. No model trial
used the initial version.

The anchored free-origin construction is related to the stationary physical
origin by the Poisson free-gap/Palm split. Bounded endpoint lengths and
bounded reweighting of finitely many length marks preserve the speed-t
rate; bounded-window Poisson counts have superexponential order-t tails.
This accounts for the size-biased origin gap and its local length labels.
It does not assume that an arbitrary random order-one correction is harmless.

## Validation and evaluation

Calibration has 240 readings at 48 settings: 24 monodisperse SCGFs and
24 mixture variances, repeated five times. The common speed is 1.12,
sigma=.001, data seed 611701, and noise seed 611709 for 256 draws. Every
readout is linear in speed, and nonzero coefficients identify it uniquely.
The group relative-norm prediction threshold is 4%; parameter tolerance
is 3% and reduced chi-square tolerance is 1.5. No individual near-zero
SCGF is used as an error denominator.

Run `scripts/validate_hard_rod_current.py`. It reads checked-in data;
`--generate` explicitly replaces both calibration copies and hidden truth.
It checks 120 domain/corner cases, zero bias, reflection, variance, the
mixture enhancement, nested optimization refinement, and four microscopic
mixtures. The report is `results/expansion-hard-rod-current-r2-validation.json`.

After independent source review and fresh Docker controls, run exactly
three frozen unhinted gpt-5.6-luna High trials. Inspect all final source and
trajectories. Replacing the crossed composition by its typical value is a
physical failure; a correctly derived composition cost followed by an
optimization, sign, or coding error must be classified separately. Keep
revision 1's 1/3 result visible and never resample an unchanged revision.
