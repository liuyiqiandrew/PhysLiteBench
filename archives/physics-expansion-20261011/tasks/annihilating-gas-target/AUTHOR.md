# Tagged survival in an annihilating gas, revision 1

The public experiment inserts one stationary tag independently of the
Poisson background. Velocity marks apply to background particles only.
All contacting pairs annihilate, including moving/stationary background
pairs. There are no finite boundaries or long-time limits in the readout.
Before the tag is first hit, the two half-lines evolve independently.

## Finite-time physical law

On the positive half-line, let A be the initial rank of the first particle
to reach the origin; A can be infinite. If a_n=P(A=n), the generating
function f(x)=sum a_n*x^n is the solution analytic at zero of

    p*x*f^4 - (1+2*p)*x*f^2 + 2*f - (1-p)*x = 0.

The first-hit renewal decomposition includes annihilations of moving
particles with both velocity classes and with stationary particles. Its
joint-law independence from the summed gaps is established by Corollary 3
of [Haslegrave and Tournier](https://arxiv.org/html/2004.09119#S2.SS5).
This independence is needed; simply assuming that a collision-selected
particle has an independent position would not justify the formula.

For exponential Poisson gaps, the nth initial position has the gamma law.
With z=rho*v*t, the one-sided hitting probability is therefore

    q(t) = sum_(n>=1) a_n * gammainc(n,z),
    S(t) = (1-q(t))^2.

There is no extra factor two in z: two independent sides account for the
square. At p=0 the exact calibration law is

    S(t) = [ive(0,z)+ive(1,z)]^2,

where ive is the exponentially scaled modified Bessel function. At t=0 or
p=1, survival is one. Every readout is a dimensionless probability.

The oracle writes f(x)=x*g(x^2), giving

    g = (1-p)/2 + (1+2*p)*y*g^2/2 - p*y^2*g^4/2.

It builds coefficients sequentially using convolutions. The coefficients
are nonnegative probabilities, but their recurrence contains a subtraction;
the validator checks nonnegativity rather than clipping values. Degree 256
suffices throughout the public z<=93.6 domain: the omitted q is at most
P(Poisson(z)>=257)=gammainc(257,z), so the survival error is at most twice
that tail, below 1.321e-43. The oracle uses the Bessel expression at p=0.

## Physical shortcut

The completed shortcut deletes stationary background particles at time
zero, leaving a Poisson gas of moving particles with density rho*(1-p).
It then solves that two-speed annihilating gas exactly, including collisions
between moving particles. Its readout is the squared Bessel sum at
z_short=(1-p)*rho*v*t. It is a valid auxiliary process, exact for the entire
calibration family and at p=1. Its error is deleting the stationary
particles' later collision histories, not forgetting moving-particle
annihilation or using a malformed rate equation.

Stationary particles can remove an incoming particle, but can also remove
a particle that would have blocked a later arrival. No pathwise monotonic
protection claim is used. The ensemble predictions require the full
collision history. This differs from ballistic hard rods, whose collisions
exchange velocities without removing particles, and from the two archived
mobile-trap revisions, whose background traps do not react with one another.

## Independent validation

The verifier reference continues the quartic's analytic root along complex
rays and extracts its coefficients by a Cauchy/Fourier integral. The standard
calculation uses 2048 contour points at radius .98; refinement uses 4096 at
.99. This is an independent numerical representation of the same physical
renewal relation, not an independent microscopic derivation.

The independent microscopic reference uses a chronological collision heap
with live neighbor links. It inserts the tag at zero and removes contacting
pairs. Only initial particles in [0,v*t] can affect a one-sided arrival by
t; their count is Poisson and their conditional positions are uniform.
The validator compares 400,000 independently drawn finite collision
histories at eight preparations with the one-sided prediction. These are
statistical checks with reported binomial uncertainty, not grading truth.

It also enumerates every velocity assignment and every permutation of
fixed noncommensurate gaps for one, three and five particles. Those exact
finite collision probabilities check the first nonzero renewal coefficients
without using the quartic. Domain checks cover random preparations,
endpoints, the critical mixture, time zero, coefficient signs, and probability
bounds. The frozen hidden values come from the contour reference.

## Calibration and evaluation

Calibration has 256 readings at 32 p=0 density/time settings, repeated eight
times. True v=1.08 mm/s, sigma=.001, data seed 612001; the 256-draw noise
study uses seed 612009. Every positive-time calibration response decreases
strictly with v because, for A(z)=ive(0,z)+ive(1,z),

    d(A(z)^2)/dz = -2*A(z)*ive(1,z)/z < 0.

Thus noiseless calibration identifies v uniquely. The validator recovers
51 speeds across the allowed interval, allowing 1e-7 absolute for the
bounded scalar optimizer's endpoint stopping error. Both completed controls
use the same known-sigma least-squares fit. The parameter gate is 3%, the
reduced calibration chi-square gate 1.5, and every prediction group's
relative Euclidean error must be below 4%. Group norms are nonzero.

Run `scripts/validate_annihilating_gas_target.py`. It reads frozen data;
`--generate` intentionally replaces public/private calibration and hidden
truth together. The report is
`results/expansion-annihilating-gas-target-r1-validation.json`.

After independent source review and fresh Docker controls, run exactly
three frozen unhinted gpt-5.6-luna High trials. Preserve and inspect all final
sources, trajectories and outcomes. Retaining the deletion/thinning model,
or another demonstrably incorrect physical closure, can establish a
physical failure. Recognizing all collision physics but failing to derive
the specialized generating function does not by itself establish one.
A correct collision simulation with sampling error, an event-queue defect,
or an incorrect coefficient/root calculation must be classified separately.
No unchanged candidate is resampled.
