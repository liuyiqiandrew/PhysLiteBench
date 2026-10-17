# Swirling heat tails, revision 1

This task tests whether a stored-energy change can be discarded when
predicting rare long-time heat fluctuations. The approximation uses the
exact circulating-force work distribution, including its non-Gaussian
tails. Work and heat have the same mean rate and the same central rate
function. Rare initial or final trap energies of order observation time
nevertheless change the heat tails.

## Physical solution

Set r=drive_ratio, k=stiffness, and tau=mobility*k*t. With coordinates in
units sqrt(T/k), the process has drift -x+r*R*x, unit diffusion, and
stationary density exp(-|x|²/2)/(2*pi). The circulating current leaves this
density unchanged. The mean heat rate is 2*mobility*k*T*r². Thus the heat
rate corresponding to relative_heat z in dimensionless time is q=2*r²*z,
and the reported rate is mobility*k times its dimensionless rate function.
T cancels after the specified normalization; it is fixed to one publicly.

Write a for the dimensionless field conjugate to Q/T. The bulk eigenvalue
of the tilted work generator is

```
f(a) = 1 - D(a),   D(a) = sqrt(1-4*r²*a*(1+a)).
```

Heat equals circulating work minus the trap-energy change. Its exponential
moment therefore has terminal factor exp(-a*|x|²/2), initial factor
exp(+a*|x|²/2), and stationary initial averaging. A radial backward ansatz
exp(c-b*|x|²/2) gives

```
b' = -2*[b²+b+r²*a*(1+a)],  b(0)=a,
c' = -2*b,
E[exp(a*Q/T)] = exp(c)/(1-a+b).
```

Convergence of the terminal solution to the stable radial branch and
convergence of the initial Gaussian integral restrict the interior domain:

```
-1 < a < a_plus,
a_plus = (1-r²)/(1+r²)                 if r² < 1/3,
         (sqrt(1+1/r²)-1)/2            otherwise.
```

The rate function is the Legendre supremum over this open domain, using
interior limits at its endpoints. The unconstrained saddle has

```
1+2*a = q*sqrt(1+r²)/(r*sqrt(q²+4*r²)).
```

Clipping it to the domain closure gives the oracle's compact expression.
For z<-1 the heat rate is -q; for sufficiently weak drive there is also a
linear positive tail. These branches correspond to rare endpoint energies,
not to a different bulk stationary eigenvalue. The Gaussian integral's
boundary singularity gives the linear tail in inverse Laplace asymptotics;
the result is not inferred solely from a differentiability theorem for an
unrestricted eigenvalue.

The exact boundary field is exceptional. At a=-1 the finite-time moment
grows as exp(2*tau), while the interior limit of f(a) is zero. The oracle
uses the interior supporting line for the rate function, not this isolated
moment value. The public observable is the finite rate I(z), not a moment
at a singular field.

The completed shortcut Legendre-transforms the same bulk eigenvalue over
its full real square-root domain. It predicts

```
I_work(q) = sqrt((1+r²)*(q²+4*r²)/(4*r²)) - q/2 - 1.
```

This is a valid work rate function, nonnegative and finite across the domain.
Its physical error is treating the difference of stationary trap energies
as negligible for every heat fluctuation. It is exact throughout calibration
z=.4,.7,1.3,1.6, for every drive and stiffness, rather than merely agreeing
to second order near the mean.

## Independent Gaussian-path reference

The private reference uses exact OU transitions over step h and evaluates
the force integral by its midpoint rule on each path segment. Write
d=exp(-h), s=1-d², and A=d*exp(r*R*h). A sampled path has a Gaussian
block-tridiagonal precision. The tilted work changes its off-diagonal
block to B=-A^T/s+a*r*R. The heat contribution additionally changes the
initial and terminal diagonal blocks by -a*I and +a*I. These signs follow
directly from the public force line integral.

All Schur pivots are scalar times I. With C=|B|² and H=(1+d²)/s, their
bulk recurrence is P_next=H-C/P. The stable pivot is the larger fixed
point, and the stationary initial and final pivots must remain positive.
The reference finds the field domain numerically from those conditions,
then maximizes the discrete rate. It never inserts the oracle's piecewise
heat-domain formula. Quadratic step extrapolation at .006,.003,.0015 gives
the continuous-path answer; refinement halves all three steps.

The stable discriminant is

```
H²-4*C = 1 - 8*a*r*d*sin(r*h)/s - 4*a²*r².
```

This avoids subtracting two terms of order h^-2. During development, a
review suggestion had coefficient 4 instead of 8 in the linear term. The
calibration-equivalence check rejected that expression before data were
generated. Both prototype and reference were corrected. No model trial or
frozen dataset used the erroneous expression.

A separate finite-chain recursion integrates every pivot, including both
ends, without using the bulk fixed point. Eight field/drive cases at
tau=24,48,96 converge to the discrete bulk moment within 2.3e-9 after
time extrapolation. Outside each derived field interval the actual
finite-chain Gaussian integral diverges. This checks the endpoint
conditions as well as the bulk eigenvalue.

## Calibration, validation, and evaluation

Calibration contains 256 readings at 32 settings, sigma=.0004, true mobility
1.08, data seed 611601. Its response coefficients are nonzero and known,
so weighted least squares uniquely identifies mobility. Both controls fit
1.079961240603 with reduced chi-square .950082. Across 256 independent
noise draws (seed 611609), maximum oracle group error is .1884%; the weakest
shortcut group remains above 11.84%. The 4% prediction gate exceeds noise
and numerical error by a wide margin. Parameter tolerance is 3% and reduced
chi-square tolerance is 1.5. Groups use error norm divided by truth norm,
without dividing by the zero rate at z=1.

`scripts/validate_swirling_heat_tails.py` reads frozen data by default;
`--generate` intentionally replaces public/private calibration and hidden
truth together. It checks 146 domain and crossover cases, with maximum
absolute oracle/reference difference 1.08e-7, reference refinement, rate
positivity, the zero at the mean, stiffness scaling, and reverse tails.
Frozen hidden truth keeps verification inexpensive. The report is
`results/expansion-swirling-heat-tails-r1-validation.json`.

After independent source review and fresh Docker controls, evaluate exactly
three frozen unhinted gpt-5.6-luna High trials. Inspect final source and
trajectories. Dropping endpoint energy or using an unrestricted tilted
eigenvalue is a physical error. A correct boundary derivation followed by a
sign, saddle, or numerical error needs a separate classification.

Archived thermal-bodies-r12 concerns finite-window heat variance. This
task instead measures regular long-time rate-function tails, where
unbounded stationary endpoint energies remain relevant on the large
deviation scale. No inertial degrees of freedom or ambiguous calorimeter
convention are implicit in the specified overdamped line integral.

Primary analytic context is [Noh and Park, Fluctuation Relation for Heat](https://arxiv.org/abs/1204.1004)
and [Noh, Fluctuations and correlations in nonequilibrium systems](https://arxiv.org/abs/1402.0193).
The independently discretized path reference checks the specialized task
formula and its two endpoint constraints.
