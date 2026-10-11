# Pulsed kinetic gas, revision 1

This candidate tests memory of thermal velocity across a spatial impulse. The
single-impulse density calibration identifies friction but cannot test that
memory. A second spatial impulse acts on the distribution left by the first.
The resulting density can recover correlations after the initial density
modulation has largely disappeared. All pulses, bath dynamics, initial
conditions, and detector conventions are public. There is no hidden force or
change of preparation.

The completed approximation keeps the exact deterministic relaxation of both
velocity impulses and the exact random displacement distribution on either
side of the second impulse. It takes those two thermal displacements to be
independent. Physically, it renews the thermal part of velocity at that instant,
while retaining the first impulse's deterministic contribution. That is a
well-defined auxiliary Gaussian process, but not the continuous bath contact
in the apparatus. Its density predictions and nonlinear mode mixing are
nonzero. This is a completed physical approximation, not a truncated fit.

## Derivation

Write `t=delay`, `u=after`, `s=thermal_speed`, and `g=friction`. Define

```
A(t) = (1-exp(-g*t))/g
V(t) = 2*s**2*(g*t-1+exp(-g*t))/g**2
C(t,u) = s**2*A(t)*A(u).
```

If `Y` and `Z` are the unforced thermal displacements on the two intervals,
their variances are `V(t)` and `V(u)` and their covariance is `C`. Linear
Langevin propagation gives the exact trajectory map

```
x1 = x0 + Y + a1*A(t)*sin(k1*x0+phase1)
x2 = x0 + Y + Z + a1*A(t+u)*sin(k1*x0+phase1)
     + a2*A(u)*sin(k2*x1+phase2).
```

The second kick does not reset the incoming thermal velocity. The covariance
can be derived by integrating `s**2*exp(-g*abs(r-r'))` over the adjacent time
intervals, or from the shared initial velocity and shared Wiener increments.

For the complex Fourier moment `E[exp(i*m*x2)]`, expand the second sine kick
using Jacobi–Anger. Sum over integers `n` for which `ell=-(m+n*k2)/k1` is an
integer. With `q=m+n*k2`, each summand is

```
J_n(m*a2*A(u)) * J_ell(a1*(m*A(t+u)+n*k2*A(t)))
* exp(i*(n*phase2+ell*phase1))
* exp(-(q*q*V(t)+m*m*V(u)+2*q*m*C(t,u))/2).
```

The real and imaginary parts are the two public detector components. The
oracle sums `n=-64,...,64`. The maximum second Bessel argument is below 10
throughout the public domain, so the omitted tail is negligible. The
approximation sets only `C=0`. The private reference instead integrates the
trajectory map directly over initial position and a Gaussian displacement,
analytically averaging the remaining conditional Gaussian. It computes
covariances by quadrature of the white-noise response kernels and uses no
Bessel functions or mode-selection rule.

## Calibration and scientific checks

Calibration uses `a1=0` and `mode=k2=1`. Only the term `n=-1` survives;
`q=0`, so the covariance drops out exactly at every fitted friction. The
response is `-J1(a2*A(u))*exp(-V(u)/2)*exp(-i*phase2)`. There are 160 noisy
readings at 80 settings, with fixed instrument sigma `0.0003` and seed
611017. The true friction is 0.17. Short-delay calibration is monotone over
the full allowed friction interval; the validator checks that property and
both controls' identical calibration functions.

The hidden groups retain single-impulse anchors and vary delay, spatial
phases, impulse signs, wave numbers and detector quadrature. The gate is
group RMSE divided by the reference group's RMS, below 4%. Calibration uses
reduced chi-square below 1.5 and parameter error below 3%. No hidden tolerance
depends on an agent's performance.

Run scientific validation with:

```
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 \
  --with pytest==8.4.2 python scripts/validate_pulsed_kinetic_gas.py
```

Use `--generate` only to intentionally replace both calibration copies and
the hidden reference values. The validator recomputes frozen hidden values,
checks doubled reference resolution, tests 72 random domain/parameter cases,
and fits 256 independent noise datasets (seed 611019). The report is
`results/gpt6-pulsed-kinetic-gas-r1-validation.json`.

## Evaluation and interpretation

The target is three fixed, unhinted `gpt-6-luna` trials at high reasoning
effort, with the standard 600-second agent and 60-second verifier limits.
Qualification requires all three completed submissions to fail for reviewed
physical reasons. A wrong sign, numerical integration failure, unfinished
fit, or covariance mistake made after correctly deriving the continuous
Langevin process is not the intended failure. Review the trajectory as well
as the final source, and test a same-parameter covariance repair when the
submitted model admits one. Keep every exploratory outcome.

This is not the collisionless-screening task: it has no self-consistent
electric field, susceptibility or causal pole. It is not single-file-memory:
particles do not interact, and nonlinear spatial impulses couple a density
measurement to hidden velocity structure. Both nevertheless test the
importance of retaining unresolved dynamical correlations.

Classical phase-mixing echoes have primary background in O'Neil and Gould,
[Temporal and spatial plasma wave echoes](https://authors.library.caltech.edu/records/qddf6-xjm06).
Only the general rephasing mechanism motivates this apparatus. Its
noninteracting Langevin gas, Gaussian calculation, parameters and calibration
are constructed here; plasma collective effects are not imported.
