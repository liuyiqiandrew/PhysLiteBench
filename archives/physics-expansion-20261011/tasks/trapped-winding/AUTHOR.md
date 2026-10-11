# Trapped winding, revision 1

Write a=mu*k and D=mu*T. Conditional on the radial path, the unwrapped
angle is a Brownian motion with clock `2D integral dt/r^2`. Consequently
its characteristic function has a positive radial Feynman-Kac kernel
with backward generator

```
L_lambda = D*(d_rr + r^-1*d_r) - a*r*d_r - D*lambda^2/r^2.
```

For nu=abs(lambda), the regular solution at the origin behaves as r^nu.
The positive ground eigenfunction is exactly r^nu, with eigenvalue -a*nu.
Thus the required rate is `mu*k*abs(lambda)`. Radial scaling removes every
positive temperature from this rate. A zero-temperature deterministic
trajectory or a finite central core is a different apparatus, explicitly
excluded from the public domain.

The completed shortcut confines the particle to the most probable
radius of its equilibrium radial density, `r_mode=sqrt(T/k)`. On that
circle, angular diffusion is `D/r_mode^2=mu*k`, giving the valid Gaussian
characteristic rate `mu*k*lambda^2`. This auxiliary process fits probes
+1 and -1 exactly for every mobility, stiffness and temperature. It fails
elsewhere because visits close to the origin create a singular angular
clock. Replacing that clock by a finite typical-radius value is the
physical approximation. The stationary mean of 1/r^2 diverges; ordinary
finite-variance angular diffusion does not follow from finite Cartesian
correlation time. An incorrect differential equation or eigenvalue
calculation should be classified separately from this approximation.

Calibration uses 24 settings repeated eight times, sigma=0.001 inverse
seconds, true mobility 1.09, and seed 615001. At |lambda|=1, the positive
response basis is k, so mobility is globally identified. The oracle and
shortcut use the same bounded weighted linear fit. Hidden inputs include
shared probes 0,+1,-1 and three groups of fractional, higher, and negative
probes. They remain within the public continuum apparatus.

The private reference integrates the angular heat kernel on the covering
space, whose radial factor is the modified Bessel function I_nu. Averaging
both stationary radii gives, with rho=exp(-a*t),

```
C_lambda(t) = rho^nu * Gamma(1+nu/2)^2/Gamma(1+nu)
             * hyp2f1(nu/2,nu/2;1+nu;rho^2).
```

The implementation evaluates the original two-radius integral using
64-point generalized Laguerre quadrature on each radius. It divides out
rho^nu to avoid underflow, then takes a logarithmic slope between a*t=10
and 14. This differs from the asymptotic rate by less than 6e-10 over the
checked domain. The validator refines to 128 nodes and times 14/18, compares
the finite-time hypergeometric series, and checks integer harmonics with
four independent 524288-sample exact Cartesian OU endpoint ensembles.
Only integer harmonics can be checked from endpoint principal angles;
fractional probes require the full unwrapped path kernel.

The ground-eigenfunction and stationary-kernel arguments establish the
fixed-frequency rate directly. Do not infer that rate solely from weak
convergence of Theta_t/t to a Cauchy variable. For primary background on
the winding process and its singular small-radius contribution, see
[Vakeroudis, sections 2,3,5](https://arxiv.org/abs/1209.4027).

Noise seed 615009 supplies 256 independent calibration realizations; the
validator checks every hidden group at every fitted mobility. It also
checks 41 noiseless fits, 136 domain cases, parity, the zero probe, and
positive-temperature scale invariance. Ordinary validation reads fixed
data; `--generate` intentionally replaces both calibration copies and
hidden truth. Prediction groups require relative Euclidean error below 4%,
parameter error below 3%, and reduced calibration chi-square below 1.5.
See the saved validation report for margins and numerical convergence.

Use `scripts/validate_trapped_winding.py` in the pinned NumPy/SciPy
environment, then separate Docker oracle and completed shortcut controls.
Only `environment/` enters the agent image. The private reference, hint,
true parameter and author explanation remain outside. Standard 600/60-second
limits remain. Freeze the revision before its three unhinted Luna High
trials and review retained source and trajectories before classification.
