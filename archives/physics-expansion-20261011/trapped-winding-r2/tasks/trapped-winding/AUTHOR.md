# Trapped winding, revision 2

Revision 1 passed all three trials: each agent recovered the exact marginal
winding law. Its complete source and evidence remain in the expansion
archive. Revision 2 introduces a joint radial-occupation readout and
calibration of both marginals. The completed shortcut already has the
correct winding law. The new physical omission is the shared radial path,
not the previous fixed-radius approximation.

Write a=mu*k, D=mu*T and nu=abs(lambda). Conditional on radius, winding is
Brownian motion with angular clock `2D integral dt/r^2`. The joint
characteristic/Laplace readout therefore has positive radial expectation

```
E exp[-integral (D*lambda^2/r^2 + s*r^2) dt].
```

Its backward generator is

```
D*(d_rr+r^-1*d_r) - a*r*d_r - D*lambda^2/r^2 - s*r^2.
```

Let omega=sqrt(a^2+4*D*s). The positive ground eigenfunction is
`r^nu exp[-(omega-a)*r^2/(4D)]`, regular at the origin, and its decay rate is

```
Gamma(lambda,s) = (nu+1)*omega-a.
```

The completed shortcut replaces the two simultaneous records by their
values on independent copies of the same OU particle. This is a complete
positive joint probability model with both exact marginals; it predicts
`Gamma_independent=a*nu+omega-a`. The missing rate contribution is
`nu*(omega-a)`. Independent Cartesian noises do not make winding and radial
occupation independent: the angular clock depends on the radial path.
The public specification states that both records come from the same
particle. Neither a new boundary condition nor an undisclosed measurement
rule is introduced in hidden tests.

Calibration uses 42 settings repeated six times, with Gaussian standard
deviation 0.001 inverse seconds and true mobility 1.09. Angular-only
settings contain probes +/-0.5 and +/-1.5. Radial-only settings use three
positive radial probes. Both controls agree on these complete marginal
families for every allowed mobility. Angular settings alone identify
mobility globally through their positive linear response. Radial marginal
rates also increase with mobility. Hidden groups use joint nonzero
probes, with separate shared marginal and zero-probe anchors.

The oracle evaluates the ground-state rate. The private reference instead
integrates the full radial oscillator heat kernel over both stationary
radii. Set `H=a*sinh(omega*t)+omega*cosh(omega*t)`. In dimensionless squared
radii x,y, the finite-time joint expectation is

```
C = [2*a*omega*exp(a*t)*sinh(omega*t)/H^2]
    * integral exp(-x-y) I_nu(2*omega*sqrt(x*y)/H) dx dy.
```

The reference uses 64-point generalized Laguerre integration per radius,
absorbing the x^(nu/2) y^(nu/2) endpoint factors. It divides out the small
Bessel power and takes a logarithmic slope between omega*t=10 and 14.
The validator refines to 128 nodes and times 14 and 18, checks a positive
hypergeometric series at finite time, and compares 144 domain cases.

A separate check discretizes exact stationary Cartesian Gaussian OU paths.
For integer angular probes, endpoint angles give the winding harmonic.
Trapezoidal radial occupation is a quadratic Gaussian weight, evaluated
by a determinant and its tilted endpoint covariance. Refining 257 to 513
time points and extrapolating the quadratic error agrees with the radial
kernel within 2.5e-11 in log characteristic. This is not used to justify
fractional probes from endpoint angles; they require the continuous
covering-space kernel. The fractional-order reference and regular ground
mode are checked independently in the physics review.

Calibration seed: 615101. Noise seed: 615109. Domain seed: 615113.
The validator tests 256 independent calibration noise draws and every
hidden group at every fitted mobility, plus 41 noiseless fits. Ordinary
validation reads checked-in data; `--generate` intentionally replaces both
calibration copies and frozen truth. The group error gate is 4%, parameter
error 3%, and reduced calibration chi-square 1.5. Numerical and noise
errors are far smaller than the 40--47% shortcut gap; see the saved report.

Only `environment/` enters the agent image. The private reference, hint,
true parameter and explanation stay outside. The 600/60-second limits and
dependencies are unchanged. Validate with `scripts/validate_trapped_winding.py`
and run separate Docker oracle and completed-shortcut controls. Freeze
before three new unhinted Luna High trials. Only retention of the independent
records assumption counts as the intended physical failure; separate
algebra, branch, numerical, and implementation mistakes during review.

For primary background on the well-defined continuous OU winding process,
see [Vakeroudis, sections 2, 3 and 5](https://arxiv.org/abs/1209.4027).
The additional occupation tilt, kernel and calibration here are derived
from the stated apparatus. Cauchy weak convergence alone is not used as a
proof of a fixed-frequency or joint decay law.
