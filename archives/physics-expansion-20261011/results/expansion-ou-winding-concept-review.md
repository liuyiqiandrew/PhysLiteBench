# Stationary OU winding: concept review

Verdict: approve the revised concept for development if a backup is needed. The fixed-frequency decay law is correct, the stationary finite-time reference is correct, and the circle-diffusion shortcut is a valid auxiliary process. Use calibration only at `|lambda|=1`, as in the latest proposal. No task implementation or model evaluation was performed.

## Physical preparation and limit

Write `a=mu*k>0` and `D=mu*T>0`. The defining process is the stationary planar diffusion

```
dX = -a X dt + sqrt(2D) dW,
rho_stat(X) = a/(2*pi*D) exp[-a*|X|^2/(2D)].
```

Temperature and stiffness must be strictly positive. At zero temperature the stationary process sits at the origin and has no polar angle. The stationary initial radius is positive almost surely, and the two-dimensional diffusion does not hit the exact origin. Consequently a continuous unwrapped angle exists along every finite-time path almost surely. Its initial branch is irrelevant to the increment.

Specify the continuous lift of the polar angle, retaining every full turn. Equivalently, use the Stratonovich line integral `(X dY-Y dX)/(X^2+Y^2)`. Endpoint principal angles, sampling with finite angular resolution, and a reflecting or excluded disk are different measurements or dynamics. There is no core radius or additional boundary prescription in this ideal apparatus.

Conditionally on the radial path, the angular increment is centered Gaussian with variance `2D integral_0^t R_s^-2 ds`. Thus its characteristic function is the positive real number

```
C_lambda(t) = E exp[-D*lambda^2 integral_0^t R_s^-2 ds].
```

Its logarithm is well-defined. The pathwise clock is finite over each finite interval, but its stationary expectation diverges because `E_stat[R^-2]=infinity`. Replacing it by a typical-radius clock therefore misses an essential feature of the stated diffusion; there is no finite mean angular diffusivity to substitute.

## Fixed-frequency decay and independent reference

For `nu=|lambda|`, the radial tilted generator is

```
L_nu = D*(partial_rr + r^-1 partial_r) - a*r*partial_r
       - D*nu^2/r^2.
```

The regular positive eigenfunction `r^nu` has eigenvalue `-a*nu`. The singular `r^-nu` branch is outside the finite-energy realization selected by the diffusion and Feynman-Kac expectation. The resulting decay rate is `a*nu`, rather than `a*lambda^2`.

The supplied finite-time reference establishes this limit without relying solely on the eigenfunction argument. With `rho=exp(-a*t)`, stationary initial averaging gives

```
C_lambda(t) = rho^nu * Gamma(1+nu/2)^2/Gamma(1+nu)
              * hyp2f1(nu/2,nu/2;1+nu;rho^2).
```

It equals one at `t=0`, is positive, and has a finite positive prefactor multiplying `exp(-a*nu*t)` at long time. Therefore `-lim log(C_lambda(t))/t=a*nu` for each fixed frequency. The Cauchy weak-scaling result alone would not prove that statement. [Vakeroudis's primary paper](https://arxiv.org/abs/1209.4027) supplies the OU/Brownian time change, well-defined winding process, and large-time Cauchy scaling; the explicit stationary characteristic function and fixed-frequency inference here were checked directly.

After scaling radii by `sqrt(D/a)`, the same characteristic function is the two-radius integral

```
C_lambda(t) = (1-rho^2) integral_0^infinity integral_0^infinity
    exp(-x-y) I_nu(2*rho*sqrt(x*y)) dx dy.
```

This follows by integrating the regular angular-Fourier transition kernel against the Rayleigh stationary initial radius. It is valid for real noninteger winding frequencies; it is not a replacement by a principal-angle endpoint correlation. The latter agrees automatically only for integer frequencies.

I read `staging/gpt6-expansion/winding_prototype.py`. Its generalized-Laguerre weights and division by `rho^nu` implement this integral correctly. Independent checks at seven frequencies in `[.2,2]` and dimensionless times `.2,.7,2,10` gave integral/hypergeometric agreement within `2.96e-14`. Changing quadrature order from 40 to 80 changed the normalized integral by at most `2.98e-14`. Slopes formed from the integral at dimensionless times 10 and 14 agreed with `nu` within `1.72e-10`. This supplies a practical independent reference route. The normalized integral avoids exponentially small intermediate characteristic values.

## Shortcut, calibration, and novelty

The mode of the equilibrium radial density is `r_mode=sqrt(D/a)=sqrt(T/k)`. Brownian motion constrained to that circle with tangential diffusivity `D` has angular generator `a*partial_theta_theta`, so its unwrapped characteristic decay rate is `a*lambda^2`. This is a complete positive stochastic auxiliary model and a plausible typical-radius approximation. Call it the mode of the radial density, not the most probable Cartesian position, which is the origin.

At `|lambda|=1`, both rates equal `mu*k` for every positive temperature and stiffness. Varied stiffness identifies positive `mu` directly, with no parameter alias. Both true and approximate rates are independent of positive temperature by radial scaling. That cancellation is physical, not an unused missing mechanism. Frequencies `.2`, `.5`, `1.5`, or `2` give substantial nonzero differences.

The earlier idea of additionally calibrating exact radial relaxation would be inconsistent with a literal fixed-radius process. The revised winding-only calibration avoids that problem. A different approximation that retained radial fluctuations while freezing only the angular diffusion coefficient could also be defined, but is unnecessary here.

I reviewed the closest geometric-rotor, magnetic-tracer, swirling-heat-tails, and actuator-frame-current descriptions. None tests the inverse-square angular clock of a point-centered planar diffusion. Geometric rotor concerns a quantum connection and global boundary condition; magnetic tracer concerns the small-mass correction to signed mechanical area; swirling heat concerns stationary energy endpoints. Sharing an OU process or angular language does not make this a relabeling of their physical omission.

The concept is mathematically simple once the radial clock is recognized. A retained Gaussian-angle or typical-radius approximation is a physical failure. Deriving the correct singular radial problem but selecting a wrong special-function branch, confusing wrapped and unwrapped endpoint formulas during implementation, or failing a numerical cutoff must be examined separately. The apparatus should define continuous unwrapped winding plainly; it should not conceal a core convention to manufacture failure. Scientific approval does not predict the target model's pass rate.
