# Trapped winding r1: independent physics review

Verdict: approve the scientific package for frozen evaluation, subject to the separately run Docker controls. No blocking physics, specification, reference, or grading issue was found. I read the seventeen task files, completed shortcut, validator, and saved validation report. No task source or data was changed, no validator output was overwritten, and no model evaluation was run. The twenty reviewed file hashes are below.

## Apparatus and physical rate

The public apparatus specifies stationary full-plane overdamped dynamics, independent Cartesian Wiener increments, positive stiffness and temperature, units, and one common positive mobility. Its mobility interval `[.7,1.4]` changes only the time scale. A zero-temperature stationary point, an excluded or reflecting central disk, inertia, and finite observation cutoffs are explicitly absent. The angle is a continuous unwrapped record, with an equivalent Stratonovich line integral and an explicit warning against endpoint principal angles. Initial angle zero labels the increment and does not alter the stationary spatial preparation.

The planar diffusion avoids the exact origin almost surely. Over a finite interval its continuous radial path has positive minimum almost surely, so the angular integral exists pathwise. However, under the stationary radial density, the expectation of `1/r^2` diverges. Finite Cartesian correlation time therefore does not justify replacing the angular clock by a finite deterministic diffusion constant.

For `a=mu*k`, `D=mu*T`, and `nu=abs(lambda)`, conditional angular averaging gives the positive characteristic function

```
C_lambda(t) = E exp[-D*lambda^2 integral_0^t R_s^-2 ds].
```

Thus the absolute value in the public rate definition is harmless; there is no hidden complex-log convention. Its radial tilted generator is

```
L_nu = D*(partial_rr+r^-1*partial_r)-a*r*partial_r-D*nu^2/r^2.
```

The regular positive function `h=r^nu` has eigenvalue `-a*nu`. More specifically,

```
h^-1*(L_nu+a*nu)*(h*g)
    = D*[g''+(2*nu+1)*g'/r]-a*r*g',
```

which is a conservative radial OU generator. This identifies the principal decay rate rather than merely exhibiting an arbitrary eigenfunction. The singular origin branch is not the finite-energy branch selected by the underlying diffusion. The oracle `mu*k*abs(lambda)` is correct, including probe zero and both signs. Scaling radius by `sqrt(T/k)` removes every positive temperature from the result, explaining the temperature-independent output without discarding a physical effect.

## Completed shortcut and calibration

The mode of the equilibrium radial density is `sqrt(T/k)`. Diffusion constrained to that circle with tangential diffusivity `D=mu*T` gives a well-defined angular Brownian motion with characteristic rate `mu*k*lambda^2`. This is the completed shortcut. Its assumption is a typical-radius replacement of the singular random angular clock. It is not an incorrectly implemented eigenvalue problem or an invalid probability law.

Only probes `+1` and `-1` are calibrated. Both physical models then equal `mu*k` throughout the entire parameter and preparation domain. The positive design vector of stiffnesses makes weighted least squares strictly convex, so mobility is globally identifiable; the bounded fit is implemented correctly. There is no radial-response calibration that would contradict the circle approximation.

The 192 public and private calibration readings are byte-identical. I reproduced both fitted mobilities, `1.0899838174123069`, and every stored group error. The saved reduced chi-square is `1.048601674134213`. I also checked exact oracle/shortcut equivalence at all calibration-probe parameter corners.

## Reference and independent checks

The private reference uses the continuous-winding heat kernel, whose regular radial factor is `I_nu`, and averages both stationary radii. Its generalized-Laguerre quadrature and origin weights implement

```
C_lambda(t) = (1-rho^2) double_integral exp(-x-y)
              * I_nu(2*rho*sqrt(x*y)) dx dy,
rho = exp(-a*t).
```

The code divides out `rho^nu` using the Bessel small-argument power, then restores this known finite-time factor when taking the logarithmic slope. That normalization is legitimate and avoids underflow. The moderate-time comparison with the full characteristic function checks its prefactor, so validation is not restricted to restating the asymptotic exponent.

The stationary characteristic function is

```
rho^nu * Gamma(1+nu/2)^2/Gamma(1+nu)
       * hyp2f1(nu/2,nu/2;1+nu;rho^2).
```

Its finite positive long-time prefactor independently establishes the fixed-frequency rate. The background paper's Cauchy weak-scaling result alone would not establish that limit; AUTHOR correctly distinguishes the two arguments. See the [prior concept review](expansion-ou-winding-concept-review.md) and [Vakeroudis's primary paper](https://arxiv.org/abs/1209.4027) for the process and winding conventions.

The ordinary reference uses dimensionless times 10 and 14, then restores the physical time scale by multiplying by `mu*k`. Refinement uses later times 14 and 18 and twice the quadrature order. The saved 136 domain comparisons have maximum absolute rate error `5.796e-10`, negligible beside measurement uncertainty and the grading tolerance. The checked-in truth is a finite-time approximation to the exact rate, with that bias explicitly tested and documented.

The validator compares kernel quadrature with the stationary hypergeometric expression at finite times, and checks four independent exact Cartesian OU endpoint ensembles, each with 524,288 samples. Those checks correctly use only integer harmonics, for which full turns leave the phase factor unchanged. They are not used to infer fractional-probe winding from endpoint angles.

I additionally evaluated the positive power series obtained by expanding the Bessel kernel, using 70-digit decimal summation for its radial-series factor. Across seven probes including zero and `.001`, and times `.4,1,3,10,14`, its scaled log-characteristic agreed with the reference within `6.051e-15`. The longest sum needed 188 positive terms. This is a separate numerical check of the same kernel, not a claimed third microscopic derivation. At positive and negative probes from `1e-10` through `.1`, with the largest physical time scale, ordinary-reference absolute error was below `2.951e-12`; later-time refinement reduced it below `9.437e-16`.

## Grading and public/private separation

All 256 saved noise realizations pass calibration and parameter gates. The worst oracle group error is `0.01721%`, while the smallest diagnostic shortcut error is `38.9506%`. On checked-in data the three shortcut errors are `38.9605%`, `111.7056%`, and `57.4566%`; oracle errors are about `0.001485%`. These margins comfortably bracket the 4% relative Euclidean group threshold. Forty-one noiseless fits recover mobility within `2.221e-16`.

Diagnostic true rates range from `.218` to `4.1856` inverse seconds. The shared group includes the correctly vanishing zero-probe anchor, but no diagnostic group relies on a zero or small denominator. Hidden probes, temperatures, and stiffnesses all lie within the public apparatus domain. Positive and negative fractional and larger probes test the same stated process.

The Docker context copies only the public apparatus, starter, public test, and calibration. The private hint, true mobility, author explanation, solution, reference, and hidden truth stay outside it. The hint asks for the radial characteristic generator and origin behavior without changing the apparatus. The verifier checks frozen and declared hidden input lists, calibration, fitted parameter, output shapes, finite predictions, and group errors; it records metrics before assertions. The oracle installer changes only the model. Runtime limits and dependency pins match the existing task pattern.

This source matches the approved distinct winding concept. No difficulty claim follows from scientific controls. Trial review should distinguish a retained Gaussian-angle or typical-radius model from correct physical setup followed by mathematical, numerical, wrapped-angle, or implementation mistakes; the source and trajectory determine the actual cause.

## Reviewed source hashes

SHA-256 values identify the reviewed bytes; generated Python caches are excluded.

| File | SHA-256 |
|---|---|
| `tasks/trapped-winding/AUTHOR.md` | `659e52ec32dccb94b4763ef233469ab2548a2e161eb21a876a13d5b4cf913e40` |
| `tasks/trapped-winding/environment/Dockerfile` | `375a34ee073c6cb5cb31cf35811cf0ae93156f315efd3747575b06fb97deeb70` |
| `tasks/trapped-winding/environment/README.md` | `94c318906925aaecad5df07586be17732d690db9606c31d154671fa0812b9c5c` |
| `tasks/trapped-winding/environment/data/calibration.json` | `63092cbf355a231489b2a69f5b964c63db17193cbb4920d500ca749d4e3159b3` |
| `tasks/trapped-winding/environment/model.py` | `6e48b8cace7b8eb8425bc4a6e43babb19a5dba7fce84ec0cdcf9a89b8f1f065e` |
| `tasks/trapped-winding/environment/test_public.py` | `9bc14b7a85105baed797c4ed7c091416402989805279601a9a83be8ae6d703c9` |
| `tasks/trapped-winding/hint.md` | `226ce5e2fafb6bf32d92b35cb5b0df572830b5df0cebc825a238f0dacb2cc74f` |
| `tasks/trapped-winding/instruction.md` | `3d4882bd2b491c5af08db7bfb90d3f30b40181a35a879c7cab0c2cde06d2a3af` |
| `tasks/trapped-winding/solution/model.py` | `7826a0c4ce9f8d8ca09a7cb642ff25769aee9256831f937be5dda6fbc93f1985` |
| `tasks/trapped-winding/solution/solve.sh` | `bc0ad4cb71c36a0064d7787dcbac008e37e438e6b6ecbfca0d28c7370d9cf644` |
| `tasks/trapped-winding/task.toml` | `cbc3d931f382c98d33542e9bf5c9f2acc623d7c7cebc3a95f9fb4b46127cb6af` |
| `tasks/trapped-winding/tests/data/calibration.json` | `63092cbf355a231489b2a69f5b964c63db17193cbb4920d500ca749d4e3159b3` |
| `tasks/trapped-winding/tests/data/hidden_truth.json` | `4ac62f509c8585495fc233c691eb28260c00f0d40f51cb794d72a5208007a017` |
| `tasks/trapped-winding/tests/metadata.json` | `b9a20f300e1f93cb9193564b3af8439692e8525b3f207e92d4ba1644b7e7b661` |
| `tasks/trapped-winding/tests/reference.py` | `c9e9f1f9250dd36461cd54a45cc04e542cd9485c60145d1d1606b2c1ddd000b2` |
| `tasks/trapped-winding/tests/test.sh` | `74e0598dd777b91b4a639ded0d84e5e5a4ea688344e70a5ae18c2a4e7ff4620c` |
| `tasks/trapped-winding/tests/test_hidden.py` | `17400c4cbe5cdff85077429fb3551d1c9e64b23cf472353331f416e8c9681ca4` |
| `scripts/trapped_winding_baseline.py` | `4d38687ea41249a37644cb2fa2767c25de917bfc52c8f2918cf97a35e929cc1e` |
| `scripts/validate_trapped_winding.py` | `f8334b8fff87775631768205b6aae5274b8fc8879bd2e0b5657665842bca0aba` |
| `results/expansion-trapped-winding-r1-validation.json` | `098ccc4ec4c70cca47c5eab9f8424a3b3ebf62c0b59e41037e7d15fba1e29ccd` |
