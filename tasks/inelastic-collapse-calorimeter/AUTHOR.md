# Inelastic collapse calorimeter

Under a white-noise force the wall-return speed has a broad distribution, and collision number and laboratory time are different clocks. This wall chooses its restitution using both the flight duration and incoming speed. The wall heat and elapsed time refer to the same sequence stopped at the first accumulation of impacts. The model explicitly absorbs a bead at that accumulation. The underlying constant-restitution transition and its survival distribution are treated by [Cornell, Swift and Bray](https://doi.org/10.1103/PhysRevLett.81.1142) and [Swift and Bray](https://doi.org/10.1103/PhysRevE.59.R4721).

## The joint return law

Write R=v/u and a=u²/(Dτ). McKean's first-return density with respect to dR da is

    p(R,a)=√3 R/(2π) exp[−(1−R+R²)a] erf(√(3Ra)).

The positive flux of the unrestricted Gaussian phase-space propagator gives the same expression without the error function. The verifier's independent physics check recovers the difference by integrating over the first earlier impact and then freely propagating from its incoming negative velocity to the observed crossing. This renewal identity tests both the half-range boundary and the time convention. The published random-acceleration results also give the incoming speed marginal in the public problem. [Burkhardt, First-passage and extreme-value statistics for a particle subject to a constant force plus a random force](https://arxiv.org/abs/0807.1671)

Put γ=1−R+R², κ=3R/γ and x=γa. Conditional on R the density of x is exp(−x)√((1+κ)/κ) erf(√(κx)). The slow gate cuts it at x=γ/[D threshold (1+R²)]. Integrating below that limit gives the selector probability. Strong Markov independence and Brownian scaling make successive pairs (R,a) independent and identically distributed. Launch speeds multiply by rR. Their log increments have mean π/√3+E[log r]. A negative mean contracts the speeds exponentially in collision number. The flight duration has finite log moment, so the sum of durations converges almost surely. A nonnegative mean fails to contract, and the sum diverges almost surely. Equal restitution reduces to the known critical value. The losing side is assigned zero for the collection observable.

For the sticking-side answer F(u), integrate over (R,x) the product of F(urR) and

    exp[−u² clock γ/(D x) − (dose/2) u² R²(1−r²)].

The analytic implementation integrates the two coating regions separately. It chooses log-R quadrature nodes so the next launch speed lands on its logarithmic mesh for each coating. The resulting linear system has boundary values one at tiny launch speeds and zero at very large speeds. The audit checks a larger interval, finer integration and the exact scaling relation between the two monitor rates and launch speed.

## Independent measurements

A positive collision sampler generates the reference values. R³ is beta-prime with parameters 5/6 and 1/6. Conditional on R the Laplace transform of a factors as γ/(γ+s) times √[(γ+3R)/(γ+3R+s)]. Draw a as the sum of an exponential with rate γ and an independent gamma with shape 1/2 and rate γ+3R. This samples a complete physical return directly. It uses neither the integral-equation discretization nor a finite time step for the white noise. The selector references use 4,194,304 contacts per threshold. Each collection program uses 2,097,152 independent bead histories, with a largest standard error below 0.00028. The frozen probabilities are rounded to six decimal places.

The sampler stops following a history if all remaining readings are below exp(−36), or if its relative launch speed crosses exp(−65). For p=1/4 the last truncation has absolute error bounded by u^p exp(−65p) M/[1−E[(rR)^p]], where M bounds the p/2 moment of one scaled timer plus heat load. This follows by subadditivity and independence of future returns. R has moment E[R^p]=1/[2cos(π(1+p)/3)]. The conditional sampling representation bounds the time moment by Γ(1−p/2)(1+E[R^p]). Each tested program has a tail bound below 0.000006.

The completed control uses the exact feedback choices, rebound ratios, sticking side and heat transfer. It draws the elapsed time sent to the timer independently from the correct flight-time marginal at that launch speed. It passes the selector, sticking and heat-only checks. Its errors on timer and simultaneous time/heat selection are 0.1244 and 0.0894. The oracle matches all frozen measurements within 0.00075 against a tolerance of 0.004.

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_inelastic_collapse_calorimeter.py` from the repository root. Add `--generate` to regenerate the independent sampled measurements.
