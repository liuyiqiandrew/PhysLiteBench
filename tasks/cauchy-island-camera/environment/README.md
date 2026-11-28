# Brownian excursion camera

A fluorescent carrier moves by isotropic Brownian diffusion in the upper half-plane. In the bulk its generator is D(∂²/∂x² + ∂²/∂z²), where z ≥ 0 is height. Each Cartesian bulk coordinate has variance 2Dt over time t. The floor at z = 0 reflects normally for −1 < x < 1 and absorbs on the rest of the floor. The carrier starts at height zero with horizontal coordinate x₀ in the reflecting interval. It can move over an absorbing section while still aloft.

Calibrate D from unconfined bulk contrast measurements E[cos(k(X(t)−X(0)))] = exp(−D k² t). D also sets the physical clock of a camera shot.

## Detectors

Let τ be the physical absorption time and Y the horizontal absorption coordinate. Let g be the last floor contact before absorption. Thus the carrier departs from the reflecting floor at time g, stays aloft during (g,τ), and lands on the absorbing floor at τ.

The final arch has height M = max z(t) on [g,τ] and physical duration T = τ−g. Its preceding history reaches height A = max z(t) on [0,g]. The cameras read the undisturbed path after each shot. Conditions involve that same shot. A landing camera at y reports the limit of a narrowing bin centered at y.

Write F for h₀ < M < h₁, P for A < p, and W for t₀ < T < t₁. A null p accepts every preceding path.

| observable | reading |
| --- | --- |
| final | P(F given Y = y) |
| history | P(P given Y = y and F) |
| duration | P(W given Y = y, F and P) |

Times are in bulk physical units. Endpoints of the time and height bands have zero probability.

## Python interface

`Model().fit(rows)` returns that instance with numeric attribute `diffusivity`. Each calibration record contains `input` with `mode="free"`, `wave_number` and `lag`, and `value` equal to the bulk contrast rounded to eight decimals. Free predictions return the same contrast. `predict(inputs)` returns a finite real NumPy array of shape (len(inputs),), including empty inputs.

Each camera input contains `launch` x₀, `landing` y, `heights` [h₀,h₁], `prior` p or null, `window` [t₀,t₁], and `observable`. The operating ranges are D in [0.75,1.2], x₀ in [−0.65,0.65], and |y|−1 in [0.25,0.90]. The lower height is zero or in [0.22,0.65]. The upper height is in [0.80,1.65]. An active preceding height p is in [0.45,1.35]. The time bounds are in [0.05,0.24] and [0.45,1.25], respectively.

The supplied `arch_below(displacement,height)` and `arch_time_cdf(displacement,time,diffusivity)` give the exact marginal height and time CDFs for a single floor-to-floor Brownian arch with prescribed horizontal displacement in the half-plane. The starter treats the full shot as one arch starting at x₀ and uses its unselected marginal duration. You may replace it. Run `python -m pytest -q /app/test_public.py`. Each free calibration record requires absolute error below 0.00002. Each worked and held-out camera reading requires absolute error below 0.004.

## Worked shots

With D = 0.89, x₀ = 0, y = 1.5, heights [0,1], prior 1 and time window [0.1,0.65], the three readings are **0.37593348, 0.65709984, 0.79144132**. The product of the first two readings, 0.24702583, is the chance that the complete path stayed below 1 given this landing. It is the ratio of the landing densities with and without an absorbing ceiling at 1.

At the same D, use x₀ = 0.32, y = 1.4, heights [0.35,1.1], prior 0.76 and window [0.07,0.85]. The readings are **0.50750033, 0.67360928, 0.91330529**.
