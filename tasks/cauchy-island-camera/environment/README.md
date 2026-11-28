# Cauchy island camera

A one-dimensional carrier follows a symmetric Cauchy process. Its free increments are independent and stationary, with characteristic function E[exp(ik(X(t)−X(0)))] = exp(−D t |k|). All distances and times below use the same dimensionless units. Fit D from the supplied free-transport measurements.

## Live islands and detectors

A carrier is launched at x on a live interval. Its run ends at the first time τ that its **position** is outside the union of the live intervals. A jump from one live interval straight onto another continues the run. Only the two endpoints of such a jump matter. Let Y = X(τ) be the final landing coordinate and Z = X(τ−) the departure coordinate of this last jump. Boundary points have probability zero.

The landing camera reads a density per unit coordinate at y outside the live set. A second camera, the donor camera, responds to Z. It accepts with probability A(Z). With a selected island [l,r], its response is zero on the other island. On the selected island, put s = (Z−l)/(r−l). The response is a(1−s)² + 2b s(1−s) + c s² for the supplied profile [a,b,c]. A null donor accepts every shot.

On a given trajectory, Tᵢ is its total physical residence time on island i and N counts jumps from one live island onto another. An independent exponential gate of rate kᵢ operates during residence on that island. A relay transmits the gate with probability p on each inter-island jump. These devices record survival without altering the carrier path. Conditional on that path, their joint survival probability is

    W = p^N exp(−Σᵢ kᵢ Tᵢ).

The final exit into the absorbing exterior does not use the relay. A gate or donor-camera outcome is drawn independently of the other given the full trajectory.

| observable | reading |
| --- | --- |
| landing | Unconditional density of Y at y. Both auxiliary detectors are ignored. |
| donor | Fraction accepted by the donor camera among landings at y, E[A(Z) given Y=y]. |
| clock | Fraction with a surviving gate among donor-accepted landings at y, E[W A(Z) given Y=y] / E[A(Z) given Y=y]. |

Conditioning on Y=y means the limit of a narrowing camera bin centered at y. The landing density can exceed one. The two other readings are fractions.

## Python interface

`Model().fit(rows)` takes a nonempty list of calibration records and returns the same object with fitted attribute `diffusivity`. Each record contains `input` with `mode="free"`, `wave_number` and `lag`, and `value` equal to the free characteristic function, rounded to eight decimal places. `predict(inputs)` returns a finite real NumPy array of shape (len(inputs),), including an empty input. Free inputs request this same characteristic function.

A camera input contains `islands` as a list of ordered [left,right] pairs, `launch`, `landing`, `donor` (null or an object with zero-based integer `island` and three-entry `profile`), `rates` in island order, `relay`, and `observable`.

The first island is [−1,1]. If present, the second is [1+g,1+g+2h], with g in [0.35,0.90] and h in [0.50,1.25]. Within the launch island, the distance from its midpoint is at most 0.65 times its half-width. A landing coordinate is between one quarter and three quarters of the way across the gap, or is 0.25 to 1.10 beyond either outer edge. All profile entries are in [0.1,1], each rate in [0,1.2], relay in [0.3,1] and D in [0.75,1.2].

The starter uses exact single-island exit densities, reinjects transfers at the next island's midpoint and replaces each residence time by its single-island mean. It uses the donor response at the midpoint of the final island. You may replace this approximation. Run `python -m pytest -q /app/test_public.py`. Grading requires absolute error below 0.00002 on each calibration record, 0.006 on each landing density and 0.004 on each donor and clock fraction. Public worked checks use the same tolerances.

## Worked readings

Set D = 0.91. With only [−1,1], launch 0, y = 2, donor island 0 with [1,1/3,1/9], and rate 0.83, the readings in table order are **0.09188815, 0.38490018, 0.54381329**. The single-island landing density at launch 0 is 1/(π |y| sqrt(y²−1)). Here A(z) = (2−z)²/9, so the donor profile cancels the spatial factor 1/(2−z)² of the final jump intensity. The mean residence time is 1/D at this launch. The relay has no effect on a single island.

Add a second island [1.4,2.8]. Keep launch 0 and take y = 1.16, donor island 1 with [0.9,0.3,0.1], rates [0.8,0.33] and relay 0.7. The three readings are **0.53586148, 0.07038923, 0.31554063**. The positive donor fraction records carriers that reached the second live island before their final jump back into the gap.
