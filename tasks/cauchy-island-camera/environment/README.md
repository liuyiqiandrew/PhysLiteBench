# Fission coincidence camera

A one-dimensional transport channel occupies (−1,1). A live carrier moves as a symmetric Cauchy process with independent, stationary increments and free characteristic function E[exp(ik(X(t)−X(0)))] = exp(−D t |k|). Fit the transport parameter D using measurements with fission switched off.

## Creation and absorption

Each live carrier splits after an independent exponential time of rate λ per unit physical time, provided it is still in the channel. At a split at z, the parent disappears. One daughter is placed at z−d and the other at z+d. The displaced daughters start new, mutually independent transport processes and fission clocks under the same rules. Recoil distances and the channel are fixed throughout a shot. There is no generation cap.

A carrier terminates on its first landing outside (−1,1). A daughter placed outside at birth terminates immediately at her placement coordinate. If a terminal coordinate is ≤ −1−tᵣ, the red counter clicks with probability εᵣ. If it is ≥ 1+tᵦ, the blue counter clicks with probability εᵦ. All other landings are silent. A carrier generates at most one terminal click. Counter efficiencies act independently on different terminal carriers. All descendants belong to the same shot, however late they arrive.

A single source starts one carrier at x. A pair source immediately prepares two at x−d and x+d. This initial preparation does not use a fission clock. The live domain, transport and clocks then apply to each.

The chosen rates give extinction with probability one. At every interior position the rate of jumps to the exterior is at least 2D/π, which exceeds every supplied split rate.

## Camera readings

Let R and B be the total red and blue click counts over the entire family shot.

| observable | returned reading |
| --- | --- |
| red | P(R ≥ 1) |
| coincidence | P(B ≥ 1 given R ≥ 1) |
| clean | P(R = 1 and B = 1 given R ≥ 1 and B ≥ 1) |

Counts in the last row refer to the entire family. Silent terminations are allowed. Clean readings are requested only when coincidence has positive probability.

## Python interface

Implement `Model().fit(rows)` and `predict(inputs)`. The fit returns the same model and sets numeric `diffusivity`. Each calibration record contains `input` with `mode="free"`, `wave_number` and `lag` and `value` equal to the characteristic function rounded to eight decimals. Free predictions return that same characteristic function. A prediction must be a finite real NumPy array of shape (len(inputs),), including for an empty input.

A family input has `source` ("single" or "pair"), `launch` x, `recoil` d, `rate` λ, `thresholds` [tᵣ,tᵦ], `efficiencies` [εᵣ,εᵦ] and `observable`. The operating ranges are D in [0.75,1.2], x in [−0.46,0.46], d in [0.20,0.45], thresholds in [0.12,0.72] and efficiencies in [0.35,1]. Rate is either zero or in [0.12,0.36].

The supplied `apparatus.terminal(q,x)` gives exact single-carrier red and blue probabilities with fission switched off, including an exterior placement. The starter allows at most one split per initial carrier, positions the split at its initial location and estimates the split probability from its mean ungated residence time. You may replace this approximation.

Run `python -m pytest -q /app/test_public.py`. Each supplied calibration value must be reproduced within absolute error 0.00002. Every public worked value and held-out camera reading requires absolute error below 0.004.

## Worked shots

With D = 0.89, select a pair source at x = 0, d = 0.30, λ = 0, thresholds [0.25,0.25] and efficiencies [1,1]. The three outputs in table order are **0.49435347, 0.34109898, 1**. If the two daughters' red probabilities are a and b, symmetry interchanges them for blue. The red probability is a+b−ab and the joint probability is a²+b². There are exactly two terminal carriers, so every coincidence is clean.

With a single source at x = 0.23, d = 0.31, λ = 0.32, thresholds [0.18,0.42] and efficiencies [0.8,0.9], the readings are **0.29043, 0.15084, 0.66374**. This shot admits arbitrarily many descendants.
