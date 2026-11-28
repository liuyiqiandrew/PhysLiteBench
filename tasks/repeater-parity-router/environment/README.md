# Two receivers on a parity-heralded quantum repeater

A mesh of elementary optical links loads Bell pairs into memories at its stations. After loading, each station reports the parity of its number of stored qubits. Two receivers request distinct remote Bell pairs from the same loading. Predict the probability of serving both receivers, their resource contention and a correlation of their output fringes.

## Preparation and station readout

The stations are numbered 0 through n−1. Each row in `links` has the form `[u,v,h,f,visibility]`. On this undirected link the numbers of narrowband-compatible pairs H and other pairs F are independent Poisson variables with means a h and a f. Counts on different links are independent. Every successful source event stores one qubit at each endpoint. All stored qubits survive the routing interval and each stored pair is individually addressable. Links can hold several distinct pairs.

At station i, the recorded parity is the total number of stored qubits from all incident links modulo two. In `parities`, 0 means even, 1 means odd and null means the station did not report. Every science reading is conditioned on the entire supplied parity image. An even reading includes all even occupations. A separate calibration measures the probability of loading zero pairs from a single illuminated source at the stated `exposure`; its mean pair count is a times that exposure. The common dimensionless factor a lies in [0.75,1.15].

## Two-pair delivery

`requests=[[s₀,t₀],[s₁,t₁]]` gives the labeled endpoint pairs. On each shot the routing controller knows the full collection of elementary heralds. To make a remote Bell pair along a simple path, it consumes one stored pair on every link of the path and performs ideal entanglement swaps at the internal stations. Stations have sufficient memory, deterministic swapping and ideal feed-forward corrections.

Receiver 0 requires a narrowband-compatible pair on every hop. Receiver 1 can use either type at each hop through a lossless converter. Routes may share stations. Routes sharing a link must use two distinct stored pairs there, with at least one narrowband-compatible pair for receiver 0. Requests can share endpoints or request two pairs between the same endpoints. Both transmissions succeed on a loading if at least one feasible *ordered pair of simple paths* exists. The two paths are chosen jointly after the heralds are available.

- `observable="delivery"` asks for the probability that both pairs can be delivered from the same loading.
- `observable="contention"` restricts to loadings where each request can be served when tested alone on that loading. Return the fraction of those loadings where the two deliveries together are impossible.

## Output fringes

Each link e has a binary optical phase sign σₑ with E[σₑ]=`visibility`. The sign is independently redrawn for each link and each shot. All pairs generated on a given link in that shot share its sign, including different colors and different memory slots. Signs are independent of herald counts. A stored pair is in (|00⟩+σₑ|11⟩)/√2. Ideal swapping and the known Pauli corrections give a remote pair whose phase sign is the product of the signs on its path. Its XX fringe is the expectation of the Pauli product X⊗X on its two qubits.

For `observable="fringe"`, retain the jointly successful, parity-selected loadings. On each retained shot the controller draws one feasible ordered pair of station paths uniformly. Each ordered pair of paths has one entry in this draw, however many distinct local memory slots could implement it. The controller does not see the phase signs. Return the expectation of the **product of the two remote XX observables**. Each retained shot contributes one joint measurement.

## Worked checks

For these checks use a=1 and write t=ln 2. A single link `[0,1,t,t,-0.35]` connects two stations with parity image [1,1]; both receivers request 0→1. The recorded total is odd. Individually both can be served whenever at least one narrowband-compatible pair is present. In this ensemble a single narrowband pair cannot serve both. The three readings are

| Delivery | Contention | Joint fringe |
| --- | --- | --- |
| 3/5−8t/15 ≈ 0.230321504 | 8t/9 ≈ 0.616130827 | 1 |

For a looped example use links `[[0,2,t,0,0.5],[0,1,t,0,0.6],[1,2,t,0,-0.8]]`, no station reports, and both requests 0→2. A receiver can use the direct link or the two-hop bypass. If the two receivers use different routes the averaged joint fringe is −0.24. When both take the same route it is 1. The delivery, contention and fringe readings are 0.251839639, 0.597056578 and 0.463835506. The direct path supports N₀ copies and the bypass supports min(N₁,N₂), where the three independent counts each have mean t.

## Interface and accuracy

Implement `Model` in `/app/model.py`. `fit(records)` receives calibration dictionaries containing `input` and `value`, stores `activity` and returns self. `predict(inputs)` returns a finite one-dimensional real NumPy array in order, including an empty array for an empty input. Graded meshes are connected simple graphs with 2–8 stations and at most 10 links. The two flux exposures on a link are in [0,1.3], and visibilities in [−1,1]. Every conditioning event in the readings has positive probability.

The public apparatus evaluates each receiver's individual delivery probability and conditioned single-pair fringe. It provides the posterior distribution of link count parities given the station reports, the four elementary link resource classes, and enumeration of simple paths. The starter gives the two receivers independent parity-selected loadings.

The ten calibration readings require maximum absolute error below 0.00002. The worked checks and the three held-out groups require maximum absolute error below 0.003. Each science group contains seven readings at settings separate from the two examples.
