# Photon-counted phase references in a quantum repeater

A mesh of elementary optical links loads Bell pairs into memories at its stations. After loading, each station reports the parity of its number of stored qubits. Two receivers request distinct remote Bell pairs from the same loading. An auxiliary pilot counter records the number of selected π-phase heralds. Predict the probability of serving both receivers, their resource contention and a correlation of their output fringes.

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

## Phase-reference pilot counter

Every elementary herald also sends one classical pilot pulse through a balanced optical phase analyzer locked to the link's phase. The pilot occupies a separate optical mode and leaves the stored Bell pair intact. When the link's sign is +1 the pulse goes to a dump. When it is −1 the pulse goes to the counting port. The analyzer sorts the common physical link phase shared by **both colors** in that shot.

`pilot={"links":[...],"count":k}` connects the listed elementary links to one ideal photon counter. Link numbers are zero-based indices into `links`. Each detected pilot counts once. The instrument reports exactly k detected pilots from those links in total and discards their link identities. For example, three heralds on one monitored negative-phase link add three counts. An empty link contributes zero for either phase. Unlisted links do not contribute. Counts refer to every elementary herald in the loading, including pairs that the router ultimately does not use.

Condition every reading on this pilot count and on the reported station parities. For contention additionally condition on both requests being individually possible. For the fringe additionally condition on successful joint routing. The pilot counter, the routing controller and the two XX analyzers observe the same loading and the same phase signs. The path controller continues to draw its feasible ordered path pair uniformly without favoring an inferred phase.

## Worked checks

Use a=1 and t=ln 2. A single link `[0,1,t,t,-0.35]` connects stations with parity image [1,1]. Both receivers request 0→1 and the pilot counter records three on link 0. This identifies a negative link with exactly three stored pairs. Each is narrowband-compatible with probability 1/2 independently, so delivery has probability 7/8. Among individually possible loads there is no contention. Two distinct pairs from this same negative-phase link have joint fringe 1.

A second mesh has links `[[0,2,t,0,0.5],[0,1,t,0,0.6],[1,2,t,0,-0.8]]`, no station reports, and both receivers requesting 0→2. Monitor only link 0 and retain pilot count 1. The direct path now has exactly one stored narrowband-compatible pair of negative phase. Both receivers are individually possible. Joint delivery requires at least one pair on each bypass link and occurs with probability 1/4, giving contention 3/4. When the two routes differ, the mean phase product is 0.48. If each bypass link has at least two pairs, the pair of bypass routes is also feasible, with phase product 1. The resulting mean fringe is 0.48+(1−0.48)(1−t)²/3 ≈ 0.496320833.

## Interface and accuracy

Implement `Model` in `/app/model.py`. `fit(records)` receives calibration dictionaries containing `input` and `value`, stores `activity` and returns self. `predict(inputs)` returns a finite one-dimensional real NumPy array in order, including an empty array for an empty input. Graded meshes are connected simple graphs with 2–6 stations and 1–7 links. Exposures on a link are in [0,1.3], visibilities are in [−1,1], and pilot counts are integers from 0 through 3. All conditioning events have positive probability.

The public starter implements the complete two-receiver experiment with the pilot counter disconnected. It resolves compatible link resources, station parities, shared phases and per-loading route selection. Its `flow.weights` and `fringe.ensemble` helpers compute the corresponding probability and fringe without conditioning on a counted phase reference. `apparatus.py` supplies calibration, elementary resource-class weights, simple paths and the parity-conditioned distribution.

The ten calibration readings require maximum absolute error below 0.00002. The worked checks and the three held-out groups require maximum absolute error below 0.003. Each science group has seven readings at settings distinct from the examples.
