# Two-height imaging of terrace bunches on a cylinder

A rough crystal surface carries N directed steps. At longitudinal height h each step has a continuous azimuthal coordinate modulo the circumference L, increasing clockwise. Before conditioning, their continuous lifts are independent Brownian motions with displacement variance 2γh. The lateral and longitudinal length units are fixed. The same diffusivity γ in [0.75,1.15] applies to every preparation.

An open calibration surface has one step with a free upper end. For `mode="wandering"`, `height` is H in [0.3,6.5]. The measured `value` is the mean squared unwrapped displacement, rounded to eight decimal places.

A pinned surface starts at the lower coordinates in `bottom`, marked in ascending order. At H its upper coordinates modulo L must be the unordered set of pins in `top`; a pin accepts any label. Condition on these end positions in the Brownian bridge sense. Retain only paths with no contact at any height, including contact around the back of the cylinder. Start each lift at its listed bottom coordinate in [0,L). If its lift ends at an upper pin y plus kL, assign it winding k. The sum of these integers over all lines must equal `winding`.

Some steps nucleate together or meet at an upper pin. A run of m equal coordinates c in either pin list denotes an ordered set of m pins at c+(j−(m−1)/2)ε for j=0,…,m−1. Perform the endpoint, survival and winding conditioning for positive ε, then take ε→0. Step labels retain their order in the lower pin list. This defines a limiting ensemble even when distinct steps have the same nominal pin.

Two cameras examine the same retained surface. At height aH the step with lower index `tag` (zero based) must lie inside `window`. At bH exactly `count` of all the steps must lie inside `count_window`. Let P be the probability that both requirements are met. Both windows use coordinates modulo L in [0,L) and include their lower edge.

* Pinned inputs have `mode="pinned"`, N in {2,3,4,5}, `circumference` L in [7,9], and `height` H in [3.2,4.5]. Each nondecreasing pin list contains N entries in [0.25,L−0.25]. Distinct pin centers on a list have circular distance at least 0.7. A list may consist of a single bunch. The integer `winding` is −1, 0 or 1.
* `levels=[a,b]` has a in [0.2,0.4] and b in [0.6,0.8]. Each window is a pair [lo,hi] with 0 ≤ lo < hi ≤ L. The integer `count` lies in [0,N].
* If `open_pair` is null, report P. Otherwise it is the lower index of a bunch containing exactly two steps in `bottom`. Open just these two pins from their shared center c to c−sL/2 and c+sL/2. Call the resulting joint camera probability P(s), taking the ε→0 limit for every other bunch at fixed s. Report the dimensionless slit response lim(s→0⁺) [P(2s)−P(s)]/(3s²). Each slit opening uses its own retained ensemble.

For the following example the calibrated camera probability is 0.26814. Changing only `open_pair` to 0 gives a slit response of −0.83267.

```json
{"mode":"pinned","circumference":8,"bottom":[2,2],"top":[3,3],"height":3.8,"winding":1,"levels":[0.3,0.69],"tag":1,"window":[0.4,4.4],"count_window":[0,8],"count":2,"open_pair":null}
```

Implement `Model.fit(records)` returning self with the inferred diffusivity in `self.rate`, and `Model.predict(inputs)` returning a flat finite real NumPy array in input order, including empty inputs. Absolute errors must be below 0.00002 for each calibration reading, 0.003 for each probability, and 0.015 for each slit response. The starter uses independent Gaussian bridges and treats a narrow slit as a point at every opening.
