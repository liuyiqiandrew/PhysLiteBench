# Two-height imaging of crystal steps on a cylinder

A cylindrical crystal surface has L azimuthal lattice sites, numbered 0 through L−1 clockwise. Each directed terrace step traces a path of sites as the longitudinal coordinate h increases. An isolated step jumps one site clockwise and one site counterclockwise at independent Poisson rates γ per unit height. The lateral lattice spacing is one. The same γ in [0.75, 1.15] applies to every preparation.

An open calibration surface carries one step. The microscope records the mean squared unwrapped displacement after a height H, measured in lattice spacings squared. Each calibration input has `mode="wandering"` and `height` in [0.3, 6.5]. Its `value` is rounded to eight decimal places.

A pinned surface has N distinct steps beginning at the ordered sites in `bottom` at h=0. Generate independent paths by the same jump rule. A trial is discarded at the first contact of any two steps, including contact across the join between L−1 and 0. At h=H the set of occupied sites must equal `top`; these pins accept any step. From 0 to H record the net number of seam crossings summed over all paths. A hop from L−1 to 0 contributes +1 and the reverse hop contributes −1. Retain the surface only if this net count equals `winding`. All pinned readings are probabilities conditional on these three acceptance conditions.

A harmless isotope marks the step starting at `bottom[tag]` (zero-based index). Two cameras examine the same retained surface. At the first height its marked step must lie inside `window`. At the second height exactly `count` steps, including the marked step if present, must lie inside `count_window`. Report the probability that both cameras accept.

Each pinned input supplies the following settings.

* `mode="pinned"`; integer N in [2,5] is the length of both pin lists. The integer `circumference` L lies in [3N,4N]. Each pin list is strictly increasing with N entries and every cyclic spacing lies in [2,5] sites, including the last-to-first spacing.
* `height` H in [2.5,6.5]. `levels=[a,b]` locates the exposures at aH and bH with a in [0.18,0.42] and b in [0.58,0.82].
* `winding` in {−1,0,1} and `tag` in {0,…,N−1}.
* Both windows are integer pairs [lo,hi] with 0 ≤ lo < hi ≤ L; a site belongs to the window when lo ≤ site < hi. The integer `count` lies in [0,N].

Implement `Model.fit(records)` returning self with the inferred rate stored in `self.rate`, and `Model.predict(inputs)` returning a flat finite real NumPy array in input order, including an empty array for empty input. Reproduce every calibration reading to absolute error below 0.00002, and each new pinned probability to absolute error below 0.003. The starter implements independent pinned bridges, which allow contacts.

For example, at the calibrated rate the following pinned camera reads approximately 0.53848.

```json
{"mode":"pinned","circumference":9,"bottom":[0,3,6],"top":[1,4,7],"height":4,"winding":1,"levels":[0.3,0.7],"tag":2,"window":[6,9],"count_window":[2,6],"count":1}
```
