# Two contact recorders on a cylindrical terrace

A cylindrical surface carries two directed terrace steps. Its circumference is L. With height h each step diffuses azimuthally with unconditioned variance 2γh in fixed length units. The same diffusivity γ in [0.75,1.15] applies to every shot. A free single-step run has `mode="wandering"` and only a `height` H in [0.3,6.5]. Its reading is the mean squared unwrapped displacement after H.

On the contact-recording surface the steps reflect whenever they meet. They retain their material identity and spend zero ordinary height at exact contact. Start their continuous lifts X₀,X₁ at the two listed `bottom` pins in increasing order and keep the gap G=X₁−X₀ in [0,L]. In the interior each lift has independent noise of variance 2γ dh. At either contact the two lifts are normally reflected, with equal and opposite pushes. These reflections preserve the center position. There are no other forces or exchanges.

At height H require the unordered exit coordinates modulo L to equal the `top` pin pair, in the Brownian bridge sense. Each lift ends at a top pin y plus an integer multiple kL. Retain the specified sum `winding` of the two k values. At height t=`level`×H a camera records whether the initially marked step, with zero-based lower index `tag`, lies in `window` modulo L. The camera and the two contact recorders refer to the same retained pair.

The recorders distinguish contact through G=0 from contact through G=L. Their dimensionless readings through height t are

    U₀ = (2γ/L) lim(d→0⁺) (1/d) ∫₀ᵗ 1{0≤G(h)<d} dh,
    U₁ = (2γ/L) lim(d→0⁺) (1/d) ∫₀ᵗ 1{L−d<G(h)≤L} dh.

For the input `clocks=[u,v]`, a positive setting selects a bin from that setting up to that setting plus ε. A zero setting selects an exactly zero recorder reading. The reported signal is the probability of both selected readings and a positive tag image, conditional on the stated exit pins and winding, divided by ε once for each positive setting. Take ε→0⁺. The result can be a probability, a density or a joint density. It is scaled per unit of the dimensionless recorder coordinates.

Contact experiments use `mode="contact"`, `circumference` L in [7,9], `height` H in [9,18], and `level` in [0.35,0.85]. Each pin list contains two distinct coordinates in [0,L) in ascending order; both cyclic spacings are at least 0.65. The net winding is −1, 0 or 1 and tag is 0 or 1. The window [lo,hi] accepts coordinates lo ≤ x < hi and satisfies 0 ≤ lo < hi ≤ L. Both clock settings lie in [0,0.75].

For example, the calibrated reading is approximately 0.49221 for the input below. Changing only the clocks to [0,0] gives approximately 0.03089.

```json
{"mode":"contact","circumference":8,"height":17,"level":0.65,"bottom":[1,3.5],"top":[0.7,5],"winding":1,"tag":1,"window":[0.4,6.7],"clocks":[0.15,0.25]}
```

Implement `Model.fit(records)` returning self and storing the fitted diffusivity in `self.rate`. Each open calibration value is rounded to eight decimal places. `Model.predict(inputs)` returns a flat finite real NumPy array in input order, including empty inputs. The maximum absolute error must be below 0.00002 for the calibration and below 0.004 for each new contact reading. The starter uses an independent half-line bridge for each recorder at a given exposure gap.
