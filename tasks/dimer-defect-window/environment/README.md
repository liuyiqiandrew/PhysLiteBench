# A contour microscope for two dimer sheets

A sheet has adsorption sites at integer points (x,y) with 0 ≤ x < w and 0 ≤ y < h. A dimer occupies a horizontal or vertical nearest-neighbor bond. Nothing crosses the open outer edges. At equilibrium each site belongs to exactly one dimer. The probability of a covering is proportional to the product of the bond activities, 1 horizontally and z times the input `ratio` vertically. The unknown z in [0.6,1.4] is common to all sheets. The dimensionless ratio is in [0.4,2.2].

Two sheets of the same size equilibrate independently at the same ratio. The microscope superposes their bond images at corresponding lattice coordinates. It erases a bond wherever the two images coincide. The remaining edges form closed, nonintersecting contours. Each geometrical contour counts once. The sheet pair is drawn again for each exposure.

Two small optical marks are placed at the centers of specified plaquettes. An entry [x,y] in `marks` places the mark at (x+0.5,y+0.5). A contour surrounds a mark if the mark lies in its bounded interior. Two nested contours can both surround the same mark. The marks do not affect binding.

The event counter accepts a shot if exactly `count` contours each surround **both** marks. If `allow_separate` is false it also requires that no contour surround just one of the marks. Contours surrounding neither mark may occur freely. Report the acceptance probability per prepared sheet pair, including the shots rejected by either condition in the denominator.

A `mode="bond"` record calibrates the instrument on one 2 by 2 sheet and reports the probability that its left vertical bond is occupied. It supplies only `mode` and `ratio`. A `mode="loops"` input has integer w,h in [2,32] with wh even, `ratio`, two marks in different plaquette columns, an integer `count` in [0,4], and a boolean `allow_separate`. Each mark obeys 0 ≤ x ≤ w−2 and 0 ≤ y ≤ h−2. Requests with an impossible count have probability zero.

For example, a 3 by 2 sheet has only three coverings. For marks [[0,0],[1,0]], count=1 and allow_separate=true, the probability is 2/(2+(z ratio)²)². The pair forms a six-edge contour when its two sheets have horizontal dimers on opposite sides.

Implement `Model.fit(records)` returning self and storing the fitted z as `self.activity`. Calibration rows in `data/calibration.json` give an `input` and its probability `value` rounded to eight decimals. `Model.predict(inputs)` returns a finite real NumPy array of shape (len(inputs),) in input order. Absolute errors must be below 0.00002 on every calibration reading and below 0.004 on every new reading. The supplied contour routine computes this closed-sheet reading.

## Boundary contacts on a folded image

A `mode="fold"` exposure uses two transparent plates. Each plate carries its own pair of independently equilibrated sheets. The four coverings are statistically independent. The plates share a rectangular layout and a list of distinct boundary sites (`ports`) given in cyclic order around the outside. Port numbers start at zero. On a plate, port i exists only in sheet `layer[i]` (0 or 1). All sites outside the list exist on both sheets. A dimer can bind only two present neighbors. Within a plate both sheets have the same vertical activity z times that plate's ratio. Each sheet still has a perfect covering of its present sites.

Superimpose the two images on each plate and erase their coincident bonds. At an ordinary occupied site the remaining image has degree zero or two. At a port it has degree one, so a strand leads from that port to another port on the same plate. Now join each port of plate zero to the port of the same number on plate one. The combined image has closed cycles. Light injected along a strand can reach exactly the ports on its cycle. The two plates are optically separate everywhere else, even where their drawn strands cross in projection.

On a single combined image, retain the shot if the two port numbers in `gate` are on the same cycle. Among the retained shots report the fraction having the two port numbers in `read` on the same cycle. A port always shares a cycle with itself. Both inquiries use the same four coverings, then the entire preparation is renewed.

Inputs give `w,h,ports,plates,gate,read`. Each of the two `plates` entries supplies `ratio` and `layer`. The dimensions are between 2 and 22 with even area and aspect ratio at most 2. There are 4, 6, 8 or 10 ports, equally many on each checkerboard color. Ratios range from 0.6 to 1.7. The supplied preparations admit coverings on every sheet and retain at least a tenth of shots. The same 0.004 absolute tolerance applies. The starter assigns each allowable planar boundary pairing a product of independent two-port weights and follows the resulting cycles.

For example, on a 2 by 2 square list its four corners clockwise starting at the lower left and proceeding along the bottom. Let every port be present in sheet zero on both plates. Sheet one is empty. The probability of horizontal dimers on plate j is Hⱼ=1/[1+(z ratioⱼ)²]. For gate [1,2] and read [0,1], the reading is [H₀(1−H₁)+H₁(1−H₀)]/(1−H₀H₁). Different orientations on the two plates make a single cycle through all four ports.
