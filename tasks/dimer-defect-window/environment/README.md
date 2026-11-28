# A contour microscope for two dimer sheets

A sheet has adsorption sites at integer points (x,y) with 0 ≤ x < w and 0 ≤ y < h. A dimer occupies a horizontal or vertical nearest-neighbor bond. Nothing crosses the open outer edges. At equilibrium each site belongs to exactly one dimer. The probability of a covering is proportional to the product of the bond activities, 1 horizontally and z times the input `ratio` vertically. The unknown z in [0.6,1.4] is common to all sheets. The dimensionless ratio is in [0.4,2.2].

Two sheets of the same size equilibrate independently at the same ratio. The microscope superposes their bond images at corresponding lattice coordinates. It erases a bond wherever the two images coincide. The remaining edges form closed, nonintersecting contours. Each geometrical contour counts once. The sheet pair is drawn again for each exposure.

Two small optical marks are placed at the centers of specified plaquettes. An entry [x,y] in `marks` places the mark at (x+0.5,y+0.5). A contour surrounds a mark if the mark lies in its bounded interior. Two nested contours can both surround the same mark. The marks do not affect binding.

The event counter accepts a shot if exactly `count` contours each surround **both** marks. If `allow_separate` is false it also requires that no contour surround just one of the marks. Contours surrounding neither mark may occur freely. Report the acceptance probability per prepared sheet pair, including the shots rejected by either condition in the denominator.

A `mode="bond"` record calibrates the instrument on one 2 by 2 sheet and reports the probability that its left vertical bond is occupied. It supplies only `mode` and `ratio`. A `mode="loops"` input has integer w,h in [2,32] with wh even, `ratio`, two marks in different plaquette columns, an integer `count` in [0,4], and a boolean `allow_separate`. Each mark obeys 0 ≤ x ≤ w−2 and 0 ≤ y ≤ h−2. Requests with an impossible count have probability zero.

For example, a 3 by 2 sheet has only three coverings. For marks [[0,0],[1,0]], count=1 and allow_separate=true, the probability is 2/(2+(z ratio)²)². The pair forms a six-edge contour when its two sheets have horizontal dimers on opposite sides.

Implement `Model.fit(records)` returning self and storing the fitted z as `self.activity`. Calibration rows in `data/calibration.json` give an `input` and its probability `value` rounded to eight decimals. `Model.predict(inputs)` returns a finite real NumPy array of shape (len(inputs),) in input order. Absolute errors must be below 0.00002 on every calibration reading and below 0.004 on every new reading. The starter approximates the marked plaquettes as isolated local rearrangements.
