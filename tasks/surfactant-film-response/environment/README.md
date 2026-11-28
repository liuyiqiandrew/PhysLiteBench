# Velocity imaging on a driven film

An ideal narrow surfactant monolayer has one molecule per adsorption site on an infinite line. Sites have integer coordinates in lattice-spacing units. Molecules carry fixed species indices. At time zero the molecule at site j has index j, for every integer j. On each neighboring bond the two molecules exchange places at rate ν if the index on the left is smaller, or qν if it is larger. There are no other transitions. These are continuous-time Markov rates with independent clocks. The ratio q is an apparatus setting in [0, 0.5]. The same unknown frequency ν in [0.85, 1.15] applies to every run.

Write X_j(t) for the coordinate of the molecule initially at j. A camera follows a consecutive block of these molecules in a single preparation, starting at j=0. Its normalized speed reading is the long-time limit of (X_j(t)−j)/[(1−q)t]. Speeds and the thresholds below are in lattice spacings per unit time. All probabilities refer to fresh realizations of the clocks with the stated initial condition.

Each experiment supplies two thresholds `lo` and `hi` in [−0.70, 0.70], at least 0.12 apart, and a `word` of 1 to 7 letters. Letter j refers to the j-th molecule, in order of **initial** site. A star accepts any reading.

* In `mode="bands"`, L accepts speeds below lo, M accepts speeds in [lo,hi], and H accepts speeds above hi. Report the probability that every letter accepts.
* In `mode="lock"`, the last molecule is the reference. Keep a shot only if its speed is in [lo,hi]. Relative to that molecule's speed, L accepts a smaller speed, M an equal speed and H a greater speed. The final letter is M. Report the fraction of retained shots in which every letter accepts. Equivalently the three regions about the reference have a middle half-width ε; take the long-time limit first, then ε to zero. Stars have the same meaning in both modes.

For example, a locked exposure with `q=0`, `word="MM"`, `lo=-0.30` and `hi=0.30` measures the probability that two neighbors travel at the same speed. Its reading is 1/4 − 0.30²/(12ν²).

The calibration camera looked at one molecule at a time. Each row of `data/calibration.json` has an `input` dictionary and its probability `value` rounded to eight decimal places. Implement `Model.fit(records)` returning self with the fitted frequency in `self.rate`, and `Model.predict(inputs)` returning a one-dimensional real NumPy array in input order. The fitted frequency must have absolute error below 0.0001. Absolute probability errors must be below 0.00002 on every calibration reading and below 0.001 on every new preparation.
