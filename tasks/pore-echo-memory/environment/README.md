# Coincidence imaging in a disordered capillary

A capillary is an infinite lattice of spacing one. Set the thermal energy and
the microscopic attempt rate to one. Fabrication chooses a potential V, with
V(0)=0. The successive differences V(j+1)−V(j), for all integer j, are
independent Gaussians of variance 2 and mean −2f/h. Thus positive f favors
motion to the right. Independently, mark each lattice site with probability
1−exp(−c a/h²), where a is a known coating strength and c in [0.4, 1.3] is
common to all preparations. The potential and marks stay fixed on each sample.

Release distinguishable tracers at sites 0 and floor(y h²). Each makes
continuous-time nearest-neighbor jumps. Its rate from i to either neighbor j
is exp[−max(0,V(j)−V(i))]. Both tracers use the same capillary and independent
thermal noise. They do not interact. On a marked site a tracer is lost
immediately, including if it starts on one.

At time t=exp(u h) a coincidence counter records one when both tracers are
still present, their separation is less than ε h², and their midpoint divided
by h² lies inside the detector window. A null window covers the entire line.
A counter reading is the probability per release, averaged over both the
thermal walks and fabricated samples. Lost tracers contribute zero. For each
choice of inputs take h→∞ first and then ε→0. The two releases and the
detector coordinates are set in the same laboratory frame throughout.

Each input contains u in [0.5, 1.4], y in [0, 2], f in [−0.6, 0.6],
a in [0.25, 1.6], and window (null or a two-element increasing list with
coordinates in [−3,4]). Calibration uses coincident sources (y=0), zero
force and a full detector. Each row has an input and its probability value
rounded to eight decimal places. Implement Model.fit(records) returning
self with the fitted float intensity c stored in self.intensity. Implement
Model.predict(inputs) returning a finite NumPy array of shape (len(inputs),)
in input order, including an empty array for an empty input.
