# Pore echo memory, revision 4

The previous transmission crossover passed 2/3 Luna High trials. R4 asks for
local coincidence of two activated tracers in a common random potential with
dilute absorbers. Its full detector counts the atom at zero rescaled separation.
That event includes survival to readout. A finite detector also localizes the
bottom of the occupied valley. Exact lattice-site coincidence would retain
microscopic thermal fluctuations and give a different observable.

## Landscape average

Use energy units u h and length units (u h)², so the scale of active barriers
is one. The Gaussian walk in the spatial coordinate converges to Brownian
motion with variance 2 per unit length and drift −2d, where d=fu. The
absorber intensity in these units is r=c a u². Arrhenius crossing times
separate barriers below and above the observation scale. The reversible
site rates respect the specified potential. Microscopic attempt factors
disappear from the leading logarithmic time scale.

At this scale a tracer is localized at the bottom of its surviving valley.
In the finite-interval renormalization measure each absorbing endpoint has
an edge region that collects its losses. Each remaining interior valley
has two bulk slope lengths. In Laplace variables their transforms are

    P_d(p) = exp(−d) k / [k cosh(k) − d sinh(k)], k = sqrt(p+d²).

The descending (left) flank uses d, and the ascending (right) flank uses
−d. These are the exit-time laws at 1 for a Brownian motion with generator
∂²−2d∂ on [0,1], reflected at 0 and started there. Poisson averaging weights
the two lengths by exp(−r times their sum). Summing over neighboring valleys
and the two edge regions multiplies the resulting measure per unit length
by C=r/[1−P_d(r)P_{−d}(r)].

Let B_d(v) be the mass of the tilted slope-length law above v. For a valley
with bottom at z to contain both sources, its left flank must exceed max(z,0)
and its right flank must exceed max(y/u²−z,0). The signal is the integral
over the scaled pickup window of C times these two tails. For coincident
sources at zero field with full acceptance this gives 2√r/sinh(2√r).
The limiting finite-interval measure, the Brownian units, and the full-line
meeting observable are derived in sections 2, 3 and 7 of
[Le Doussal, EPL 87, 20006 (2009), extended paper](https://arxiv.org/abs/0906.0267).
The finite-window weighting here follows by restricting the bottom selected
by that measure.

## Controls and checks

The oracle inverts P_d by residues at the roots q cot(q)=d. In the public
domain |d|<1 all poles correspond to real q. It integrates the two tilted
tails over pickup position. The independent numerical reference instead
discretizes the reflected drifted Brownian generator in space, diagonalizes
its birth-death matrix and integrates the exit flux. The validator refines
the mesh from 128 to 256 intervals and checks the exact unbiased full-line
stop-loss series and simultaneous reflection of force and detector.

An independent direct simulation counted surviving valleys in finite random
walk landscapes. At u=1 and r=0.7, unbiased full acceptance at separations
0, 0.25, 0.8 and 1.6 gave 0.65145, 0.45514, 0.14650 and 0.019403, within
0.0031 of the scaling law with spatial step 0.00005. For d=0.65 and
separation 0.65, windows [−0.6,−0.1], [0.15,0.60] and [0.85,1.45] gave
0.006018, 0.076409 and 0.049412 over 40,000 absorber intervals, compared with
0.006001, 0.076232 and 0.048902 analytically.

The completed shortcut models the two flanks by exponential domains with the
correct unmarked directional mean lengths. It applies the measured zero-field
survival and the probability that the source segment contains no mark. It fits
all calibration readings and misses the broad, absorber-selected distribution
of the two slopes around the measured bottom. Its maximum absolute errors in
the three families are 0.0488, 0.0281 and 0.0263. Grading allows 0.003 absolute
probability error, so probability tails have the same absolute resolution as
the main signal. Run `python scripts/validate_pore_echo_memory.py` to check
the shipped data and both controls. Use `--regenerate` for intentional updates.
