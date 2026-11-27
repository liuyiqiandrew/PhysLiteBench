# Pore echo memory, revision 2

Revision 1 studied signed exposure counts in a four-site capillary. Three
unhinted Luna High agents solved it. Revision 2 asks how connected spin loss
signals scale in an extended disordered capillary.

## The effect of centering

After averaging the spin paths in a sample, subtracting the product of the
sample-ensemble means is equivalent to subtracting the same two-path experiment
with independent defects for the two paths. In one dimension a continuous
path visits its whole range. For two path ranges I and J, the Poisson
void probabilities give the following integrand for the covariance:

    exp[-a|I| - b|J| - c|I ∪ J|] (1 - exp[-c|I ∩ J|]).

Disjoint paths contribute exactly zero. For ranges with a strictly positive
overlap on the scaled length axis, the last factor tends to one.

## Long-time rate

Write u, v for the range widths in units of t^(1/3). For a Brownian motion
of diffusivity D confined to width u t^(1/3), the lowest Dirichlet mode
costs π²D/u² on the t^(1/3) log scale. The same exponent applies to paths
visiting neighborhoods of both ends. One can subtract the probabilities for
slightly shorter confinement intervals: their principal eigenvalues are
strictly larger. Binning the four endpoints on a scaled mesh gives the
matching upper bound by confinement. Brownian and Poisson weights outside
a sufficiently large width cutoff are exponentially small. Refine the mesh
after taking the large-time limit.

Both starts must lie in their respective intervals. Overlap requires u+v≥s.
The union covers both starts and both intervals, so its minimum possible
width is max(s,u,v). If the wider interval spans both starts, nest the shorter
inside it. Otherwise set the two outer endpoints at the two starts. Boundary
starts and touching intervals are limiting configurations reached by expanding
and translating the intervals slightly. Use strict interior starts and overlap
before taking the limit, then take the expansion to zero. This yields

    inf {a u + b v + c max(s,u,v) + π²(D_A/u² + D_B/v²) : u+v≥s}.

For one spin the cost is (a+c)u + π²D_A/u². Single-coil calibration therefore
identifies c. The completed shortcut drops centering at leading exponential
order. The raw product moment contains independent survival as well as the
connected covariance, so its rate is the smaller of the sum of the two
single-coil rates and the connected rate. This works for nearby sources.
At larger separation it counts paths whose disorder fluctuations cannot
contribute to the requested covariance.

## Numerical checks

The oracle fixes an outer width, clips each preferred private-defect width
to that cap, and minimizes the remaining convex objective. If their widths
would sum to less than s, it allocates exactly s by equating marginal costs.
The independent reference optimizes directly over both paths' left and right
extrema and the surrounding endpoints with linear geometric constraints. For
symmetric scans, exchanging labels forces equal optimal widths. The validator
also checks this closed-form limit for coincident and widely separated sources.

Run `python scripts/validate_pore_echo_memory.py` from the repository root.
Both controls fit c to within 1e-8 and all near-source rates to within 1e-8
relative error. The shortcut's maximum relative errors are 64% for separated
sources and 70% in the unequal-spacing family. Grading allows 0.8% relative
error per rate. Calibration is deterministically rounded to eight decimals.

For single-particle context see
[Sznitman, Brownian motion in a Poisson obstacle field](https://smf.emath.fr/system/files/2017-08/smf_ast_266_91--111.pdf).
