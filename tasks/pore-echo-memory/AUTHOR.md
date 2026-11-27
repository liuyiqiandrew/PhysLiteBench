# Pore echo memory, revision 3

R1 studied spin phases in a four-site pore. R2 measured the leading log rate
of a two-isotope survival covariance. Luna High solved both revisions in all
three trials. R3 measures the shape of a transmission crossover. At this
scale, the spatial profile of a surviving diffusion mode enters the result.

## Wall and diffusion contributions

Write T=t^(1/3). In a pore of length L the ground wavenumber is k/L,
where k=π for DD walls and π/2 for a mixed pair. The normalized eigenfunction
is √(2/L) sin(kx/L) with a left absorbing wall, or √(2/L) cos(kx/L)
with a left reflecting wall, with x measured from the left wall. Spectral
gaps suppress the other modes exponentially on the contributing length scale.
For a single DD isotope the cost is cL + π²D t/L². Minimizing gives the
reported log rate 3(π²D c²/4)^(1/3).

Set A=D_A k_A²+D_B k_B². In the preliminary scan the leading negative
log weight is cL+A t/L², including the Poisson cost for an empty pore.
The locations of 0 within a pore and the two local eigenfunction products
contribute algebraic factors. Consequently the weighted L/T concentrates
at w=(2A/c)^(1/3).

For large t both transmission paths cross the same interval between the
leftmost source and the rightmost receiver. All four points must be in one
pore. The outer walls have joint weight c² exp(-cL) per unit of their two
coordinates. Put l=min(left), r=max(right), and write the exterior distances
from the two outermost points as a√T and b√T. Then L=wT+z√T with
z=r-l+a+b. Expanding the total cost around wT gives curvature factor
exp(-3A z²/w⁴). Both ends of the pore remain integration variables.

The slow eigenfunction near an absorbing face is proportional to distance
from the face. Near a reflecting face it tends to a constant. Each isotope
i with a left absorbing face contributes a factor a+left_i-l. A right
absorbing face contributes b+r-right_i. Constants, powers of T and the
leading exponential cancel in the reported ratio. Integrate the resulting
polynomial times the curvature factor over a,b≥0, and divide by the same
integral with all four offsets zero. This accounts for unequal source and
receiver displacements even when their outer span stays fixed.

## Controls and reference

The oracle integrates the polynomial over a at fixed a+b using beta
integrals. Its remaining integral is one-dimensional. The completed
approximation uses the exact spectral attenuation, Poisson wall geometry
and fitted intensity, but treats each slowly decaying mode as spatially
uniform. It fits DD single-isotope calibration. Its relative errors exceed
50% in every hidden family. The oracle's maximum relative error is below
0.0001%. The verifier allows 0.8% relative error in each family, well above
the numerical reference uncertainty.

The independent reference integrates over the two wall locations at finite
time. It evaluates three modes of the diffusion heat kernel for each isotope
with the specified boundary conditions. It removes the same exponential from
both signals before normalization. At T=10⁸ and 4×10⁸, extrapolation in
1/√T removes the leading finite-time bias. The validator refines the time
and quadrature order, and checks exact translation invariance. Reproduce
with `python scripts/validate_pore_echo_memory.py`; pass `--regenerate` to
recreate the deterministically rounded calibration and prediction fixtures.
