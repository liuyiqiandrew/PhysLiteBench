# Terrace winding camera, revision 1

A directed crystal terrace step on a cylinder has independent Poisson hops in either direction as its height grows. The open-line variance is 2γH. The eight calibration readings identify γ=0.927. Pinning conditions on the final set of sites, avoidance of every contact throughout the interval, and a specified total seam current. The cameras observe a tagged step followed by a count of steps in an arc. Their coincidence is a joint probability.

## Positive occupation-space calculation

For a sorted set of N occupied sites, attempt each of the 2N possible directional hops with equal probability. Remove its probability mass if the destination is occupied. For an allowed hop record the new sorted set and add +1 or −1 to an integer current register for a clockwise or counterclockwise seam crossing. Other hops leave the register fixed. The holding and killing rates follow by Poisson uniformization with mean 2NγΔh. Replacing blocked attempts by waiting events would change the path ensemble.

After a clockwise seam crossing the last site becomes the first in the sorted coordinates. Thus a dye initially on pin index j occupies rank (j+r) mod N after net current r. The reference propagates forward from the bottom, applies the tag gate at the first exposure, propagates to the second exposure and applies the count gate. A backward density from the upper pins in the requested current sector supplies the last factor. The generator on sorted sets and integer currents is symmetric. Dividing by the same pin-and-winding mass with both gates open gives the required probability. This computation uses only nonnegative transition weights. It discards paths leaving the current range [−9,9]. Enlarging the range to [−13,13] is checked at high jump rate and height, along with comparisons to the independent determinant calculation.

## Determinant calculation

Give a one-step clockwise seam hop a multiplier u and the reverse hop its reciprocal. A single radial line moving from i to j in height t has kernel

    Kᵤ(i,j;t) = Σₛ uˢ exp(−2γt) I_|j−i+sL|(2γt).

The absorbing collision boundary follows by antisymmetrizing the path propagator. At a seam hop the ordered sites rotate, contributing a permutation of sign (−1)ᴺ⁻¹. In the one-body kernel set u=(−1)ᴺ⁻¹ exp(iθ) so the surviving many-body paths carry exp(iθW) with the physical positive weights. Fourier projection in θ fixes the final net winding. The same construction applies to several consecutive propagators with diagonal one-body counting fields at the two observation heights. Their product between the two pin arrays has a determinant because the ordered intermediate configurations sum by Cauchy–Binet. The cancellation of intersecting paths and the parity twist for a circle also appear in [Liechty and Wang, *Nonintersecting Brownian motions on the unit circle*](https://arxiv.org/abs/1312.7390).

To count a tagged position below a cutoff x, multiply the first segment's seam weight by an additional Nth root of unity. This resolves its current r modulo N. Multiply rows for each site at the first height by z when its coordinate is below x, and at the second height by y inside the count window. The resulting determinant is a polynomial with powers from 0 through N in each of z and y. Its coefficient for m early sites below x and k late sites in the count window is the desired marked cumulative event whenever m exceeds (j+r) mod N. Subtract the two cumulative probabilities for the tag interval. The oracle uses 21 phases for final winding, N phases for early winding modulo N, and N+1 phases for each occupation count. The unobserved normalization uses only the first Fourier projection.

## Validation

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_terrace_winding_camera.py` from the repository root. Pass `--generate` to regenerate the fixtures deliberately. Seed 11467 sets the pin configurations, heights and camera gates for the 24 hidden records. The three groups measure an unlabelled count, the tagged position and a two-camera coincidence. They include both winding signs, odd and even particle numbers, and a five-step ring of circumference 20. The validator checks probability normalization, reversal of a tagged bridge, the published camera example and a larger current cutoff. At the fitted rate the largest oracle deviation from the ten-decimal fixtures is below 5.1 × 10⁻¹¹.

The completed starter and shortcut connect every label to its appropriate lifted endpoint with an independent unwrapped bridge. This enforces its final pin and current but lets intermediate paths cross. Its errors are 0.1960, 0.2182 and 0.2391 across the three groups. The 0.003 absolute error allowance is 0.3 percentage points of the conditional probability and exceeds the two reference calculations' discrepancy by several orders of magnitude. A 1,800-second solve limit and a 180-second verifier limit match the previous five-task expansion.
