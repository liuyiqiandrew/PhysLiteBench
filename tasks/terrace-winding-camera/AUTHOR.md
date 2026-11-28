# Terrace contact camera, revision 3

Previous terrace revisions measured surviving paths with absorbing contacts. On this material both steps reflect and retain their identities. A pair of local-time ledgers is read at the same height as the tagged image. This is an experiment on a reflected path conditioned on its endpoints, including shots that visit a boundary repeatedly. A local-time reading of exactly zero has discrete probability. Two positive readings require the same pair to have visited both sides before the camera exposure.

## Geometry and normalization

The lifted center and gap evolve independently. The center has diffusivity γ/2. The gap has diffusivity D=2γ and normal reflection at the ends of (0,L). For label i the winding specifies top[(i+w) mod 2]+L floor((i+w)/2) as its lifted endpoint. This fixes the terminal center and gap. At intermediate height t the center is Gaussian with mean interpolated between its endpoints and variance γt(1−t/H). Given gap y, integrate that Gaussian over all periodic images of the window for the position M+(tag−1/2)y.

Let B(y) be that tag probability times Kᴺ(H−t;y,g_H)/Kᴺ(H;g₀,g_H), where Kᴺ is the Neumann interval heat kernel. Multiplying B by the joint propagator from g₀ to y with the requested boundary readings, then integrating over y, implements both the tag image and endpoint conditioning. A zero clock at both ends selects the Dirichlet propagator through t. After t all paths may reflect.

## Contact propagator

The physical local times are ℓ₀=LU₀ and ℓ₁=LU₁ in the one-sided residence convention used in the question. Laplace weights exp(−q₀ℓ₀−q₁ℓ₁) impose Robin coefficients q₀,q₁ on the two ends. In the height-Laplace variable p put z=√(p/D), c=z coth(zL), d=z/sinh(zL), v(x)=[sinh(z(L−x)),sinh(zx)]/sinh(zL). The resolvent is the Dirichlet resolvent plus

    v(g₀)ᵀ [[q₀+c,−d],[−d,q₁+c]]⁻¹ v(y)/D.

This follows by matching the Robin derivatives of the two homogeneous solutions at the boundaries. Inverting in q₀,q₁ separates a Dirichlet atom, densities on each axis, and a density in the interior. For ℓ₀>0, ℓ₁=0 the extra matrix has only its 00 entry exp(−cℓ₀). Exchange the endpoints for the other axis. For positive readings let x=2d√(ℓ₀ℓ₁). The common factor is d exp(−c(ℓ₀+ℓ₁)). The off-diagonal entries have factor I₀(x), while 00 and 11 have factors √(ℓ₀/ℓ₁)I₁(x) and √(ℓ₁/ℓ₀)I₁(x). Multiply by L for each positive reading to convert from physical lengths to the reported dimensionless densities. The boundary local-time construction on an interval, including its singular axes, is derived in [Grebenkov, *Joint distribution of multiple boundary local times*](https://arxiv.org/abs/2008.12986).

The oracle uses a Talbot contour for the inverse height transform and a cosine series for the later reflecting bridge. The private reference sums visits between opposite boundaries, reconstructs positive-time values from real frequencies with high-precision Gaver–Stehfest weights, and uses Gaussian images for the later bridge. A further check integrates the full joint reading over its atom, axes and interior with two exponential penalties. An independent heat kernel from the Robin interval eigenfunctions gives the same tilted tag probabilities within 1 × 10⁻¹¹. A separate sweep over 108 allowed parameter extremes gave a maximum difference of 3.4 × 10⁻⁸ between the time inversions.

The eight open records identify γ=0.927. Seed 529408 sets 36 frozen preparations. The complete shortcut computes both exact marginal recorder readings under the pin and tag conditions and multiplies them as if independent. Its largest errors in the no-contact, one-recorder, two-recorder and narrow tag families are 0.101, 0.167, 0.251 and 0.083. The public starter computes the tagged position from the reflecting bridge while replacing the recorder histories by independent half-line bridges to its exposure gap. It misses the joint density by up to 0.296. Each fits open calibration and runs well within the verifier allowance. The uniform 0.004 tolerance measures absolute signal error in the dimensionless detector units.
