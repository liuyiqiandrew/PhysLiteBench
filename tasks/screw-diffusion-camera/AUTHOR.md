# Screw diffusion camera, revision 1

A Brownian rigid body's mobility can couple translation to rotation. The position tracked by a camera can also differ from the material origin at which the diffusion coefficients were specified. Both effects matter when optical selection weights the orientations of the same particle at two times. Translational and rotational cross-correlations and their dependence on the chosen tracking point are derived by Cichocki, Ekiel-Jeżewska and Wajnryb. The apparatus here specifies its complete positive diffusion matrix through the six independent Brownian rigid motions. [Brownian motion of a particle with arbitrary shape](https://arxiv.org/abs/1502.03747)

## Backward equation at the tag

Let u=Rᵀe be the camera direction in the body frame. Its invariant initial distribution is uniform on the sphere. If the body rotates about its j-th body axis, u evolves by the vector field Jⱼ=(u×eⱼ)·∇. For the tagged coordinate the same rotation moves the observed point along the body vector cⱼ=hⱼ+eⱼ×b. This follows by differentiating x+Rb along the specified finite screw. Thus in the spatial Fourier sector exp(ikz), the joint rotational channel acts by

    Aⱼ(k) = Jⱼ + ik u·cⱼ.

The backward generator in that sector is

    L(k) = ν ∑ⱼ [rⱼ Aⱼ(k)² − dⱼ k² uⱼ²].

The square acts on the full product. Its first-order part in k includes Jⱼ(u·cⱼ) in addition to 2(u·cⱼ)Jⱼ. Dropping that derivative changes the tagged drift. The free rotational correlator along principal body axis j is exp[−ν(∑r−rⱼ)t], since the other two rotational vector fields each square to −1 on uⱼ. The calibration scale follows from a linear fit to the logarithms.

Write W₀ and W₁ for multiplication by the two gate acceptance probabilities. In the uniform initial ensemble, the complex numerator is the spherical average of

    W₀ exp[t₁ L(k₁)] W₁ exp[t₂ L(k₂)] 1.

The denominator is the spherical average of W₀ exp[t₁ L(0)] W₁. Taking the real part of their ratio times exp(−i phase) gives the reported fringe. The two intervals share a single intermediate orientation. A product of separately angle-averaged characteristic functions misses this memory.

## Independent checks

The oracle applies this backward generator in complex spherical harmonics through degree seven. Angular momentum matrices implement Jⱼ. It forms coordinate products with an extra angular degree before truncation. The private reference propagates the original material origin with full finite positive and negative screw rotations, including the exact translation integrated along each curved body-axis path. It differences the two transfers and projects the result at degree nine. All three tag offsets enter through explicit endpoint phase factors. The reference does not shift the mobility tensor to the tag or use differential angular-momentum matrices.

Seed 305791 gives four families of seven readings each. The origin family measures screw motion itself. The tagged family adds a lever arm. The quadrature family selects the imaginary part of the conditional scattering signal. The reversal family gives the successive displacements opposite gradient signs. Checks include mirror reversal, zero-gradient normalization, refinement to degree eleven and halving the finite angular step from 0.003 to 0.0015 radians. The largest difference between oracle and frozen target is 2.6 × 10⁻⁷. The calibration and fringe absolute tolerances are 0.000002 and 0.003.

The completed shortcut gives the material origin and body orientation independent noises with their correct marginal diffusion tensors. It retains the geometric motion of the off-origin tag, both orientation gates and the shared intermediate orientation. It fits the rotational calibration and errs by 0.0217 to 0.0868 across the four families. The public starter instead treats the tag as an isotropic Gaussian random walk with its exact orientation-averaged instantaneous variance. It fits the same calibration and predicts zero for every sine quadrature. The public worked example exposes this approximation.

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_screw_diffusion_camera.py` to reproduce the stored observables and compare the models. Add `--generate` to update the data intentionally. Both the training and grading images set a single BLAS thread. The 1,800-second solve and 180-second verifier limits match recent physical-failure panels.
