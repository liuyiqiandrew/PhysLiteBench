# Screw diffusion camera

A rigid particle diffuses by six independent infinitesimal body motions. The material origin follows a screw during each rotational kick. The tagged point picks up a further displacement when its body offset rotates. Optical selection, the displacement fringe and the gyroscope gradient signal all refer to the same trajectory. Translational and rotational cross-correlations and their dependence on the chosen tracking point are derived in [Cichocki, Ekiel-Jeżewska and Wajnryb](https://arxiv.org/abs/1502.03747).

## Tagged scattering

Let u=Rᵀe. A body rotation about axis j differentiates a function of u by Jⱼ=(u×eⱼ)·∇. The tag displacement per radian is u·cⱼ with cⱼ=hⱼ+eⱼ×b. For spatial Fourier wave number k, define Aⱼ(k)=Jⱼ+ik u·cⱼ. The backward solvent generator is

    L(k) = ν Σⱼ [rⱼ Aⱼ(k)² − dⱼ k² uⱼ²].

The square differentiates the coefficient of the multiplication term as well as the function. On uⱼ the zero-wave generator is −ν(Σr−rⱼ)uⱼ. This gives the exponential calibration curve and the fitted mobility. Let W₀ and W₁ multiply by the gate acceptance probabilities and ⟨·⟩ denote the initial uniform average. The free gated fringe has complex numerator ⟨W₀ exp[t₁ L(k₁)] W₁ exp[t₂ L(k₂)] 1⟩ and real denominator ⟨W₀ exp[t₁ L(0)] W₁ 1⟩.

## First spatial moment of signed area

A phase coil with field proportional to horizontal position measures the first horizontal moment of signed area. Green's theorem gives M=½∮X² dY. For an open record ending at ordinate d put I=½∫X² dY. The straight chord contributes −dX_T²/6. Horizontal and vertical bare angular channels are independent Brownian motions of diffusion constant ν. They also drive the same material screw motions that determine the scattering phase and both optical gates.

In the characteristic sector exp(ipY), each horizontal gyro increment differentiates the remaining function by ∂_X. A vertical increment acts through ip+B, with B=X²∂_I/2. In the corresponding body channels the tagged generator thus replaces Aₐ by Aₐ+∂_X/√rₐ and A_b by A_b+(ip+B)/√r_b. For slope or curvature propagate the terminal polynomial (I−dX²/6)ʳ with r=1 or 2. The largest polynomial space required consists of the nine monomials XⁱIʲ with i+2j≤4. Initialize at X=I=0 and divide by the tagged gate acceptance at the same pins. A final pin uses one Fourier variable shared by both intervals. A checkpoint selects two vertical increments separately. The horizontal coordinate and integrated moment continue through the middle gate.

## Balanced actuator stroke

Periodic controls can generate motion along Lie brackets of otherwise inaccessible rigid directions. [Murray and Sastry, Nonholonomic motion planning](https://authors.library.caltech.edu/records/4g4pk-07277)

The positive drive is the group word [A,[A,B]], with the convention [A,B]=ABA⁻¹B⁻¹ in chronological body motions. Its logarithm starts at cubic order in the edge length. The stroke contributes a finite deterministic drift because e³=|stroke|dt. For body twists (w,v), their Lie bracket is (w₁×w₂, w₁×v₂−w₂×v₁). Apply that formula twice to the two axes and their screw translations, including the normalization rₐ√r_b. The resulting angular part is proportional to −e_b. At the tag the linear part acquires the usual rotation cross product with b. The active stroke has zero signed planar area, yet its two opposite squares differ in first horizontal moment by e³. Its macroscopic contribution to M is therefore stroke × T, with its physical sign.

The reference independently composes all ten finite rotations and translations and integrates the gyro word edge by edge. Its centered cubic difference estimates the limiting body-and-moment generator. The oracle evaluates the nested rigid-body bracket and its constant moment drift directly.

## Independent checks

The exact free moments follow by conditioning on the whole independent horizontal path. Write w(t)=X(t)²/2−X_T²/6, so M=∫w dY. Conditional on the specified Y values, each vertical bridge contributes a mean ΔY times the interval average of w and a variance 2ν times its integrated squared deviation from that average. Wick's rule gives E[X_t²X_u²]=4ν²[tu+2 min(t,u)²]. Integrating these polynomials and shifting M by stroke × T gives the public worked checks for both signs and both pin protocols.

The oracle uses body angular derivatives at the moving tag, polynomial differentiation and Gauss–Hermite Fourier integration. The reference instead advances the exact finite positive and negative rigid screws at the material origin, shifting every polynomial by the actual change in X or I. It uses a rectangular Fourier grid for the vertical pin and explicit phase factors for the tag at its three exposures. The validator regenerates the ordinary 28 solvent readings and all new observables, and compares the models at finer spatial and Fourier resolutions.

The completed physical comparison keeps the full joint thermal scattering and gyro dynamics plus the exact programmed polygon moment. It treats the zero-net, zero-area stroke as leaving the body pose undisturbed. Its errors on the slope, curvature and checkpoint families are 0.04028, 0.07988 and 0.03081. The public starter additionally replaces the thermal X(t)² by its free mean and draws the scattering noise independently. Its corresponding errors are 0.05722, 0.36962 and 0.03904. The oracle differs from all frozen monitor targets by less than 0.0000015, against the public 0.003 tolerance.

In the validator’s nonzero-wave example, reversing only the stroke from −0.65 to +0.65 takes the slope from 0.01169334 to 0.20588342. Omitting the rigid displacement while retaining the correct gyro moment gives −0.00721470 and 0.16559616.

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_screw_diffusion_camera.py` from the repository root. Add `--generate` to write the fixture file.
