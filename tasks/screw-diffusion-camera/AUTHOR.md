# Screw diffusion camera

A rigid particle diffuses by six independent infinitesimal body motions. The material origin follows a screw during each rotational kick. The tagged point picks up a further displacement when its body offset rotates. Optical selection, the displacement fringe and the spin-interferometer brightness all refer to the same trajectory. Translational and rotational cross-correlations and their dependence on the chosen tracking point are derived in [Cichocki, Ekiel-Jeżewska and Wajnryb](https://arxiv.org/abs/1502.03747).

## Tagged scattering

Let u=Rᵀe. A body rotation about axis j differentiates a function of u by Jⱼ=(u×eⱼ)·∇. The tag displacement per radian is u·cⱼ with cⱼ=hⱼ+eⱼ×b. For spatial Fourier wave number k, define Aⱼ(k)=Jⱼ+ik u·cⱼ. The backward solvent generator is

    L(k) = ν Σⱼ [rⱼ Aⱼ(k)² − dⱼ k² uⱼ²].

The square differentiates the coefficient of the multiplication term as well as the function. On uⱼ the zero-wave generator is −ν(Σr−rⱼ)uⱼ. This gives the exponential calibration curve and the fitted mobility. Let W₀ and W₁ multiply by the gate acceptance probabilities and ⟨·⟩ denote the initial uniform average. The free gated fringe has complex numerator ⟨W₀ exp[t₁ L(k₁)] W₁ exp[t₂ L(k₂)] 1⟩ and real denominator ⟨W₀ exp[t₁ L(0)] W₁ 1⟩.

## Stored gyroscope record

The controller records a classical sequence of solvent rotation increments. It subsequently programs two coherent histories on an auxiliary spin. A common classical record is sufficient to correlate the two playback arms with each other and with the translation measured during the solvent shot. Coherent control over the order of spin operations enters the recorded-word protocol through the ordinary interference of the two path amplitudes. [Procopio et al.](https://www.nature.com/articles/ncomms8913)

Write F for chronological playback and R for reverse playback. V=R† is chronological, with the opposite sign on each elementary replay rotation and the adjoint middle pulse. The four terms of the path-selected intensity contain F⊗F*, V⊗V*, F⊗V and F*⊗V*. The second density term uses tr(E V†ρV)=tr(ρ VEV†). The two cross terms use tr(E FρV) and its complex conjugate for each stored record. After weighting by its complex displacement phase their expectations must be propagated separately.

For each pair of chronological spin histories let Sⱼ be the sum of their infinitesimal spin generators acting on the two tensor factors, including the appropriate complex conjugations. Their spatial and electronic changes are driven by the very same rotational increment. In the joint backward propagator each Aⱼ(k) becomes Aⱼ(k)⊗I + I⊗Sⱼᵀ. Transposition restores time ordering when the backward expectations are multiplied from the later interval to the earlier interval. At the middle exposure insert the two pulse matrices and W₁. The denominator counts optical acceptance alone. This use of a joint evolution for classical stochastic motion and the spin histories is an application of the stochastic Liouville equation. [Kubo](https://doi.org/10.1063/1.1703941)

## Independent checks

The solution uses angular momentum derivatives and Cartesian coordinate multiplication in spherical harmonics at the tracked tag. The oracle takes degree six for the replay readings and degree seven for the passive fringes. The private calculation uses actual finite signed screw rotations at the material origin and endpoint phases for the tag at all three observation times. It advances the four arm pairs using their original forward or reverse playback order, with the early and late spin indices interchanged on the reverse arm. The frozen replay labels use degree seven and step 0.002 radians.

In the ungated public example the spin diffusion is isotropic and R=F† for each record. The bright-port probability is cos²(θ/2)=[1+χ₁(F)]/4 where θ is the net rotation and χ₁ is its spin-one character. The three spin-one modes decay as exp(−2νt), giving 0.31598089. At collinear positive-sense playback the arms coincide trajectory by trajectory, giving the ordinary solvent fringe. A further fixed-parameter check uses noncollinear coils, unequal pulses, nonzero signed waves and a spin analyzer. Flipping only the playback sense in the validator's example changes the signal from 0.07820919 to 0.04193766.

The validator regenerates the 12 calibration values, the 28 passive readings and the 24 stored-record readings. It evaluates the exact limits and checks refined angular and finite-screw steps against the independent solver. The maximum discrepancy on the stored targets is below 0.0000003. The calibration and fresh-reading absolute tolerances are 0.000002 and 0.003.

The completed physical control preserves the conditional spin click probability and the conditional optical fringe individually. It draws their two trajectories independently. Its maximum errors on the stored, reversed-sense and analyzed readings are 0.00972, 0.00664 and 0.00451. The public starter retains the complete solvent propagation and programmed spin pulses, and replaces the noisy thermal rotation angles by their zero means. The equivalent errors are 0.04911, 0.09351 and 0.02127.

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_screw_diffusion_camera.py` from the repository root. Add `--generate` to regenerate the frozen values.
