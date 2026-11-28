# Screw diffusion camera

A rigid particle diffuses by six independent infinitesimal body motions. The material origin follows a screw during each rotational kick. The tagged point picks up a further displacement when its body offset rotates. Optical selection, the displacement fringe and the gyroscope area all refer to the same trajectory. Translational and rotational cross-correlations and their dependence on the chosen tracking point are derived in [Cichocki, Ekiel-Jeżewska and Wajnryb](https://arxiv.org/abs/1502.03747).

## Tagged scattering

Let u=Rᵀe. A body rotation about axis j differentiates a function of u by Jⱼ=(u×eⱼ)·∇. The tag displacement per radian is u·cⱼ with cⱼ=hⱼ+eⱼ×b. For spatial Fourier wave number k, define Aⱼ(k)=Jⱼ+ik u·cⱼ. The backward solvent generator is

    L(k) = ν Σⱼ [rⱼ Aⱼ(k)² − dⱼ k² uⱼ²].

The square differentiates the coefficient of the multiplication term as well as the function. On uⱼ the zero-wave generator is −ν(Σr−rⱼ)uⱼ. This gives the exponential calibration curve and the fitted mobility. Let W₀ and W₁ multiply by the gate acceptance probabilities and ⟨·⟩ denote the initial uniform average. The free gated fringe has complex numerator ⟨W₀ exp[t₁ L(k₁)] W₁ exp[t₂ L(k₂)] 1⟩ and real denominator ⟨W₀ exp[t₁ L(0)] W₁ 1⟩.

## Pinned gyro trajectories

The gyro coordinates are two independent planar Brownian drivers with diffusion constant ν. Their realization also rotates and translates the tracked particle. A planar pin fixes the integrated driver displacement and can leave a nonidentity final body rotation. Lévy's conditional stochastic-area law gives the public zero-wave benchmark. The heat-kernel connection with a constant magnetic field follows from the joint law of a Brownian endpoint and its stochastic area. [Helmes and Schwane, Lévy's stochastic area formula in higher dimensions](https://doi.org/10.1016/0022-1236(83)90053-8)

For oscillator operators P and Q with [P,Q]=−if define W(d)=exp[i(d₁P+d₂Q)]. Chronological multiplication of the finite gyro translations produces exp(ifA)W(d), where A is the signed polygonal area and d is the final displacement. The identity Tr W(d)=(2π/|f|)δ²(d) turns the trace with W(−d) into a pinned area kernel. Dividing by the free planar endpoint density multiplies that trace by 2|f|νt exp[|d|²/(4νt)]. The camera gates and the tagged spatial phase are propagated together with the oscillator.

In the two selected body directions, the joint generator replaces Aₐ(k) by Aₐ(k)+iP/√rₐ and A_b(k) by A_b(k)+iQ/√r_b. For a final pin alone take one trace after both intervals and the intermediate gate. For a checkpoint p take one pinned transfer for each interval, still propagating the common intermediate particle orientation. If d=end−p, the global polygon includes the extra triangle (p₁d₂−p₂d₁)/2. This follows by shifting the second arc's starting point from the origin to p.

The denominator is an ordinary two-dimensional Fourier inversion with no area phase and zero spatial wave. Only harmonics through degree two contribute because the terminal function is the middle gate and all body derivatives preserve degree. For a final pin the endpoint constraint spans both intervals. With a selected checkpoint, the second independent gyro increment contributes only its scalar density to that denominator.

The oracle uses the harmonic-oscillator ladder for P and Q and the tagged body generator. The reference represents one gyro translation by a Fourier derivative on a finite line and the other by a linear coordinate. It advances signed finite body screws at the material origin and restores all three tag offsets. Its endpoint displacement separately modulates, translates and modulates in the coordinate representation. It computes the denominator with a rectangular Fourier quadrature. Refining the angular cutoff, oscillator basis, Fourier box and screw step checks the frozen readings.

At the calibrated mobility the two methods differ by at most 0.0000074 on the 18 gyro readings, against the public 0.003 tolerance. In a separate sine-quadrature experiment changing the horizontal gyro endpoint from +1.1 to −1.1 leaves the unweighted area law unchanged. The full signals are +0.07903115 and −0.07038281 because the very same rotations move the off-origin tag.

The completed comparison draws the gyro record and the solvent trajectory independently. It keeps the exact signed area law for an endpoint and for both arcs of a checkpoint, the deterministic triangle, and the ordinary tagged-particle fringe. The public starter instead replaces the gyro record by straight chords between the supplied positions. Both models fit the rotational calibration and propagate the passive camera. The independent-record model misses the final-pin, checkpoint and oriented-area groups by 0.05372, 0.11381 and 0.05554. The straight-chord approximation misses them by 0.33682, 0.16649 and 0.11761.

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_screw_diffusion_camera.py` from the repository root to regenerate and compare the observables. Add `--generate` to write the frozen targets.
