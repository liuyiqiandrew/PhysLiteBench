# Screw diffusion camera

A rigid particle diffuses by six independent infinitesimal body motions. The material origin follows a screw during each rotational kick. The tagged point picks up a further displacement when its body offset rotates. Optical selection, the displacement fringe and the monitored spin record all refer to the same trajectory. Translational and rotational cross-correlations and their dependence on the chosen tracking point are derived in [Cichocki, Ekiel-Jeżewska and Wajnryb](https://arxiv.org/abs/1502.03747).

## Tagged scattering

Let u=Rᵀe. A body rotation about axis j differentiates a function of u by Jⱼ=(u×eⱼ)·∇. The tag displacement per radian is u·cⱼ with cⱼ=hⱼ+eⱼ×b. For spatial Fourier wave number k, define Aⱼ(k)=Jⱼ+ik u·cⱼ. The backward solvent generator is

    L(k) = ν Σⱼ [rⱼ Aⱼ(k)² − dⱼ k² uⱼ²].

The square differentiates the coefficient of the multiplication term as well as the function. On uⱼ the zero-wave generator is −ν(Σr−rⱼ)uⱼ. This gives the exponential calibration curve and the fitted mobility. Let W₀ and W₁ multiply by the gate acceptance probabilities and ⟨·⟩ denote the initial uniform average. The free gated fringe has complex numerator ⟨W₀ exp[t₁ L(k₁)] W₁ exp[t₂ L(k₂)] 1⟩ and real denominator ⟨W₀ exp[t₁ L(0)] W₁ 1⟩.

## Projective spin record

Repeated projection onto a moving multidimensional subspace transports internal amplitudes and preserves coherence among unresolved levels. The body-mounted observer reports two magnetic bands. Between observations the laboratory spin has no Hamiltonian. Brownian angles have finite quadratic variation and generate finite band-change probabilities even in the limit of frequent observations. The continuum generator here follows from the actual finite-kick instrument. [Burgarth et al., Non-Abelian phases from quantum Zeno dynamics](https://doi.org/10.1103/PhysRevA.88.042107)

Let Bₐ be the four-by-two matrix of body spin basis columns for label a. After positive body rotation δ around axis j the free laboratory spin evolves inversely in the new body frame. With the outcome changing from a to b its unnormalized two-level state therefore changes by the Kraus matrix

    K_ba(δ) = B_b† exp(i δ Jⱼ) Bₐ.

For b=a put H=Bₐ†JⱼBₐ and D=Bₐ†Jⱼ²Bₐ. To second order, K=I+iδH−δ²D/2. The difference D−H²=Bₐ†JⱼB₁₋ₐB₁₋ₐ†JⱼBₐ accounts for amplitude lost into the other observed doublet. For b≠a the leading amplitude is iδB_b†JⱼBₐ. Its squared magnitude gives a jump rate after averaging δ²=2νrⱼ dt. The order-δ no-change derivative also shares its stochastic drive with the solvent's Aⱼ(k), producing the joint spatial and electronic cross derivative.

The calculation propagates unnormalized two-level density matrices, keeping a separate block for each accepted number of label changes. The current band follows from the parity of that count and the known initial band. At the filter switch B′_b†Bₐ is the correct Kraus matrix even when the particle has not moved. Retaining a change there increments the count. For weight readings the numerator is divided by optical acceptance. For selected readings its denominator uses zero waves, the identity electronic effect and the same requested count. Every selected record in the frozen data occurs in at least 9.7% of the optically accepted shots.

## Independent checks

The oracle expands the spin rotations and body derivatives together at the tracked tag. The reference directly applies the exact signed finite material screws and the complete spin-3/2 rotation, then projects back onto each possible outcome. The three tag positions appear through endpoint phases. The oracle uses degree six for spin readings. The reference uses degree seven and an angular differentiation step of 0.002 radians. A finer comparison raises the harmonic cutoff and halves the reference step. The 28 passive fringes and 12 rotational calibration values are also reproduced.

With identical unturned central filters, adjacent magnetic transitions connect each state to the other doublet at the common rate 3ν(r₀+r₁)/2. The zero-wave ungated count is Poisson. In a different check with upper filters, the initially coherent pair of +3/2 and +1/2 levels has different losses on the two components. For the unnormalized no-change state the populations evolve with the real matrix [[−a,a],[a,−a−b]], where a=3ν(r₀+r₁)/2 and b=2ν(r₀+r₁). Its real coherence decays at rate ν(r₀+4r₁+r₂). After total time 0.36 an equal initial superposition gives 0.4386716 and 0.2424262 for the opposite equatorial analyzers. The validator also rotates a second upper filter by π, interchanging its labels without changing its physical projections. This adds exactly one switch count for every particle trajectory and preserves the count-shifted fringe.

The completed shortcut measures the individual magnetic level after every kick, then groups outcomes into the published labels. This retains level transition rates and switch probabilities while erasing coherence inside each observed doublet. It passes the original solvent groups and misses the count weight, one-change and two-change families by 0.09812, 0.00581 and 0.00484. The public starter uses unpolarized, band-averaged rates and misses these groups by 0.12008, 0.00340 and 0.02756. The full oracle's largest discrepancy on the frozen spin readings is 0.00000039, against the public 0.003 tolerance.

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_screw_diffusion_camera.py` to regenerate the physics in memory and compare the controls. Add `--generate` to write the fixture file.
