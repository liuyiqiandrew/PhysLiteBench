# Two residence recorders in a glass

The two planted reporter depths place them on the time scale of the deepest ordinary basins. Their residence clocks record repeated visits. The line at the shutter measures which basin interrupted the motion. The complete-graph scaling follows [Fontes and Mathieu, K-processes, scaling limit and aging for the trap model in the complete graph](https://doi.org/10.1214/07-AOP360). The two clocks, planted basins and spectral readout specify this experiment.

## Eliminating the ordinary basins

Put α=T/T_g and v=N exp(−E/T_g). Each ordinary v is uniform on [0,N], so the limiting points form a Poisson process of unit intensity. In scaled time the escape rates are v^(1/α) on an ordinary site and r_i=exp(−δ_i/T) at reporter i. Upon arrival at ordinary site j the shutter wins against escape with probability h_j=s/(s+v_j^(1/α)). Write H=Σ h_j. Summing over repeated uniform flights gives probability m=1/(2+H) of next reaching each specified reporter and probability h_j m of ending on ordinary site j. This ratio already holds on a finite complete graph with the finite sum H. Its value stays fixed throughout one trajectory.

In residence time a departure from A reaches B at rate a=r_A m. From B the reverse rate is b=r_B m. Self returns leave the residence clock running. Set d_i=s+r_i(1−m). For y=0 the weights ending on reporters A and B are W_A=exp(−d_A x) and W_B=0. For positive y let t=√(abxy) and K=exp(−d_A x−d_B y). Alternating visits yield

    W_A = K abx I₁(2t)/t
    W_B = K a I₀(2t).

Here W_i is the occupation weight at which the terminal event can take place, including the density of the positive clock readings. Shuttering directly on i gives weight s W_i. Reaching the shutter during the final ordinary excursion gives (r_A W_A+r_B W_B) Hm. Average each joint weight and the total separately over specimens. For a finite band F replace H in the ordinary numerator by the sum of h_j over the band. The Poisson insertion identity gives E[Σ_{v_j∈F} h_j f(H)] = ∫_F h(v) E[f(H+h(v))] dv. The reference includes the inserted site in every previous excursion.

## Independent checks

The oracle uses Gaussian quadrature for H built from its Poisson cumulants κ_n=∫h(v)^n dv=s^α Γ(1+α)Γ(n−α)/Γ(n). It sums the alternating histories with Bessel functions. The independent reference inverts the characteristic function of H on a Fourier grid. It integrates its characteristic exponent against a beta density and sums the positive histories term by term. The two methods agree within 3 × 10⁻¹¹ on the probabilities and joint selection weights at all 24 settings. The smallest selection mass per unit x or joint density per unit x and y is 0.0109.

A further check keeps the two reporters and six ordinary basins as separate microscopic Markov states. Adding distinct Laplace penalties to the residence clocks and solving the eight-state shutter resolvent gives the probabilities of all three terminal classes. Integrating the proposed continuous densities and the zero-B component agrees within 2 × 10⁻¹³. The supplied worked example also exercises the zero-B atom, both positive clock densities and an ordinary spectral band.

The completed shortcut redraws the ordinary specimen on every departure. It preserves the exact single-excursion return and spectral probabilities but loses their correlation across returns. Its maximum errors are 0.0211 on the zero-B shots, 0.0314 on two positive recorders and 0.0359 on the host spectrum. The tolerance is 0.004. Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_aging_line_memory.py` to reproduce these checks.
