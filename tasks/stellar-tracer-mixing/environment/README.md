# Echoes in a stellar stream

Two labelled collisionless stellar populations move on a periodic line of length 2π. Use dimensionless length, time and mass-density units. Initially the mass density per unit velocity of population s at every position is Gaussian, with total density 0.62 for A and 0.38 for B. Its mean velocity is the corresponding entry of `drift`. The velocity standard deviations are σ for A and 1.25σ for B. The same positive σ in [0.8,1.2] applies throughout.

Between preparations each star follows dx/dt=v and dv/dt=g(x,t). The periodic gravitational acceleration has zero spatial mean and obeys ∂g/∂x = −strength (ρ_A + ρ_B − 1), where each ρ is the integral of the current velocity distribution. The imposed uniform background removes the mean source. The force acts equally on both populations. There are no collisions, noise sources or external fields between preparations. Use the continuum limit. All allowed settings are linearly Jeans stable.

A numerical experiment prepares two tidal velocity patterns, which can be assigned separately to the two populations. Each experiment starts from the uniform state. At time zero add ε₁ d₁ₛ cos(k₁x) to the velocity of every star of type s. At time T add ε₂ d₂ₛ cos(k₂x) in the same manner, using its position at that instant. Positions stay continuous. These externally imposed velocity maps are the only operations that depend on the population label. Keep the self-gravity of the system on throughout, including after the second operation.

A spectrometer observes the difference harmonic k=k₂−k₁ of one population. For its velocity distribution f(x,v,t) it returns

    Re[ exp(−i angle) (1/2π) ∫dx exp(−ikx) ∫dv W(v) f(x,v,t) ].

If `band` is null the response W is 1. Otherwise `band` = [center, width] specifies W(v)=exp(−(v−center)²/(2 width²)), with unit peak transmission. At time t=(k₂/k)T+delay take the mixed derivative of the reading with respect to ε₁ and ε₂ at zero amplitudes. The requested signal is the limit of this derivative divided by T as T → ∞ at fixed delay. Each derivative is taken before the long-delay limit. The spectrometer measures mass density; it does not normalize by the fraction in the selected band.

Each input has `k` = [k₁,k₂], integers with 1 ≤ k₁ < k₂ ≤ 5; `first` and `second` are the pairs of d coefficients in A,B order, each in [−1,1]. The two drift speeds lie in [−0.9,0.9], `strength` in [0,0.5], `delay` in [−2,2] and `angle` in radians. The observed `population` is "A" or "B". Band centers lie in [−2,2] and widths in [0.25,2]. Calibration uses the same procedure with self-gravity switched off. Values are rounded to eight decimal places.

Implement Model.fit(records) returning self and setting self.dispersion, and Model.predict(inputs) returning a finite real one-dimensional NumPy array in input order. Each calibration row has an input and its measured value.

The numerical accuracy requested for a mixed echo is 0.00045 in the normalized signal units above. For calibration the absolute tolerance is 0.00002.
