# Forced surfactant film

Two immiscible, incompressible Newtonian liquids are trapped between parallel rigid plates. In equilibrium their common material interface is the plane z = 0. The lower plate is at z = −h_lower and the upper at z = +h_upper. Both plates are stationary with no slip. Both liquids adhere to the film. The cell is laterally unbounded. Neglect gravity, inertia of both phases and film, and all terms beyond linear response about rest.

An insoluble surfactant has uniform equilibrium areal density Γ₀. It diffuses in the interface with diffusivity 0.18. A small local fractional density change c = (Γ−Γ₀)/Γ₀ shifts the equilibrium tension from 1.3 to 1.3 − 0.8 c. The monolayer is an isotropic two-dimensional Newtonian fluid, with dilatational viscosity 0.42 and shear viscosity `surface_viscosity` in [0.75, 1.45]. Dilatational viscosity multiplies areal dilation in the isotropic viscous surface stress; shear viscosity multiplies twice the traceless surface strain rate. The film has no bending rigidity or connections to the plates.

A spatially patterned actuator applies an external force per unit area directly to the film. Its three components are ε Re{(fx, fy, fz) exp[i(k x − omega t)]}, with real amplitudes and positive z pointing toward the upper plate. After transients have decayed, an optical probe measures the complex velocity amplitude of material on the film divided by ε in the limit ε → 0. `part` selects the real or imaginary part of this coefficient in the stated exponential convention. Each run starts from the uniform equilibrium state.

## Units and data

All numbers use length 10 µm, time 0.1 s and bulk viscosity 1 Pa s as base units. Derived units include 100 µm/s for velocity, 10 Pa for force per unit area, 0.1 mN/m for tension, and 10 µPa s m for surface viscosity.

Each input supplies `h_lower` and `h_upper` in [0.45, 1.9], the corresponding bulk viscosities `mu_lower` and `mu_upper` in [0.45, 1.7], wavenumber `k` in [0.65, 2.1] and angular frequency `omega` in [0.25, 4.5]. Each of `fx`, `fy`, `fz` lies in [−1, 1]. `axis` is x, y or z and `part` is real or imag.

`data/calibration.json` contains records with an `input` dictionary, a scalar lock-in reading `value`, and its standard uncertainty `sigma`. The lock-in rounds readings to the nearest 0.0001. Quantization is the only measurement error. Implement `Model.fit(records)` to set `surface_viscosity` and return the model. `Model.predict(experiments)` takes a list of input dictionaries and returns a one-dimensional real NumPy array in input order.

## Spontaneous fluctuations

In a second experiment the actuator is off. The same cell measures the height h and fractional density c in the stationary fluctuating film. Each plate has its own thermostat. The film is held at absolute temperature T₀. In each liquid the local temperature interpolates linearly from the plate temperature at its wall to T₀ at the film. The prescribed profile affects only the noise. The deterministic transport coefficients and the stated equation of state are fixed throughout the cell. Neglect evaporation, thermophoresis, convection of heat and temperature fluctuations.

Use classical fluctuating hydrodynamics with local equilibrium. The two bulk liquids have independent Newtonian thermal stress fluctuations at their local temperatures. The film's own viscous stress and conservative surfactant diffusion current fluctuate at T₀. These microscopic sources are mutually independent. In a film area dA the quadratic free-energy density for c and h is (0.8 c² + 1.3 |∇h|²)/2. The stated Fick diffusivity applies to this same density field. The ideal thermostats and the optical readout apply no mechanical forces.

For each field a = c or h, take its spatial Fourier coefficient at (k,0) with kernel exp(−ikx) and normalization 1/√A over periodic area A containing whole wavelengths. Its temporal coefficient a(k,ω) uses the kernel exp(+iωt) over a record of length τ, without additional normalization. The reported two-sided spectrum for pair ab is the large-τ limit of ⟨a(k,ω)b(k,ω)*⟩/(2 k_B T₀ τ). Use the dimensionless base units above throughout, including for energy and area. Under this normalization a white dissipative channel with mobility m at T₀ has normalized noise spectrum m.

A spectral input has the usual geometry, viscosities, `k` and `omega`. It also supplies `temperature_lower` and `temperature_upper`, the plate temperatures divided by T₀ in [0.45, 1.8], `pair` in {cc, hh, ch, hc} and `part` in {real, imag}. It has no drive or axis fields. Return the selected scalar in the same prediction array as the driven measurements. Calibration spectra are stationary ensemble values rounded by the same meter.

## Travelling thermal grating

A second thermostat setting superimposes a weak travelling temperature pattern on the mean profile. The coefficients of tension, viscosity and diffusion stay fixed. The film and both plates can be patterned. At boundary j, add aⱼ cos χ + bⱼ sin χ to its temperature divided by T₀. Within each liquid interpolate the instantaneous temperature linearly in z between its two boundaries, point by point. The local-equilibrium noise law above uses this prescribed instantaneous temperature. The apparatus maintains the profile externally, and all deterministic dynamics are still linearized about a resting flat film.

In this readout, the two optical channels use different lateral wavevectors and frequencies in the same laboratory frame. The first uses **k₁** = (k, 0) and ω₁ = omega. The second uses **k₂** = k2 (cos θ, sin θ) and ω₂ = omega2. The thermostat phase is χ = (**k₁** − **k₂**)·**r** − (ω₁−ω₂)t. Use the periodic-area convention above, with an area compatible with both wavevectors. The normalization is unchanged. For pair ab the meter takes the ensemble average of a(**k₁**,ω₁)b(**k₂**,ω₂)* over the phase-locked records. The constitutive laws for stress and surfactant diffusion are isotropic in the plane.

These spectral records include a `grating` dictionary. It gives `k2` and `omega2` in the same ranges and units as the first probe, `angle` = θ in [0.3, 2.8] radians, and `amplitudes` = [[a_film,b_film],[a_lower,b_lower],[a_upper,b_upper]]. Every amplitude lies in [−0.25, 0.25]. The film's mean temperature remains T₀. Plate temperatures outside this dictionary specify the means. With no `grating` key the original single-mode protocol applies.
