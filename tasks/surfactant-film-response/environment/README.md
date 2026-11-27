# Forced surfactant film

Two immiscible, incompressible Newtonian liquids are trapped between parallel rigid plates. In equilibrium their common material interface is the plane z = 0. The lower plate is at z = −h_lower and the upper at z = +h_upper. Both plates are stationary with no slip. Both liquids adhere to the film. The cell is laterally unbounded. Neglect gravity, inertia of both phases and film, and all terms beyond linear response about rest.

An insoluble surfactant has uniform equilibrium areal density Γ₀. It diffuses in the interface with diffusivity 0.18. A small local fractional density change c = (Γ−Γ₀)/Γ₀ shifts the equilibrium tension from 1.3 to 1.3 − 0.8 c. The monolayer is an isotropic two-dimensional Newtonian fluid, with dilatational viscosity 0.42 and shear viscosity `surface_viscosity` in [0.75, 1.45]. Dilatational viscosity multiplies areal dilation in the isotropic viscous surface stress; shear viscosity multiplies twice the traceless surface strain rate. The film has no bending rigidity or connections to the plates.

A spatially patterned actuator applies an external force per unit area directly to the film. Its three components are ε Re{(fx, fy, fz) exp[i(k x − omega t)]}, with real amplitudes and positive z pointing toward the upper plate. After transients have decayed, an optical probe measures the complex velocity amplitude of material on the film divided by ε in the limit ε → 0. `part` selects the real or imaginary part of this coefficient in the stated exponential convention. Each run starts from the uniform equilibrium state.

## Units and data

All numbers use length 10 µm, time 0.1 s and bulk viscosity 1 Pa s as base units. Derived units include 100 µm/s for velocity, 10 Pa for force per unit area, 0.1 mN/m for tension, and 10 µPa s m for surface viscosity.

Each input supplies `h_lower` and `h_upper` in [0.45, 1.9], the corresponding bulk viscosities `mu_lower` and `mu_upper` in [0.45, 1.7], wavenumber `k` in [0.65, 2.1] and angular frequency `omega` in [0.25, 4.5]. Each of `fx`, `fy`, `fz` lies in [−1, 1]. `axis` is x, y or z and `part` is real or imag.

`data/calibration.json` contains records with an `input` dictionary, a scalar lock-in reading `value`, and its standard uncertainty `sigma`. The lock-in rounds readings to the nearest 0.0001. Quantization is the only measurement error. Implement `Model.fit(records)` to set `surface_viscosity` and return the model. `Model.predict(experiments)` takes a list of input dictionaries and returns a one-dimensional real NumPy array in input order.
