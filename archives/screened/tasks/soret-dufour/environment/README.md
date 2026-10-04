# Heat and species relaxation

A neutral ideal binary liquid fills a sealed straight cell of length `length`, between 0.7 and 1.5 mm. The cell has no bulk flow, reaction, external force, or magnetic field. Both species have equal constant partial molar enthalpies, and the total molar density n=1000 mol/m³ is constant. Work in linear response about T0=300 K and mole fraction c0=0.5. The volumetric heat capacity at fixed composition is C=2e6 J/(m³ K). The coefficient relating heat flux to minus temperature gradient at uniform composition is kappa=0.2 W/(m K). These constants do not vary over the small departures used here.

The chemical potentials satisfy

    (mu_A-mu_B)/T = R*log(c/(1-c)),  R=8.314462618 J/(mol K).

The measured molar diffusion law in the zero-total-molar-flow frame is

    j_A = -n*D*[grad(c) + c0*(1-c0)*S*grad(T)],
    j_B = -j_A,  S=0.05/K.

D is a common unknown positive diffusivity between 0.01 and 0.08 mm²/s. Convert to SI when combining it with SI material coefficients. This cross-gradient response is a property of the same passive material in every experiment. Its microscopic dynamics are time-reversal invariant. Use linear irreversible thermodynamics at (T0,c0), retaining the response terms at that order. Heat flux is defined after subtracting transported partial enthalpies; the equal partial enthalpies here imply no composition contribution to the stored internal energy. The unobserved bath supplies whatever local heat is needed in isothermal experiments.

Each preparation has one known cosine mode:

    T(x,0) = T0 + temperature_amplitude*cos(mode*pi*x/length),
    c(x,0) = c0 + composition_amplitude*cos(mode*pi*x/length).

`mode` is 1, 2, or 3; |temperature_amplitude|<=0.3 K and |composition_amplitude|<=0.08. Both end walls always have zero species flux. For `isothermal: true`, a distributed external bath clamps T=T0 at every point for all times, and the initial temperature amplitude is zero. For `isothermal: false`, that bath is absent and both ends are thermally insulated: there is no heat flux through either wall. Coefficients are spatially uniform. The linear response approximation is the model to solve, including for these initial profiles.

At `time` between 0 and 10 s, the detector reports the coefficient of that same cosine in either T-T0 (`observable: "temperature_amplitude"`, in K) or c-c0 (`observable: "composition_amplitude"`, dimensionless). All other input fields are required. Calibration contains only composition-amplitude measurements while the distributed bath clamps temperature. Their errors are independent Gaussian errors with the listed standard deviations. The same fitted D applies when the bath is absent.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Store D in `Model.diffusivity`, in mm²/s. Records in `data/calibration.json` contain `input`, `value`, and `sigma`.
