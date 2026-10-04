# Proposed physical input — not an evaluated task

One enzyme has three conformations labelled 0, 1 and 2. It is held in a well-mixed solution at a fixed temperature. Use kB*T=1 as the energy unit and seconds as the time unit. The conformation energies E_i are known. No external mechanical load or time-dependent field acts on the enzyme.

For each oriented conformation edge i to (i+1) modulo3, there are two elementary reactions. Route a converts one molecule of fuel F_a into product P_a during a forward conformation change; its reverse reaction performs the reverse conversion. All reservoirs are ideal and sufficiently large to retain their specified chemical potentials. Their known differences are A_a=mu(F_a)-mu(P_a). The two fuel/product pairs are distinct. There are no other reactions or relaxation mechanisms.

For each edge i and route a, the known positive attempt factor is nu_ia. The forward and reverse rates are respectively

    gamma*nu_ia*exp([A_a-(E_(i+1)-E_i)]/2),
    gamma*nu_ia*exp(-[A_a-(E_(i+1)-E_i)]/2).

The same unknown positive gamma applies to every setting. The rates above define the ideal chemical kinetic model. No diffusion, binding delay, saturation or time-dependent reservoir response is included. The enzyme and reservoirs are prepared in the stationary regime before each reading.

An external reversible regeneration apparatus restores each fuel/product pair to its initial composition at the same fixed temperature and chemical potentials. The measured quantity is the long-time mean work supplied by that apparatus per enzyme and per unit time, with positive work meaning energy supplied to restore the reservoirs. Preparatory work and the regeneration apparatus's own losses are excluded. Repeated independent measurements have fixed known uncertainty.

Proposed controls: gamma in [.6,1.4], E_0=0 and E_1,E_2 in [-.3,.3], attempt factors in [.5,1.5], both A_a in [.2,2.4]. These are ranges for this outline, not frozen task inputs. The prototype uses a diagnostic subset with A_0 in[1.6,2.4], A_1 in[.2,.6], and attempt factors in[.6,1.4]. Calibration will use several settings within the declared range. Fit gamma jointly from the calibration and predict the requested regeneration powers.

An eventual package would use the existing neutral full-replacement instruction and Model.fit/predict interface. This outline does not mandate any numerical method. It contains the complete physical reaction and measurement assumptions without prescribing a formula for the readout.
