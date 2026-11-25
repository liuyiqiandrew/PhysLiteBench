# Transverse heat transport in a bosonic lattice

A two-dimensional square lattice has two internal bosonic states at each site, with lattice spacing, hbar and k_B equal to 1. The two internal states share the same physical position. In the fixed, momentum-independent internal-state basis its complete number-conserving quadratic Hamiltonian is H=sum_k b_k^dagger h(k)b_k, with

    h(k) = exchange_scale * [4.5*I + sin(k_x)*sigma_x
           + handedness*sin(k_y)*sigma_y
           + (mass + cos(k_x) + cos(k_y))*sigma_z].

The sigma matrices are the usual Pauli matrices. Momenta lie in the first Brillouin zone [-pi,pi]^2. The real control `mass` is in [-1.6,-.6] and `handedness` is either -1 or 1. Both bands are positive and separated over this range. The only unknown is the common positive `exchange_scale` in [.8,1.2], in frequency units. There are no anomalous pair terms or interactions between excitations. The model has no other heat-carrying degrees of freedom.

There are two measurements. With `observable="frequency"`, a weak momentum-resolved spectroscopic probe measures the single-excitation frequency at the specified two-component `wavevector`. `band=0` selects the lower band and `band=1` the upper band. The probe does not change the Hamiltonian or its parameters.

With `observable="thermal_hall"`, reservoirs establish local Bose equilibrium at fixed chemical potential zero. The mean `temperature` is in [.35,1.5]. They impose an infinitesimal, macroscopically smooth temperature gradient in the y direction. External calorimeters measure net energy transferred across the sample in the x direction. The requested output is the intrinsic antisymmetric transport coefficient kappa_xy, with the convention j_energy,x=-kappa_xy*partial_y(temperature); x,y,z form a right-handed frame. Energy and heat coincide at this fixed chemical potential. Particle exchange with the reservoirs is permitted and no zero-particle-current constraint is imposed.

This measurement uses the thermodynamic and local-equilibrium linear-response limits, followed by the clean limit. Weak equilibration establishes the temperature profile before its broadening is taken to zero. Only the intrinsic antisymmetric coefficient of the stated Hamiltonian is measured; there is no disorder, skew scattering, phonon contribution or other background thermal response. The calorimeter readout is energy delivered through the sample, with a common temperature giving zero net delivered energy. The two internal states are colocated as stated, so no additional intracell position or orbital embedding is present.

All experiments contain `observable`, `mass` and `handedness`, together with the relevant measurement controls above. Calibration uses band-frequency measurements at varied momenta, masses and handedness. Each record in `data/calibration.json` contains `input`, measured `value` and independent Gaussian uncertainty `sigma`, fixed at .002 independently of exchange_scale and the response. All controls are exact.

Implement `Model.fit(records)`, returning self and storing `exchange_scale`. `Model.predict(experiments)` must return a finite NumPy array of shape `(len(experiments),)` in input order. Any implementation satisfying this interface may be used.
