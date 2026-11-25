# Thermal motion of a taut filament

A straight, overdamped filament has material coordinate x in [0,pi], fixed transverse displacement u(0,t)=u(pi,t)=0, and constant tension one. In the linear small-deflection regime its elastic energy is (1/2) integral (partial_x u)^2 dx. The local transverse drag per length is a spatially uniform unknown gamma in [0.6,1.8]. Inertia, bending stiffness, hydrodynamic interactions and fluctuation-dependent tension are neglected. Temperature does not change the tension or drag.

Independent classical Markov reservoirs act at the fixed material positions. Boltzmann's constant is one. Their zero-mean transverse random force density xi has covariance

    <xi(x,t) xi(x',t')> = 2*gamma*T(x)*delta(x-x')*delta(t-t').

The deterministic local force is the variational elastic force minus gamma*partial_t u. There is no other forcing or feedback. Reservoir temperature is prescribed as T(x)=temperature*[1+contrast*cos(wavenumber*x+phase)]. It does not follow the fluctuating transverse displacement. temperature is in [0.005,0.04], contrast in [0,0.85], wavenumber is an integer from 1 through 8, and phase is in [-pi,pi]. All quantities use the reduced units above.

A displacement sensor reads Q(t)=integral w(x)*u(x,t) dx, where w(x)=sum from n=1 to 6 of weights[n-1]*sqrt(2/pi)*sin(n*x). The six real weights obey sum(weights^2)=1; they may have either sign. The filament has reached its stationary state. Return the two-sided angular-frequency power spectral density

    S_Q(omega)=integral from -infinity to infinity exp(i*omega*t)*<Q(t)*Q(0)> dt.

The input frequency is omega in [0,40]; there is no additional factor of two for positive frequencies. The sensor resolves only these six spatial modes; noise also acts on the remaining filament modes according to the same continuum model.

Calibration records have uniform reservoir temperature, several mixed sensor weights, and multiple frequencies including zero. Each record contains input, value and independent Gaussian standard deviation sigma. Implement Model.fit(records), returning self and setting Model.friction, and Model.predict(experiments), returning one finite NumPy spectral density per input. Fit gamma from data/calibration.json. You may revise the predictor for all allowed preparations.
