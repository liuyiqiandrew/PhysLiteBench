# Coupled harmonic calorimeters

Two classical main oscillators of unit mass have Hamiltonian

    H_main = (p_1²+p_2²)/2 + k_1 q_1²/2 + k_2 q_2²/2 + c(q_1−q_2)²/2.

All quantities use reduced units with Boltzmann's constant one. Main oscillator i is connected by an ideal harmonic spring to a separate coupler with coordinate y_i, momentum P_i, and known finite mass M. The link spring stiffness is k_link=gamma/tau. The common unknown gamma lies in [0.3,1.2]; the independently characterized ratio tau=gamma/k_link is a known control called `memory`. The link and coupler have energies k_link*(y_i−q_i)²/2 and P_i²/(2M). These and H_main are all stored energies of the apparatus. No mass is taken to zero. The harmonic model is exact throughout the stated domain.

Each coupler is attached to its own ideal equilibrium Langevin thermostat at temperature T_i. It experiences viscous force -gamma*P_i/M and a zero-mean Gaussian white force xi_i with covariance

    <xi_i(t) xi_j(t')> = 2*gamma*T_i*delta_ij*delta(t−t').

The two thermostats are independent. All other forces are the spring forces from the stated Hamiltonian. There are no additional damping, noise, work sources, or reservoir couplings. Positions obey their momentum divided by mass. Every preparation reaches the unique stationary joint state of all main and coupler coordinates and momenta before measurement. Temperatures and mechanical controls then stay fixed.

Three measurement settings are available. In a mode-correlation measurement, let O contain the orthonormal eigenvectors of the main Hamiltonian's 2×2 stiffness matrix, ordered by increasing eigenvalue, and let p_mode=Oᵀp. The detector reports <p_mode_alpha(t+lag)*p_mode_alpha(t)>. The arbitrary sign of an eigenvector has no effect.

Each thermostat also has a calorimeter at its interface with its own coupler. It measures signed heat supplied by that thermostat to the apparatus, positive into the apparatus. `heat_current` reports its stationary mean power. `heat_variance` reports Var(Q_i) for the heat Q_i accumulated during a stationary measurement interval of length `duration`; it is not divided by duration. All physical coupler degrees of freedom remain present during the interval. The two calorimeters distinguish the original thermostat indices.

Main springs k_i lie in [0.8,1.4], coupling c in [0.05,1.2], tau in [0.15,2], mass M in [0.05,0.3], and temperatures T_i in [0.5,2]. Correlation lags lie in [0,6] and heat measurement durations in [0,3]. The single gamma applies to every preparation.

Each input has `springs: [k1,k2]`, `coupling`, `memory`, `coupler_mass`, `temperatures: [T1,T2]`, and `readout`. For `mode_correlation`, it also has `mode` (0 or 1) and `lag`. For `heat_current`, it has `bath` (0 or 1). For `heat_variance`, it has `bath` and `duration`. All controls are exact.

`data/calibration.json` contains mode-correlation and stationary mean-power measurements, each with `input`, measured `value`, and independent Gaussian uncertainty `sigma`. The instrument uncertainties are fixed independently of gamma and of the noiseless response.

Implement `Model.fit(records)`, storing the fitted scalar `friction` and returning self. `predict(experiments)` must return a finite NumPy array of shape `(len(experiments),)` in input order. Any implementation satisfying this interface may be used.
