# Coupled harmonic calorimeters

Two classical oscillators of unit mass have Hamiltonian

H = (p_1²+p_2²)/2 + k_1 q_1²/2 + k_2 q_2²/2 + c(q_1−q_2)²/2.

All quantities are in reduced units, with Boltzmann's constant equal to one. Oscillator i is attached to its own local thermal reservoir at temperature T_i. The two reservoirs are independent. They have the same unknown friction coefficient γ and the same known memory time τ in each preparation. Their force variables z_i give the following full dynamics:

    dq_i = p_i dt
    dp_i = (−∂H/∂q_i + z_i) dt
    dz_i = −(z_i + γ p_i) dt/τ + sqrt(2 γ T_i)/τ dW_i

W_1 and W_2 are independent standard Wiener processes. There are no additional forces, damping terms or reservoir couplings. The harmonic Hamiltonian is an ideal model valid over the whole stated range. Every preparation is allowed to reach its unique stationary state before measurement; the joint stationary state includes q, p and the reservoir forces z. Temperatures and all mechanical parameters remain fixed during measurement.

Two measurement settings are available. In a mode-correlation measurement, let O contain the orthonormal eigenvectors of the Hamiltonian's 2×2 stiffness matrix, ordered by increasing eigenvalue, and P=Oᵀp. The detector reports the stationary autocorrelation ⟨P_α(t+lag)P_α(t)⟩. This does not depend on the arbitrary sign of an eigenvector. In a heat-current measurement, a force calorimeter on reservoir i reports the stationary mean power ⟨p_i z_i⟩ delivered by that reservoir to the mechanical pair. The force z_i is continuous, so this power needs no stochastic integration convention. Positive current means energy enters the mechanical pair.

The single unknown γ lies in [0.3, 1.2]. Inputs have springs k_i in [0.8, 1.4], coupling c in [0.05, 1.2], memory τ in [0.15, 2], temperatures T_i in [0.5, 2], and lag in [0, 6]. Coupling is strictly positive, so the two mode eigenvalues are distinct. Each record contains all controls for its preparation; γ is common to every preparation.

`data/calibration.json` is a list of records with `input`, measured `value`, and independent Gaussian standard deviation `sigma`. The calibration measures mode correlations. The instrument uncertainty is specified independently of γ.

Implement `Model` in `model.py`:

- `fit(records)` stores the fitted scalar `friction` and returns `self`.
- `predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order.

Every input has `springs: [k1, k2]`, `coupling`, `memory`, `temperatures: [T1, T2]`, and `readout`. For `readout: "mode_correlation"`, it also has `mode` (0 or 1) and `lag`. For `readout: "heat_current"`, it has `bath` (0 or 1). Reservoir indices follow the original oscillator coordinates, and mode indices follow the ordered stiffness eigenvalues.
