# Slowly controlled classical particle

All quantities are dimensionless. A classical particle of unit mass has canonical position q and momentum p and Hamiltonian

    H(q,p;s,delta) = p²/2 + q⁴/4 − s q²/2 + delta s^(3/2) q.

This expression fixes the energy zero. Each experiment specifies `delta_final`, `path_slope`, and `s_final`. During a ramp of duration tau, the controls are

    s(t) = .05 + (s_final − .05) [3(t/tau)² − 2(t/tau)³],
    delta(t) = delta_final + path_slope log[s(t)/s_final],
    0 ≤ t ≤ tau.

After the ramp the controls remain at their final values. Before every shot, the particle is freshly prepared at the corresponding initial Hamiltonian. The connected-orbit action J=(1/(2*pi))*integral p dq is uniform on [J0,1.2*J0]. Independently, its position on that orbit is uniform in orbital time over one initial period. The unknown J0 is common to the experiments.

There is no friction, thermal bath, noise, collision or measurement feedback. The readout is the ensemble mean final total energy, reconstructed from final position and momentum. Average over the specified preparation at each finite tau, then take tau to infinity. The idealization is classical throughout.

The allowed domains are J0 in [.12,.16], delta_final in [.15,.25], path_slope in [0,.12], and s_final in [1.8,2.2]. Position, momentum and time units and all control coefficients are known.

## Data and interface

The calibration file is `data/calibration.json`. Each record has `input`, a measured `value`, and the fixed instrument standard deviation `sigma=.001` in energy units. Repeated readings are independent Gaussian measurements. Each input has `delta_final`, `path_slope`, and `s_final` in the ranges above.

Implement `Model` in `model.py`:

- `Model().fit(records)` estimates J0, stores it as `model.action_scale`, and returns the model.
- `model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` containing mean final total energies in the input order.

Prediction inputs use the same schema and allowed ranges as calibration inputs.
