# Slowly controlled classical particle

All quantities are dimensionless. A particle of unit mass has canonical position q and momentum p, with controlled Hamiltonian

    H(q,p;s,delta) = p²/2 + q⁴/4 − s q²/2 + delta s^(3/2) q.

This expression fixes the energy zero. The asymmetry `delta` is known. Before each shot, s=.05. The particle is prepared on a connected periodic orbit whose action J=(1/(2*pi))*integral p dq is uniformly distributed on [J0,1.2*J0]. Its orbital phase is independently uniform in time over one initial period. The unknown positive preparation scale J0 is common to all experiments. Each shot is freshly prepared.

The particle is isolated during the subsequent motion, with no friction, bath, noise, collisions or measurement feedback. During a ramp of duration tau,

    s(t) = .05 + (s_final − .05) * [3(t/tau)² − 2(t/tau)³],  0 ≤ t ≤ tau.

Afterward s remains fixed. All potential coefficients, the control path and the position and momentum units are known. The readout is the ensemble mean final total energy H(q,p;s_final,delta), reconstructed from final position and momentum. Average over the stated preparation at each finite tau, then take tau to infinity. The idealization is classical throughout.

The input is a dictionary with `delta` in [.15,.25] and `s_final` in [.1,2.2]. The unknown J0 lies in [.12,.16].

## Interface

Implement `Model` in `model.py`:

- `Model().fit(records)` estimates J0, stores it as `model.action_scale`, and returns the model.
- `model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` containing the mean total energies in the input order.

Each calibration record contains `input`, measured `value`, and the known standard deviation `sigma`. Measurement errors are independent Gaussian errors. The records are in `data/calibration.json`. The input dictionary has the same format in calibration and prediction.
