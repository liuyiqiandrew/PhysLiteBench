# Spin-body torque, revision 1

This is a staged, unevaluated candidate. No model trial has been run. The neutral instruction and the 600-second agent / 60-second verifier limits are unchanged.

The unknown is the magnetic moment magnitude M in [.8,1.2]. The public apparatus specifies an isotropic clamped macrospin, its spin angular momentum S=−M m/gamma, exact Gilbert dynamics, and a local damping channel delivering angular momentum to the measured support. The sensor reports body-on-support torque. It does not measure the opposite clamp-on-body torque. These assumptions are needed: a Gilbert channel connected to an unspecified external spin sink would not determine this mechanical readout.

The completed shortcut solves the exact nonlinear periodic magnetic response, then identifies M m cross B with the support torque. Its magnetic mathematics is correct. Angular momentum balance gives

    torque_support = M m cross B − dS/dt
                   = alpha*M/gamma * m cross (dm/dt).

The two torques have the same z component in the circular periodic state because S_z is constant. Calibration therefore identifies M exactly without distinguishing these physical interpretations. The x and y components differ by the time-dependent spin angular momentum. This is an observable error, rather than an omitted magnetic mode or inaccurate susceptibility.

## Exact periodic response and independent reference

In the rotating frame let n be the constant magnetization, q=n_z^2, delta=gamma*bias−frequency, c=gamma*amplitude and d=alpha*frequency. Normalization and Gilbert dynamics give

    (1−q)*(delta^2+d^2*q) = c^2*q,
    n_x+i*n_y = c*sqrt(q)/(delta+i*d*sqrt(q)).

Both completed models use the stable quadratic root. At resonance this reduces to q=1−(c/d)^2. The allowed finite drive satisfies c<d everywhere. The public north-pole preparation selects the north-connected periodic state; no weak-drive truncation is used. The oracle then applies the mechanical torque above.

The private reference integrates the full nonlinear laboratory-frame Gilbert equation from the public initial state through transients. It samples the final cycle at the requested phase and computes the external magnetic torque minus the numerical spin angular-momentum rate. It does not use the stationary root or the oracle's Gilbert torque expression.

A 484-control grid checks normalization, the Gilbert residual, DC equality, absorbed-power balance and tangent stability. The minimum n_z is .960645 and the largest real stability eigenvalue is −.133992. Fifteen corner/off-grid cases compare the oracle with the laboratory ODE and a longer, tighter integration: maximum torque disagreement is 1.99e−11 and the periodic residual is 4.07e−11. The cold reference for all hidden inputs takes .49 seconds. All selected transverse signals have magnitude at least .00253. One draft hidden phase was moved from −.8 to −1.2 before freeze to avoid a near-zero crossing; no public or dynamical formula changed.

## Calibration and grading

The 216 calibration records comprise twelve independent repetitions of eighteen field settings. Their constant instrument sigma is 1e−5, independent of M and the response. The calibration seed is 890113; validation uses 256 independent noise draws with seed 890119. Weighted least squares has strictly positive scalar curvature 1.6091e7. Noiseless recovery at M=.8,1,1.07,1.2 agrees to 2.3e−16. The true M is 1.07 and both controls fit 1.0696847401 in the checked-in data, with reduced chi-square .95936.

Hidden groups use normalized RMS torque error with limit .04. The reference/numerical uncertainty is many orders below this tolerance. The completed shortcut remains correct on the extra z-component anchors, while all three transverse groups fail. Across all 256 noise draws the oracle's maximum hidden error is .001014; the smallest shortcut transverse error is 4.928. All calibration and parameter checks pass. No threshold was tightened to obtain separation.

Local isolated pytest controls give oracle 8/8 and shortcut 5 passed / 3 hidden failures, in about .9 seconds each. Docker controls and any Luna-high runs remain for the parent evaluation controller; these local controls are not reported as model-evaluation evidence.

Validation: [spin-body-torque-validation.json](../../results/spin-body-torque-validation.json). The prior weak-drive feasibility prototype is preserved in repository `scripts/prototype_spin_body_torque.py` and `results/spin-body-torque-prototype.json`; it is not used by the final task. Final science and public-input peer reviews are recorded separately alongside provenance.

From the repository root, reproduce the staged science with:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/spin-body-torque-r1/scripts/validate_spin_body_torque.py
```

Add `--generate` only when intentionally regenerating both calibration copies. After promotion, use the same validator under `scripts/`.

## Primary source and scope

Keshtgar, Streib, Kamra, Blanter and Bauer, [Magnetomechanical coupling and ferromagnetic resonance in magnetic nanoparticles](https://arxiv.org/html/1610.01072v2), Physical Review B 95, 134447 (2017), develops the spin/mechanical angular-momentum balance and body-frame Gilbert damping. Its equations 8 and 18–21 support the gyromagnetic sign and the identification of damping with spin-to-lattice angular-momentum transfer; setting the body's angular velocity to zero gives this task's ideal clamp. The public model explicitly localizes that channel and neglects thermal noise, anisotropy and additional mechanical forces. It makes no claim that every experimental Gilbert damping mechanism deposits torque immediately in a chosen support.

The difficulty hypothesis is that a solver will fit the exact magnetic response while preserving the field-torque readout. The actual Luna-high success rate is unknown until evaluation.
