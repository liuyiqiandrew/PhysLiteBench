# Active force in a magnetic trap

A classical charged particle moves in an unbounded two-dimensional plane in an isotropic harmonic trap with potential `spring * |x|²/2`. Its position, velocity and active force are two-component vectors x, v and f. In reduced units, Boltzmann's constant and the charge are one. The drag coefficient is gamma=1, and the equilibrium solvent temperature is T=.7. For each positive particle mass m, the full dynamics are

    dx = v dt
    m dv = (−gamma v + field J v − spring x + f) dt
           + sqrt(2 gamma T) dW
    df = (−I + chirality J) f dt/tau + sqrt(2 D) dW_f/tau
    tau = memory_ratio * m/gamma
    J = [[0,1],[-1,0]].

W and W_f are independent standard two-dimensional Wiener processes, with independent Cartesian components. The active force generator is maintained externally; it is not a second equilibrium thermal reservoir. There is no additional force, damping, feedback or interaction. The ideal harmonic trap and force generator remain valid over the full range of parameters.

The single unknown D, stored as `activity`, lies in [.4,1.2]. All other controls are known: `spring` in [.6,1.6], signed `field` in [-3,3], signed `chirality` in [-3,3], and positive `memory_ratio` in [.2,3]. D is common to all experiments. Every finite-mass preparation reaches the stationary state of the full (x,v,f) process before measurement.

The reported apparatus uses the joint limit m→0 with every listed control held fixed, so tau decreases according to the specified relation. Two measurement settings are available:

- A position detector measures the stationary two-time correlation ⟨x_i(t+lag)x_j(t)⟩ at a fixed `lag` in [0,3], then takes the joint limit. `readout` is `xx`, `xy`, or `yy`; `xy` means i=0 and j=1, in that order.
- A work detector measures the active generator's work on the particle, `integral_0^(m/gamma) f(t) dot v(t) dt`, in a window lasting one inertial relaxation time. It averages this work in the full finite-mass stationary state and then takes the joint limit. `readout` is `work`. Positive work means energy delivered by the active force. This is an ordinary time integral of the continuous finite-mass force and velocity; it contains no contribution from the solvent force.

Each experiment dictionary has `spring`, `field`, `chirality`, `memory_ratio` and `readout`. Position readouts additionally have `lag`. All means are zero by the inversion symmetry of the stationary preparation. No initial transient is included in either measurement.

`data/calibration.json` is a list of position-correlation records with `input`, measured `value`, and independent Gaussian uncertainty `sigma`. Controls and lag values are exact. The fixed instrument uncertainty is independent of D.

Implement `Model.fit(records)`, returning self and storing the fitted `activity`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Use the same fitted D for all controls and readouts.
