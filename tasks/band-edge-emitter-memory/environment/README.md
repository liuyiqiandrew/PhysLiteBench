# Band-edge emitter memory

An infinite one-dimensional tight-binding guide has one site per integer and nearest-neighbor hopping +1. Its one-excitation band is [-2,2]. There are two identical side emitters labelled 0 and 1, no direct hopping between them, and no other excitations. For scale parameter lambda>0, each emitter has energy 2+lambda^(4/3)*delta and couples with real matrix element gain*lambda to one site. Emitter 0 couples at site 0. Emitter 1 couples at the integer nearest to `separation`/lambda^(2/3). The time evolution is the coherent Schrödinger evolution of these guide sites and emitters, with hbar=1 and every on-site guide energy zero.

For `registered`, take either the given `detuning` as delta, or if `locked_order` is supplied set delta exactly to -(pi*locked_order/separation)². The guide starts empty and the emitter excitation vector is the normalized `initial` phasor. Evolve for `scaled_time`/lambda^(4/3). Project on the state with guide empty and normalized emitter excitation phasor `probe`. Return the probability of that outcome, taking lambda to zero after the finite-lambda probability. Each phasor is two [real,imaginary] pairs in emitter order. The guide is unbounded throughout the limit.

For `isolated_transfer`, use one emitter resonantly coupled to one isolated mode, with both on-site energies zero and coupling matrix element gain. Initialize the emitter excited and the mode empty. Return the probability in the mode at the given time `setting`.

The common unknown gain is in [.59,1.35]. Separation is in [.70,3.3], scaled time in [.15,3.45] and isolated setting in [.17,1.07]. For an unlocked preparation, detuning is in [-4.8,1.60] with magnitude at least .15. For negative unlocked detunings, sqrt(-detuning)*separation differs from every positive multiple of pi by at least .10. Locked order is 1 or 2. All settings use the same time and energy units.

Implement `Model.fit(records)`, returning self and storing the inferred value as `self.gain`. Each calibration row has `input`, `value` and positive Gaussian standard deviation `sigma`. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in input order.
