# Momentum measurements of a uniform condensate

A uniform three-dimensional bosonic gas occupies a cubic periodic box of volume V. Use hbar=k_B=1 and known particle mass m=1. Its homogeneous condensate has zero momentum and fixed known density n0=10. All nonzero momentum modes are described by the following specified quadratic Hamiltonian:

    H = sum_(p != 0) [(epsilon_p + interaction) a_p^dagger a_p
        + interaction/2 * (a_p^dagger a_(-p)^dagger + a_p a_(-p))]
    epsilon_p = |p|^2/(2m),  p = 2*pi*(integer vector)/V^(1/3).

The operators obey the usual boson commutation relations. The known nonnegative control `interaction` is the product of contact coupling and condensate density and lies in [0,1]. An additive constant sets the ground-state energy to zero and has no effect on the statistical state. The quadratic model is the exact ideal model for this task: retain its full dispersion and mode mixing, with no interaction between its excitations, condensate evolution, or additional many-body correction. There is no external potential, flow, rotation or lattice.

Before each shot, a preparation bath equilibrates the system to the full Gibbs state exp(-H/temperature)/Z, without constraining the total momentum. The bath can exchange momentum with the gas and is then removed. `temperature` is either zero, denoting the ground state, or lies in [.08,.9]. Repeated shots use independent equilibrium preparations. The zero-momentum condensate contributes no momentum to the measurement.

A detector measures all particles' momenta in each shot and adds their x components to obtain the total P_x = sum_(p != 0) p_x*a_p^dagger*a_p. The readout is the ensemble variance of this total, divided by V and multiplied by a common positive detector factor `variance_gain` in [.7,1.3]. A translation-invariant release or interaction switch used during measurement imparts no external impulse and conserves each shot's total momentum. The reported prediction is the thermodynamic limit of this variance per volume. There is no finite momentum acceptance, omitted population, background detector variance or additional normalization. The detector factor is the sole unknown common parameter.

Each experiment contains `temperature` and `interaction`. `data/calibration.json` contains measurements at interaction=0 across several temperatures. Records have `input`, measured `value` and independent Gaussian uncertainty `sigma`. The fixed instrument uncertainty is .00001 in the stated variance-density units, independent of the detector factor and the response. All preparation controls are exact.

Implement `Model.fit(records)`, returning self and storing `variance_gain`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Use the same fitted detector factor in every preparation.
