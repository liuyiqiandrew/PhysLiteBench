# Thermal dumbbells in a prescribed planar flow

A dilute solution contains independent two-bead Hookean dumbbells moving in a plane. For each molecule the connector is R=r_2-r_1. A linear spring exerts force -H*R on bead 2 and the opposite force on bead 1. Set H=k_B*T=1 and the dumbbell number density to one. There is no finite-extension constraint or interaction between dumbbells.

The prescribed solvent velocity is u(r,t)=gradient(t)*r. Each bead has isotropic drag force -zeta*(v-u) and an independent Gaussian white thermal force with covariance 2*zeta*k_B*T times the identity times delta(t-t'). Neglect bead inertia and hydrodynamic interactions between beads. The unknown constant relaxation time is `relaxation_time`=zeta/(4*H), between 0.4 and 1.2. No other force or source of noise acts during an experiment. The dilute molecules do not change the solvent flow.

At time zero the connector has zero mean and a known normalized Gaussian distribution with covariance `initial_covariance`, a symmetric positive-definite 2 by 2 matrix whose eigenvalues lie between 0.3 and 2.5. It is prepared by equilibrating the connector under a temporary anisotropic quadratic spring with stiffness matrix inverse(initial_covariance), then switching to the actual isotropic spring H=1 at time zero. Both the bath and drag remain fixed. Measurements concern overdamped motion after this switch; there is no momentum impulse or finite bead-inertia transient.

`segments` contains one or two successive intervals. Each has a nonnegative `duration` and a real 2 by 2 `gradient` matrix. The total duration is at most 6. The gradient is constant within each interval, has trace zero, and is applied continuously to the same molecules; changing it does not reprepare their connectors. Write D=(gradient+gradient.T)/2 and Omega=(gradient-gradient.T)/2. The absolute eigenvalues of D are at most 0.35, and |Omega[1,0]| is at most 1.2. These bounds keep constant extensional flows below the Hookean stretching instability for every allowed relaxation time.

The polymer extra stress is the spring configurational stress minus its equilibrium isotropic value,

    S_ij = <R_i*R_j> - delta_ij.

At the end of the complete sequence, return the requested `observable`:

- `shear_stress`: S_xy.
- `normal_stress`: S_xx-S_yy.
- `trace_stress`: S_xx+S_yy.

Every input contains `initial_covariance`, `segments`, and `observable`. Each measurement is an ensemble average over independently prepared dumbbells. Calibration observes relaxation of normal stress from a known anisotropic preparation during rigid solvent rotation. It identifies the common relaxation time. Errors are independent Gaussian errors with the stated standard deviations; settings and initial covariance are exact.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Expose `Model.relaxation_time`. Records in `data/calibration.json` have keys `input`, `value`, and `sigma`. The starter transports a relaxing stress tensor with local solvent rotation and strain; check its constitutive assumptions against the microscopic apparatus.
