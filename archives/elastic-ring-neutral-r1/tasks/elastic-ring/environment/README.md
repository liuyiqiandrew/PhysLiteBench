# Planar elastic ring

Four identical labeled classical point beads of mass 1 move in a plane. Beads 0, 1, 2, 3 form a strictly convex counterclockwise polygon. Its four sides are harmonic springs of rest length 1 and stiffness K. The center of mass is fixed; the overall orientation is free. There is no angular-momentum constraint. In units with Boltzmann constant 1, the Cartesian Hamiltonian is

    H = sum_i |p_i|^2/2 + (K/2) sum_i (|r_(i+1) - r_i| - 1)^2
        + stiffness * [1 - cos(psi - preferred)],

with cyclic bead indices. The angle psi in (0, pi) is between r_1-r_0 and r_3-r_0. Impenetrable geometric walls restrict psi to [cutoff, pi-cutoff] and prevent loss of convexity. There are no other interactions. The positive angular stiffness is temperature independent and lies in [0.5, 1.5]. Length and energy units are fixed as above, and all angles are in radians.

Two measurements use the same angular element:

* `readout="torque"`: at zero temperature, external actuators prescribe a unit-side rhombus at the supplied `angle`. A transducer on the angular element measures its intrinsic generalized torque, positive when it does positive work for increasing psi. The reported torque excludes forces from the side springs and the external actuators. `preferred` specifies the angular element's preferred angle.
* `readout="sine"` or `"cosine2"`: the ring reaches canonical equilibrium at the supplied `temperature` at each finite K, and the reported value is the K-to-infinity limit of the equilibrium mean of sin(psi) or cos(2*psi), respectively. Bead masses, temperature, angular stiffness, preferred angle, and cutoff remain fixed in this limit. Both the overall orientation and the permitted internal configurations equilibrate.

Released-ring controls satisfy 0.5 <= temperature <= 2, 0.2 <= cutoff <= 0.6, and 0.5 <= preferred <= pi-0.5. Torque preparations have 0.2 <= angle <= pi-0.2 and the same preferred-angle range. Each experiment is prepared independently.

`data/calibration.json` contains records with `input`, measured `value`, and known independent Gaussian standard deviation `sigma`. Implement `Model.fit(records)`, returning self and setting the inferred scalar `stiffness`. Implement `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`, in input order. Each experiment is a dictionary with `readout` and the controls listed above for that readout.
