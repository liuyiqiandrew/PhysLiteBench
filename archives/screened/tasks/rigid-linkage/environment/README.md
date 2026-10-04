# Rigid thermal linkage

A planar linkage consists of two massless, perfectly rigid rods of length l=1. The first rod is pivoted at the origin and carries a point mass m1 at its free end; the second rod joins that mass to a point mass m2=1. The known mass ratio r=m1/m2 is specified by `mass_ratio`, between 0.02 and 1. The rods and point masses move in a horizontal plane, with no gravity, excluded volume, or additional interactions. Rods may cross. All joints and the fixed pivot are ideal. Angles theta1, theta2 denote the absolute orientations of the two rods relative to the fixed +x axis, modulo 2*pi.

Positions are

    r1 = l (cos(theta1), sin(theta1)),
    r2 = r1 + l (cos(theta2), sin(theta2)).

Use reduced units with l=m2=k_B*T=1. The classical kinetic energy is therefore

    Tkin = [(r+1)*theta1_dot^2 + theta2_dot^2
            + 2*cos(theta1-theta2)*theta1_dot*theta2_dot] / 2.

The applied static orienting potential is

    U = stiffness * [factors[0]*(1-cos(theta1-bias[0]))
                     + factors[1]*(1-cos(theta2-bias[1]))].

The unknown positive `stiffness` lies in [0.3, 2.0], and is shared by all experiments. Each known factor is in [0,2], and each known bias is an angle in [-pi,pi]. The temperature remains fixed. The prescribed potentials can exert torques relative to the laboratory axes on either rod.

A weak equilibrium heat reservoir establishes the classical canonical ensemble of this exactly constrained mechanical Hamiltonian, obtained from Tkin-U by the Legendre transform. The rod lengths and planar restriction are exact holonomic identities of the mechanical model. There are no radial or out-of-plane coordinates or momenta. The apparatus is not a limit of radial spring degrees of freedom. No compensating potential or feedback changes its equilibrium distribution.

For an experiment with `locked: true`, a mechanical clamp fixes theta1=0 and removes its velocity and momentum degree of freedom before the apparatus equilibrates. Only theta2 remains free. This is a separately equilibrated clamped apparatus, not a conditional sample of the two-free-angle ensemble. With `locked: false`, both joints move and the linkage is allowed to reach full canonical equilibrium before the measurement. No release transient is measured.

The requested observable is the equilibrium mean of

    cos(harmonic[0]*theta1 + harmonic[1]*theta2 + phase).

Each harmonic entry is an integer from -2 to 2, and `phase` is in [-pi,pi]. Every input contains `locked`, `mass_ratio`, `factors`, `bias`, `harmonic`, and `phase`. Results are dimensionless.

Calibration clamps the first joint, sets both biases and phase to zero, and measures cos(theta2) while varying its known potential factor. These data identify `stiffness`. Measurement errors are independent Gaussian errors with the listed standard deviations. The same fitted stiffness applies to every allowed preparation and observable.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Expose `Model.stiffness`. Records in `data/calibration.json` have keys `input`, `value`, and `sigma`. The starter supplies angular quadrature; determine the equilibrium weight required by the physical apparatus.
