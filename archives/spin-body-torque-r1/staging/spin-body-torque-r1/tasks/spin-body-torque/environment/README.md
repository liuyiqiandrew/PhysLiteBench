# Torque transmitted by a driven magnetic body

An ideal insulating single-domain body has a uniform classical magnetic moment mu=moment*m, where m is a unit vector. Its spin angular momentum is S=−moment*m/gamma, with known positive gamma=1. The unknown constant moment lies in [.8,1.2]. Units are chosen so that magnetic energy is −mu dot B and angular frequency has the units gamma*B.

The rigid body is held at fixed orientation by an ideal torque clamp. Its magnetic energy is isotropic apart from the applied field: there is no magnetocrystalline anisotropy or orientation-dependent demagnetizing term. The magnetization follows the deterministic Gilbert equation

    dm/dt = −gamma*m cross B + alpha*m cross (dm/dt),

with known alpha=.18. This damping is entirely a local spin–lattice interaction within the measured body. All angular momentum transferred through this interaction is delivered to its rigid mechanical support, with no other angular-momentum reservoir, spin current or transfer delay. Thermal fluctuations are neglected in this ideal model. The field is uniform, and direct electric, orbital and field-gradient forces are absent.

For each preparation the body begins with m=(0,0,1) under the static field bias*(0,0,1). At time zero a circular radio-frequency field is switched on and maintained:

    B(t) = (amplitude*cos(frequency*t), amplitude*sin(frequency*t), bias).

Measurements begin after transients have decayed to the stable periodic state reached from this preparation. All control settings below stay in its small-cone regime. The clamp measures the torque the body exerts on the support about the magnetic center. Its mechanical transfer and electronic gain are ideal and known.

An experiment requests the x, y or z component of this torque at the specified RF phase, defined by frequency*t=phase modulo2*pi. The output is a signed torque, in moment times field units. Calibration measures only the z component for varied finite drive amplitudes, frequencies and biases.

Each input is a dictionary with bias in [.8,1.4], frequency in [.4,1.4], amplitude in [.005,.04], phase in [−pi,pi], and component equal to x, y or z. All controls are exact. The experiment has no additional fitted constants.

Calibration records in data/calibration.json contain input, measured value and independent Gaussian standard uncertainty sigma=.00001 in torque units. This instrument uncertainty is fixed independently of the response and unknown moment.

Implement Model.fit(records), returning self and storing the fitted numeric attribute moment. Model.predict(experiments) must return a finite one-dimensional NumPy array with one torque per input in the same order. Any implementation satisfying this interface may be used.
