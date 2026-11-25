# Driven spin in a fluctuating field

A spin-1/2 probe starts in |+z> and experiences a stationary classical longitudinal magnetic field with finite correlation time. The field detuning xi(t), in radians/second, is a zero-mean Gaussian Ornstein-Uhlenbeck process. Its correlation function is

    E[xi(t) xi(s)] = noise_width^2 * exp(-abs(t-s)/tau),  tau = 1 second.

Equivalently, dxi = -xi dt/tau + sqrt(2 noise_width^2/tau) dW, where W is a standard real Wiener process. `noise_width` is the unknown stationary standard deviation, between 0.3 and 1.6 radians/second. Each shot begins with an independent stationary realization of the field, uncorrelated with the prepared spin. The field then evolves continuously through the entire shot. Applied controls do not change the field process. There is no other noise, relaxation, or bath back-action.

The Hamiltonian during each drive segment is

    H(t) / hbar = [Omega_x sigma_x + Omega_y sigma_y
                   + (Delta + xi(t)) sigma_z] / 2.

`field = [Omega_x, Omega_y, Delta]` gives known angular frequencies, each in [-4, 4] radians/second. A sequence contains one or two consecutive constant-control segments. Each segment duration is in [0, 4] seconds, and the total duration is at most 4 seconds. Changes of controls between segments are instantaneous; the noise process is not restarted at those changes.

An ideal instantaneous preparation rotation precedes the segments, and an ideal instantaneous readout rotation follows them. Each is specified as `[azimuth, angle]`, with both entries in [-pi, pi], and implements U=exp[-i angle*(cos(azimuth)*sigma_x + sin(azimuth)*sigma_y)/2]. These control pulses are noiseless. After readout measure along +z. Return the ensemble probability of the + outcome, averaged over the classical field trajectories.

Each experiment has `preparation`, `readout`, and `segments`; each segment contains `duration` and `field`. Calibration uses a pi/2 preparation about +y, a -pi/2 readout about +y, zero applied field, and varying wait duration. These measurements identify the one unknown noise width. Measurement errors are independent Gaussian errors with the stated standard deviations. Other allowed experiments can contain nonzero drive fields.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Expose the fitted scalar as `Model.noise_width`. Calibration records in `data/calibration.json` have keys `input`, `value`, and `sigma`. The starter includes a reduced evolution calculation; determine whether its physical assumptions apply to the entire apparatus.
