# Two driven spin probes

Two otherwise uncoupled spin-1/2 probes are exposed to one spatially uniform fluctuating longitudinal magnetic field. They start in the product state |+z,+z>. There is no direct spin-spin interaction, dissipation, or additional noise. The field's detuning xi is a zero-mean Gaussian variable of unknown standard deviation `noise_width`, between 0.3 and 1.6 radians/second. Its correlation time is much longer than each complete preparation/evolution/readout sequence, so its value is constant during each experimental shot. Between shots the values are independent samples of that Gaussian distribution. Preparation and drive settings do not affect the noise distribution. This quasistatic description is the exact noise model for this apparatus.

During a drive segment, probe i has Hamiltonian

    H_i / hbar = [Omega_x,i sigma_x + Omega_y,i sigma_y
                   + (Delta_i + xi) sigma_z] / 2.

`fields[i] = [Omega_x,i, Omega_y,i, Delta_i]` supplies the known angular frequencies in radians/second. Each component is between -3 and 3. A sequence has one or two consecutive segments, each of duration 0 to 2 seconds. Changes between segments are instantaneous. The magnetic field keeps its value through the sequence. Controls can differ between probes.

Preparation and readout each apply an ideal instantaneous local rotation to each probe. A rotation is `[azimuth, angle]` in radians and implements U=exp[-i angle*(cos(azimuth)*sigma_x + sin(azimuth)*sigma_y)/2]. Rotations are noiseless. Azimuths and angles are between -pi and pi. After the preparation rotations, execute the drive segments in their listed order, then the readout rotations. Measure both probes along +z. Return the ensemble probability that both outcomes are +, averaged over experimental shots. The measurement is a joint event, with no conditioning on an individual probe's outcome.

An input has keys `preparation` and `readout` (each a two-element list of rotation specifications), and `segments` (a list of dictionaries with `duration` and `fields`). For example, `fields` is a two-element list, one three-vector per probe. The same coordinate axes and Pauli-matrix convention are used for both probes.

Calibration uses no applied drive or static detuning. Probe 0 is prepared by a pi/2 rotation about +y and read by a -pi/2 rotation about +y; probe 1 has zero preparation and readout angles. Waiting times vary. The measured joint probability identifies the noise width. Calibration errors are independent Gaussian measurement errors with the supplied standard deviations. Other allowed experiments can drive and rotate both probes.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Expose the fitted scalar as `Model.noise_width`. Calibration records in `data/calibration.json` have keys `input`, `value`, and `sigma`. The starter computes driven single-probe evolution; check its physical assumptions when constructing the requested observable.
