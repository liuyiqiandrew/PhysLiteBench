# Photon counting after polarization rotation

An ideal optical source prepares N=2 or N=3 photons, with exactly one photon in each input port of a lossless interferometer. Every photon initially has horizontal polarization. Their normalized temporal and spectral modes are identical, their arrival times are matched, and no degree of freedom other than polarization distinguishes them. There is no loss, additional photon emission, detector noise, or interaction between photons. Relative delays introduced by the preparation optics are compensated exactly.

Before entering the interferometer, input photon j passes through an ideal optical rotator of known length `lengths[j]` in centimeters. Its Jones vector in the fixed horizontal/vertical basis becomes

    z_j = [cos(rotation_gain * lengths[j]), sin(rotation_gain * lengths[j])].

The unknown common `rotation_gain` is between 0.6 and 1.4 radians/cm. Each length is between 0 and 3.5 cm. The rotator adds no unmodeled distinguishability. The specified Jones states describe the complete polarization preparation, with no random polarization or phase averaging.

The interferometer acts identically on both polarizations. Its field-amplitude transmission matrix, from input j to output k, is the balanced discrete Fourier transform

    U[k,j] = exp(2*pi*i*k*j/N) / sqrt(N),   k,j = 0,...,N-1.

Thus a creation operator in input j with polarization p transforms to the sum over k of U[k,j] times the output-k creation operator with the same p. The input is a product with one photon in each spatial input; photon statistics are bosonic. No additional path phases are present. The ports are indexed from zero.

Ideal photon-number-resolving detectors count each output port without resolving polarization, arrival time, or frequency. The input `counts` is a length-N list of nonnegative integers summing to N. Return the probability of observing exactly that occupation list. Every experiment contains `lengths` and `counts`; both have length N, where N is 2 or 3. A run is prepared again independently for every measurement.

Calibration uses two photons, one unrotated and the other with known rotator length at most 1 cm, and records one-photon-per-output coincidence probabilities. These data identify `rotation_gain`. Errors on those calibration probabilities are independent Gaussian errors with the stated standard deviations.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Expose `Model.rotation_gain`. Calibration records in `data/calibration.json` have keys `input`, `value`, and `sigma`. The starter sums many-photon paths using pairwise polarization distinguishability; check how its use of the Jones states follows from the apparatus.
