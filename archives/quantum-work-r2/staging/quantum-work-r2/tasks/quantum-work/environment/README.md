# Energy records from a driven three-level system

Use hbar=1. In a fixed orthonormal basis the Hamiltonian with the controls off is H0=energy_scale*D, where

    D = [[0,0,0], [0,1,0], [0,0,2.35]],
    X = [[0,1,.25], [1,0,.8], [.25,.8,0]],
    Y = [[0,-i,.45i], [i,0,-.7i], [-.45i,.7i,0]].

The common unknown energy_scale lies in [.8,1.2]. Each repetition independently prepares one of the three H0 energy eigenstates, chosen with equal classical probability 1/3. An ideal projective energy analyzer records its energy. The system is then isolated and driven in this order: the Hamiltonian switches instantaneously to H0+amplitude_a*X for time_a; it switches to H0+amplitude_b*(cos(phase)*X+sin(phase)*Y) for time_b; then the controls switch off, restoring H0. Immediately afterward an ideal projective energy analyzer records the final energy. There are no intermediate measurements, thermal contacts or other dynamics. Switching does not itself apply any additional unitary impulse.

For each repetition the electronics records final energy minus initial energy. Across independent repetitions it reports the requested cumulant of these real-valued records: cumulant=1 is their mean, cumulant=2 their variance, cumulant=3 their third central moment, and cumulant=4 their fourth central moment minus three times the square of their variance. These are population cumulants; statistical measurement errors appear only in the stated calibration uncertainties. No postselection is performed.

Each experiment supplies amplitude_a and amplitude_b in [-1.1,1.1], phase in [-pi,pi], time_a and time_b in [0,1.3], and cumulant in {1,2,3,4}. The matrix coefficients and all controls are exact. Calibration records in data/calibration.json contain input, measured value and independent Gaussian standard uncertainty sigma=.0003 in the stated units. They report variances and third cumulants for varied finite pulse sequences. The instrument uncertainty is fixed independently of the noiseless response.

Implement Model.fit(records), returning self and storing the fitted energy_scale. Model.predict(experiments) must return a finite NumPy array of shape (len(experiments),), in input order, containing the requested cumulants. Any implementation satisfying this interface may be used.
