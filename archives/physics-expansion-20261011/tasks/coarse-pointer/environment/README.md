# Qubit and position-pointer readout

Every shot prepares a qubit in |+x>=(|+z>+|-z>)/sqrt(2) and an independent
position pointer in the pure state

    psi(q) = (2*pi*w^2)^(-1/4) exp[-q^2/(4*w^2)].

Here q and the unknown common width w are in mm; w lies in [0.6,1.0] mm.
The pointer coordinate ranges over the whole real line. Its conjugate
momentum satisfies [q,p]=i, with p in mm^-1. The qubit and pointer receive
the prescribed instantaneous unitary

    U = exp(-i*d*sigma_z*p),

where sigma_z has eigenvalues +1 and -1 on |+z> and |-z>, and d is a known
displacement in mm. There is no free evolution, extra phase operation,
feedback, or other interaction before either readout.

The apparatus then measures pointer position ideally. It keeps only the
Boolean outcome q>=h, where h is a known threshold in mm. The pointer
and the detailed position record are discarded. The qubit is measured by
a projective analyzer whose plus state is

    |a+> = cos(theta/2)|+z> + exp(i*phi)*sin(theta/2)|-z>.

Angles are in radians. Every shot uses a freshly prepared qubit and pointer.
The observable `acceptance` is the probability per prepared shot that
q>=h, independent of the analyzer. The observable `joint` is the
probability per prepared shot that q>=h AND the qubit analyzer gives plus.
It is not normalized by the number of accepted shots.

Each experiment is a dictionary with `displacement` d in [0,1.8] mm,
`threshold` h in [-2,2] mm, `theta` in [0,pi], `phi` in [-pi,pi], and
`observable` equal to `acceptance` or `joint`. The common width is the
only unknown. Calibration uses acceptance and theta=0 joint readings at
positive thresholds greater than d. Predictions must cover the full domain.

`data/calibration.json` contains independent noisy readings
`{ "input": experiment, "value": probability, "sigma": sigma }`, where
sigma is the known standard deviation of additive Gaussian readout noise.
Implement `Model.fit(records)` to estimate w, store it as `model.width`,
and return `self`. `Model.predict(experiments)` must return a finite NumPy
array of shape `(len(experiments),)` in input order. The supplied response
is a starting approximation and may be changed. Run
`pytest -q test_public.py` for the visible interface and calibration check.
