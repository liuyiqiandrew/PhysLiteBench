# Spin-dependent force response

A single spin-1 molecule is attached to a displacement-controlled transducer. The displacement `x` is dimensionless, measured in a fixed length unit. Energies and temperatures use one energy unit, with hbar = k_B = 1. In the ordered spin basis (+1,0,-1),

    Sz = diag(1,0,-1)
    Sx = [[0,1,0],[1,0,1],[0,1,0]] / sqrt(2)
    H(x) = (anisotropy - coupling*x)*Sz**2 + transverse*Sx + longitudinal*Sz.

Here `Sz**2` means the matrix square. This Hamiltonian specifies all displacement-dependent molecular energy near x=0. The positive unknown `coupling` is common to every experiment and lies in [0.8,1.4]. All other controls are known.

At each imposed displacement, a weak reservoir prepares the canonical equilibrium state at `temperature`. The transducer records the mean molecular force on its movable part. Each experiment reports the slope of this force versus displacement at x=0, using fully equilibrated readings at infinitesimally separated displacements. The force is conjugate to x and has the same energy unit as H. The empty transducer's force has been subtracted. There are no other molecular levels or displacement-dependent interactions in this model.

Allowed controls are `anisotropy` in [0.7,1.3], `transverse` in [0,1.5], `longitudinal` in [-0.7,0.7], and `temperature` in [0.2,0.8].

Implement `Model.fit(records)`, returning self, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. The fitted positive parameter is `model.coupling`. Each experiment is a dictionary with the four controls above. Calibration records contain `input`, measured `value`, and the known independent Gaussian standard deviation `sigma`. Output is the molecular force slope in the stated energy unit.
