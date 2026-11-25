# Heat fluctuations in a circulating trap

A particle follows the two-dimensional overdamped stochastic equation

```
dx = mobility * [-stiffness*x + stiffness*drive_ratio*R*x] dt
     + sqrt(2*mobility*T) dW,
R = [[0, 1], [-1, 0]].
```

The trap energy is U(x)=stiffness*|x|²/2. The temperature T=1 and Boltzmann
constant=1 in fixed apparatus units. The two components of W are independent
standard Wiener processes. The isotropic mobility is unknown, constant
across preparations, and lies in [0.8, 1.3]. The potential and circulating
force are kept fixed throughout each observation.

Each realization starts in the stationary state for that preparation,
including its circulating probability current. The position is not fixed
or conditioned at either end of an observation. The measured heat into
the bath during [0,t] is

```
Q(t) = integral_0^t [-stiffness*x + stiffness*drive_ratio*R*x] dot circ(dx).
```

The circle specifies the Stratonovich integral. This overdamped model and
this line integral define the apparatus; there is no additional inertial
kinetic-energy readout.

For each preparation, let q_bar=lim E[Q(t)]/t be its stationary mean heat
rate. It is positive on the allowed domain. Define Z_t=Q(t)/(q_bar*t).
The apparatus reports the inverse-time rate

```
I(z) = -lim_(epsilon -> 0) lim_(t -> infinity)
        log P(abs(Z_t-z) < epsilon) / t.
```

Both limits are part of the measurement definition. The experiment uses
the preparation's own q_bar for this normalization. No finite-time or
finite-bin correction is included in the reported I.

An input dictionary contains `stiffness` in [0.5, 2.0], `drive_ratio` in
[0.1, 1.2], and `relative_heat` z in [-6, 9]. Length, force, energy, and
time are in the fixed apparatus units. All parameters except mobility are
known. `data/calibration.json` contains independent Gaussian readings of I
near z=1, with fields `input`, measured `value`, and known standard deviation
`sigma`. Predictions may use any allowed z.

Implement `Model` in `model.py`. `fit(records)` returns `self` and stores
the fitted float `self.mobility`. `predict(experiments)` accepts a list of
input dictionaries and returns a finite NumPy array of shape `(n,)` in the
same order. You may change the supplied response model. Run
`python -m pytest -q test_public.py` to check the interface and calibration.
