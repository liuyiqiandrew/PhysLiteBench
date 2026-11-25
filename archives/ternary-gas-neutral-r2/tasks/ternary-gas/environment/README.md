# Gas diffusion records

A straight sealed cell of length 1 mm contains three ideal gases A, B, and C
of equal molecular mass. A bath maintains constant temperature. Pressure
equilibration is fast on the recorded time scale, and total molar density is
uniform and constant. Both ends are impermeable. There is no externally driven
flow, reaction, gravity, or thermal diffusion. Wall collisions do not limit
diffusion. Momentum exchange is through pairwise intermolecular collisions.

The binary diffusion coefficients at the operating temperature and pressure
are D_AB=0.2 D, D_AC=D, and D_BC=5 D, independent of composition. The common
positive scale D is unknown, between 0.005 and 0.1 mm²/s. Molecular inertia is
negligible on the recorded time scale. Each preparation specifies all three
initial mole-fraction profiles.

`data/calibration.npz` contains `x` (mm), `t` (s), `initial` (R×X×3),
`mole_fraction` (R×T×X×3), and `sigma` (same shape as `mole_fraction`).
The species order is A, B, C. Measurement errors are independent Gaussian;
initial profiles and settings are exact. Initial fractions sum to one and are
linear between mesh points. Binary preparations may have one absent species.

Complete `Model` in `model.py`. `fit(data)` receives that dictionary, returns
self, and stores D in `diffusivity`. `predict(t, x, initial)` returns a finite
NumPy array of shape `(len(t), len(x), 3)` of mole fractions. The equally spaced
mesh includes both ends of the cell. Times are nonnegative, strictly increasing,
measured from preparation, and need not include zero. Predictions may involve
any mixture of the three gases; use one fitted D throughout.

Run `python -m pytest -q test_public.py`.
