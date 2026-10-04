# Pressure relaxation in a porous elastic layer

A saturated homogeneous porous elastic layer is in plane strain: all out-of-plane strains vanish. Its in-plane square representative cell has side `L = 0.01 m` and periodic fluctuations. The in-plane displacement is an affine part plus a periodic part with zero mean. The affine part adjusts freely so that all three mean in-plane total stress components vanish. The solid is in quasistatic mechanical equilibrium at every time; inertia and body forces are negligible.

For small deformations, write the engineering-strain vector as

    e = (du_x/dx, du_y/dy, du_x/dy + du_y/dx).

The conjugate stress components are `(sigma_xx, sigma_yy, sigma_xy)`. The stored energy per reference volume is

    f(e, zeta) = 0.5 e^T C e + 0.5 M (zeta - alpha*(e_xx + e_yy))^2,
    C = 10^6 [[6, 2, 0.8], [2, 4, 0.6], [0.8, 0.6, 3]] Pa,
    M = 20*10^6 Pa,   alpha = 0.9.

Here `zeta` is the excess fluid volume per reference volume. Pore pressure is conjugate to this fluid content. All pressures and fluid contents are perturbations about a common positive-pressure reference state; the prescribed amplitudes are small enough for this linear theory. There is no thermal or viscous stress in the skeleton.

The relative fluid volume flux in the layer obeys Darcy's law with permeability

    K = 10^-15 [[1, 0.2], [0.2, 0.6]] m^2.

At every point, a separate local hydraulic connection drains to the same large reservoir at zero perturbation pressure. Its outflow per unit layer volume is `g*p/viscosity`, where `g = 10^-10` is dimensionless and `p` is the local pore pressure in Pa. The connections add no compliance or mechanical constraint. Their conductance and the in-plane Darcy flux have the same unknown fluid dynamic viscosity, in `[0.0008, 0.0015] Pa s`. Fluid storage is exactly the fluid content in the given energy. There are no other sources after preparation.

At time zero, the pressure is

    p(x,y,0) = initial_pressure * cos(2*pi*(mode_x*x + mode_y*y)/L),

and the solid has already mechanically equilibrated to it under the stated affine and periodic conditions. Pressure preparation then ends and the reservoir connections are opened. The measured value is the pressure coefficient of this cosine at `time`: its spatial mean for the uniform mode, and twice its spatial cosine projection for a nonzero mode. Output is in Pa.

Each experiment gives `mode`, a two-element integer array with each entry in `[-2, 2]`; `initial_pressure` in `[1000, 3000]` Pa; and `time` in `[0, 4]` s. All material constants and geometric quantities above are known. The same viscosity applies to every preparation.

`data/calibration.json` contains independent readings with keys `input`, `value`, and known fixed instrument uncertainty `sigma = 1 Pa`. Implement `Model.fit(records)` and `Model.predict(experiments)`. `fit` returns `self` and stores the inferred `self.viscosity`. `predict` returns a finite NumPy array of shape `(len(experiments),)` in Pa. You may use any numerical method consistent with the apparatus.
