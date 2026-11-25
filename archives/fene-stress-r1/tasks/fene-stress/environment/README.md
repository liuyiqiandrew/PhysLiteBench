# Dilute connector rheometry

A dilute suspension contains independent identical two-bead molecules confined
to a plane. The beads have negligible inertia and identical scalar drag `zeta`
relative to the local solvent. Hydrodynamic interactions between beads, excluded
volume, and interactions between molecules are absent. The connector vector is
`Q = r2-r1`. Its potential is

```
U(Q) = -H*L²/2 * log(1-|Q|²/L²),   |Q| < L,
H = 1.
```

For a Hookean connector, `L` is infinite and `U(Q)=H*|Q|²/2`. A molecule neither
ruptures nor resets. The finite-length potential confines the connector to its
disk with a natural zero probability flux at the boundary.

Each bead obeys the overdamped force balance

```
0 = -zeta*(dri/dt-u(ri)) - gradient_ri U + xi_i(t),
<xi_i,alpha(t) xi_j,beta(t')> = 2*zeta*T*delta_ij*delta_alpha,beta*delta(t-t').
```

The thermostat keeps temperature energy `T=kB*temperature` fixed. The imposed
spatially homogeneous solvent flow is `u(x,y)=(rate*x,-rate*y)`. Consider the
uniform dilute bulk, away from rheometer boundaries. Each measurement is made
after the stationary connector statistics have been reached at that flow. The
same beads, with the same drag, are used for all connector preparations.

The rheometer measures the excess planar mechanical tensile-stress difference
`Sigma_xx-Sigma_yy` after subtracting the solvent contribution, and divides by
the known molecular number per unit area. Planar stress is force per unit line
length, so this normalized reading has units of energy. Positive normal stress
denotes tension. The constraining reaction perpendicular to the plane does no
in-plane work.

Use fixed reference units of length, energy, and time. `H=1` is in energy per
length squared, `T` in energy, `L` in length, `rate` in inverse time, and `zeta`
in energy times time per length squared. The only unknown is the common bead
drag `zeta in [3.2,4.8]`.

Each input dictionary contains:

- `rate` in `[-.95,.95]`;
- `temperature` in `[.8,1.2]`, giving `T`;
- `max_length`, either a number in `[sqrt(6),sqrt(12)]` giving `L`, or `null`
  for the Hookean connector. For the Hookean preparation, `abs(rate)<=.25`.

## Interface

Implement `Model.fit(records)` and `Model.predict(experiments)` in `model.py`.
`fit` receives the records from `data/calibration.json`, stores the fitted drag
in `self.drag`, and returns `self`. Each record contains `input`, measured
`value`, and independent Gaussian instrument standard deviation `sigma`.
`predict` receives a list of experiment dictionaries and returns a
one-dimensional NumPy array of normalized stress readings in the stated energy
unit, in the same order.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
