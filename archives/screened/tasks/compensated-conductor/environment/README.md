# Two-carrier conductor

A long, wide semiconductor bar contains two isotropic classical carrier
populations with charges +e and -e, where e=1.602176634e-19 C. Their known bulk
number densities are n_positive and n_negative. An immobile dopant background
balances the bulk charge. The two populations have the same effective mass and
the same field-independent momentum relaxation time, so their positive
mobilities have one common unknown value mu. In the Drude approximation,
mu=e*tau/m. The mobility is unchanged across the allowed density preparations.

A voltage source maintains the specified electric field Ex along the bar's x
axis. A uniform magnetic field B points along z; y is transverse to the bar.
The lateral faces are electrically insulating. Measure steady longitudinal
current density jx locally in the uniform central bulk, far from the contacts
and lateral edges. The observable is electrical current summed over both carrier
populations, with positive current along +x.

Electron-hole pair generation and recombination are allowed and conserve total
electric charge. They maintain the prescribed bulk densities; doping fixes
their difference. Momentum relaxation is much faster than pair equilibration,
and measurements are taken after both equilibration and charge redistribution.
The bar is much wider than its recombination and diffusion edge layers. Work
in the bulk limit: neglect those layers' contribution to the measured current
and any bulk concentration gradients. Carrier numbers are not separately
conserved at the edges, although no electrical current leaves through the
lateral faces. Pair processes do not transfer charge to an external reservoir
or add a bulk momentum-drag term. Neglect heating, quantum effects and
intercarrier momentum exchange.

## Interface

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`mobility`, in m^2/(V s), between 0.08 and 1.4.
`predict(experiments)` must return a finite NumPy array with shape
`(len(experiments),)` of signed jx values in A/m^2. You may edit the entire
implementation within this API.

Each experiment contains:

- `electric_field`: Ex in V/m, between -3 and 3.
- `magnetic_field`: B in tesla, between -6 and 6.
- `density_positive` and `density_negative`: number densities in m^-3, each
  between 0 and 4e21, with sum at least 1e20. A zero density is the single-carrier
  limit of the bulk model.

All settings are exact. `data/calibration.json` is a list of records, each with
an `input` experiment, measured current density `value`, and known Gaussian
measurement standard deviation `sigma` in A/m^2. Errors are independent. Use
the same fitted mobility for all predictions over the stated input ranges.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
