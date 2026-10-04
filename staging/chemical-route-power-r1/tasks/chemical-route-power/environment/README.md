# Stationary chemical regeneration

One enzyme has three conformations labelled 0, 1, 2. It is held in a well-mixed solution at fixed temperature. Use kB*T=1 as the energy unit and seconds as the time unit. The known conformation energies are E_i. No mechanical load or time-dependent field acts on the enzyme.

For every oriented edge i to j=(i+1) modulo 3, two elementary reactions are possible. In route a, a forward conformation change converts one molecule of fuel F_a into one molecule of product P_a; the reverse event performs the reverse conversion. The two fuel/product pairs are distinct. Their large ideal reservoirs retain the known chemical-potential differences A_a=mu(F_a)-mu(P_a). There are no other reactions or relaxation channels.

Each edge and route has a known dimensionless attempt factor nu_ia. Its forward and reverse rates in inverse seconds are

    k_forward(i,a) = rate_scale*nu_ia*exp([A_a-(E_j-E_i)]/2),
    k_reverse(i,a) = rate_scale*nu_ia*exp(-[A_a-(E_j-E_i)]/2).

These rates define the ideal chemical kinetic model. The unknown positive rate_scale is the same for every experiment. There is no diffusion limitation, binding delay, saturation or reservoir backaction. All measurements are made after the enzyme has reached its stationary regime.

An external reversible regeneration apparatus restores both fuel/product reservoirs to their initial compositions at the same fixed temperature and chemical potentials. The reading is the long-time mean work supplied by that apparatus per enzyme and per second. Positive work means energy supplied to restore the reservoirs. Preparatory work and losses in the regeneration apparatus itself are excluded.

Every experiment provides:

- `energies`: three energies, with E_0=0 and E_1,E_2 in [-.3,.3].
- `attempts`: a 3-by-2 array of nu_ia, each in [.5,1.5]; the first index is the edge's initial conformation and the second is the reaction route.
- `affinities`: the two chemical-potential differences A_a, each in [.2,2.4].

The common rate_scale lies in [.6,1.4] inverse seconds. All controls are exact. Calibration records in `data/calibration.json` contain `input`, `value` and `sigma`. The measurement errors are independent Gaussian with the fixed standard deviation sigma=.003 in the stated power units.

Implement `Model.fit(records)`, returning `self` and storing the fitted estimate as `Model.rate_scale`. Implement `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)` containing the measured work rates. Run `python -m pytest -q test_public.py` for the public check.
