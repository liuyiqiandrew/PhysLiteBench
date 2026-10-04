# Equilibrium fluctuations of a coupled quantum probe

A quantum probe oscillator with coordinate q and momentum p is coupled to a second quantum oscillator with coordinate Q and momentum P. Set hbar=k_B=1 and both masses to one. Their canonical commutators are [q,p]=[Q,P]=i; operators on different oscillators commute. The full Hamiltonian is

    H = (p^2 + P^2)/2 + spring*q^2/2 + 2.56*Q^2/2
        + coupling*q*Q - force*q.

This is the complete Hamiltonian, with no additional potential or counterterm. The unknown probe spring coefficient `spring` is between 0.8 and 1.5. Every other coefficient is known. For each experiment `coupling` lies between -1.3 and 1.3, so the quadratic potential is positive definite throughout the allowed spring range. The constant static `force` lies between -0.5 and 0.5.

At its chosen force, coupling, and `temperature` T between 0.08 and 1.5, the entire interacting pair is prepared in the normalized canonical state exp(-H/T). This preparation is repeated independently for every measurement. There is no quench, subsequent dynamical evolution, measurement of the second oscillator, or conditioning on its state. Probe measurements ignore the second oscillator. Neither a classical approximation nor a weak-coupling approximation is imposed.

Every input specifies `force`, `coupling`, `temperature`, and one `observable`:

- `mean_position`: the probe expectation <q>.
- `position_variance`: the centered probe variance <(q-<q>)^2>.
- `momentum_variance`: the centered probe variance <(p-<p>)^2>.

Return that real expectation in these reduced units. The same spring coefficient applies to all experiments. Calibration consists only of mean-position measurements under known static forces and couplings. These measurements identify `spring`. Their errors are independent Gaussian errors with the stated standard deviations. All settings are exact.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Expose `Model.spring`. Records in `data/calibration.json` have keys `input`, `value`, and `sigma`. The starter calculates an effective probe spring from the static coupled potential; check its equilibrium fluctuation model against the stated quantum preparation.
