# Spectral memory in an aging glass

A local optical reporter resolves which metastable basin a small region of a glass occupies. In a freshly drawn basin the positive activation depth E is exponential with unknown mean T_g in [0.9,1.1]. Depth and absolute temperature use the same energy unit, and the microscopic attempt time and Boltzmann constant are one. A fresh-spot calibration asks whether E exceeds the input `threshold` between 0.2 and 2. Its reported value is that probability, rounded to eight decimal places.

Each dynamical shot starts in a fresh basin. At temperature T its escape is a Poisson process of rate exp(−E/T). On escape, the region enters a new independently drawn basin. It never returns to an earlier basin. Each basin has its own permanent, distinguishable optical line. Changing temperature immediately changes the escape rate in the occupied basin. There is no instantaneous reset of its depth. An optical exposure is quiet if the line stays the same throughout that exposure. Reading the line leaves the dynamics undisturbed.

Use the same sample for the following cycle. With a dimensionless age parameter w, let the time unit at the initial temperature T₀ be w. At each temperature Tᵢ use the time unit w^(T₀/Tᵢ). Take w to infinity after conditioning. Temperatures are in [0.50,0.70]. The field `temperatures` lists T₀ followed by the temperature of each pulse.

1. At T₀ age the fresh sample for A, then expose it for D, both in units of w. The field `write=[A,D]` has A in [0.5,1.3] and D in [0.15,0.9]. Retain only quiet exposures.
2. In order, apply one, two or three temperature pulses. Each entry of `pulses` is [B,G]. At that pulse's temperature, wait B in [0.4,1.5], then expose for G in [0.10,0.40]. Both use the time unit at the pulse temperature. Each of these exposures must be quiet. A pulse begins even if its temperature equals that of the previous interval.
3. Return to T₀ and expose for `recovery` in [0.20,1.20], again in units of w. Retain the shot only if that exposure is quiet as well. Two distinct quiet exposures can show different lines if an escape occurred in between.

On retained shots, read the line partway through the waiting portion of one pulse and again at the end of the recovery exposure. The field `read=[j,f]` means pulse j (zero-based) at time fB from its beginning, with f in [0,1]. Report the fraction of retained shots on which these two readings show the same line. All clocks follow the prescribed laboratory schedule irrespective of the particle's escapes.

For a check on a glass with T_g=1, take T₀=T₁=0.5, write=[1,0.3], pulses=[[0.8,0.2]], recovery=0.3 and read=[0,0.5]. The joint probability of all quiet exposures is 0.51687741. The reported conditional line memory is 0.89078616. The last two exposures are adjacent at the same temperature in this example.

Implement `Model` in /app/model.py. Its `fit(records)` receives input/value dictionaries, sets `depth_scale` to the fitted T_g, and returns self. Its `predict(inputs)` returns a flat finite real NumPy array in input order, including an empty array for an empty list. The calibration tolerance is 0.00001. Each new reading has absolute tolerance 0.004. The starter represents an ensemble by a single effective age, converting it by Arrhenius scaling when the temperature changes.
