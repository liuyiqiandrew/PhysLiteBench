# Frozen basins and line selection

A field biases the two fixed basin moments. Two preparation temperatures populate the same deep basins with different Gibbs weights. A field-free exposure at T_g preferentially retains the deeper basin of the final particle. Joint preparation and acceptance probabilities must be averaged over specimens before conditioning. The two-temperature disorder calculation builds on [Derrida and Mottishaw, *One step replica symmetry breaking and overlaps between two temperatures*](https://doi.org/10.1088/1751-8121/abd4ad). The line selections, magnetic moments and timed escape measurement specify this benchmark's own device.

## Depths and weights

Set v=N exp(−E/T_g). Before taking the limit, each v is uniform on [0,N]. Counts in disjoint fixed intervals consequently approach independent Poisson variables of means equal to their lengths. The moment marks stay independent, splitting this unit-intensity process equally between the two signs. At setting (T,h), write α=T/T_g. Up to a common factor the Gibbs weight of a site with moment m is w(v,m)=v^(−1/α) exp(hm/T). The normalization Z is the sum of these weights in the actual specimen. The deep-site sum is finite almost surely because α<1. Survival during the final field-free exposure multiplies the probability by exp(−d v).

The solution enumerates the exact equality partition of the three or four recorded labels. Every distinct block denotes a distinct Poisson point. It enforces the specified gates on each partition. If the two preparation settings occur n_A and n_B times, represent the inverse sample normalizations by gamma integrals in s and t. The unassigned sites contribute

    exp[−∫ dv E_m(1−exp(−s w_A−t w_B))].

Each assigned site contributes its equilibrium weight for each member of its block times the same exponential, integrated over its depth and mark. The block containing the first particle also applies the requested moment tag. The last block carries the survival factor. The Poisson factorial moment formula gives this product directly.

Let α and β be the reduced temperatures. Set u=s^α+t^β, p=t^β/u and v=u z. For a partition of L blocks the residual power of u is L−1. The unassigned factor becomes exp[−u H(p)] and the last block contributes exp(−d u z). The scale integral gives Γ(L)/(H(p)+d z)^L. This leaves smooth depth integrals and a beta-weighted integral over p. The oracle evaluates these with positive quadrature.

## Independent verification

The private calculation integrates the original s and t independently on logarithmic grids. It uses equality constraints on the requested lines, and inclusion–exclusion for each specified inequality. Its unconstrained sums can share sites. The moment of their product follows by attaching each factor of the first sum to every possible subset of the other factors on the same site, then recursing on the remaining sums. This implements the mixed derivatives of the joint Poisson transform. It never enumerates the allowed exact observation partitions or removes the common scale. Seed 95882394 sets the 24 preparations. All joint selection probabilities exceed 0.01.

A direct specimen check sampled the low-v sites as two independent lists of exponential spacings, one for each sign. It normalized both Gibbs distributions separately within each resulting specimen and summed over the labelled draws. With 16,384 specimens, two independently scrambled samples and 96 explicit basins per sign, a field-reversal geometry with a coincidence gate gave 0.70682 and 0.70747; the limiting score was 0.70710. Changing that gate to an inequality gave 0.52244 and 0.52258 against 0.52254. This check uses a continuum tail estimate only beyond the deepest explicitly sampled basins.

When both temperatures agree, fields and exposure vanish, and no moment is selected, the normalized weights have the Poisson–Dirichlet law with parameter α. Conditional on two equal draws the next one joins that same line with probability (2−α)/2. Conditional on two distinct draws it joins the specified one with probability (1−α)/2. These give independent limits for the public check. Simultaneously reversing both fields and the selected moment leaves every reading invariant.

The completed shortcut preserves both temperature-dependent Gibbs laws, their common quenched specimen, all label comparisons and the magnetic tag. It treats the final survival measurement as independent of the sampled line. Its predicted conditional fractions therefore equal the zero-exposure result. Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_aging_line_memory.py` to reproduce the checks and controls.
