# Spectral memory in an aging glass

This is an annealed activated-trap experiment. Exponentially distributed depths and activated escape below the mean depth give a power-law trapping time and an aging population. The same occupied depth responds to the new temperature even if the change is abrupt. [Monthus and Bouchaud, *Models of traps and glass phenomenology*](https://arxiv.org/abs/cond-mat/9601012)

Let M=w^(T₀/T_g), define v=M exp(−E/T_g), and write a=T/T_g for the current temperature. A fresh basin is uniform in v from 0 to M. At the time scale used in that temperature pulse the escape rate is r=v^(1/a), exactly. On each jump the new basin is independent of the old one. A quiet exposure of duration d multiplies the occupied-depth density by exp(−d r). The numerator also requires no escape from the chosen reading until the end. Both probabilities include all mandatory quiet exposures on the same sample.

## Renewal calculation

As M tends to infinity, the rate of visits to fresh depths per unit v after a fresh start has density C t^(a−1), with C=sin(πa)/(πΓ(1+a)). To see this, the Laplace transform of one residence time has deficit M⁻¹∫₀ᴹ s/(s+v^(1/a))dv. Multiplying the renewal sum by M⁻¹ and taking the limit gives sin(πa)/(πa s^a). After the last entry into a basin of depth v the probability to stay there for time d is exp(−d r). Its occupation density at time t is therefore C∫₀ᵗ u^(a−1)exp[−(t−u)r]du. This integrates to one over v. The oracle evaluates the positive convolution using the confluent hypergeometric function.

For an initial occupied population m(v), either it survives a new pulse of duration t at the same depth, or the first escape occurs at some s<t. The latter has density ∫m(v)r exp(−sr)dv. Convolving this flux with the fresh occupation law for the remaining time gives the unconditional population at t. The free and exposed parts of each pulse use this evolution and the survival factor in their physical time order. The starter replaces the distributed depths by a single Arrhenius age.

## Independent target calculation

The private reference uses the complete-graph master equation in the Laplace variable s. For an initial population m, the directly surviving term is m(v)/(s+r). The recycled term is proportional to 1/(s+r), with coefficient

    [sin(πa)/(πa s^a)] ∫ m(v)r/(s+r) dv.

The integral is replaced by one for a fresh preparation. The coefficient follows by solving the finite-M master equation for its total incoming flux before taking M to infinity. The reference inverts this resolvent on a complex contour and evolves each successive depth distribution. It uses a separate depth grid and never evaluates a hypergeometric density. The generator script fixes the preparations with seed 47282.

For the isothermal single pulse, let R denote the time to the first escape after age A. Its density in scaled units is sin(πa)A^a/[π R^a(A+R)]. On a fresh process, no renewal between s and t has probability I_(s/t)(a,1−a), the regularized incomplete beta integral. Decomposing on the first escape produces a separate one-dimensional check of both the retained probability and the reported fraction for all three reading positions. The worked half-depth example gives retained mass 0.51687741 and joint memory mass 0.46042724.

The completed control keeps the full thermal dynamics and all subsequent optical selections. At the end of the initial write exposure it uses the unconditional population of the correct elapsed age, treating that first selection as independent thinning. This erases the bias toward deeper basins induced by the quiet exposure. The control fits the ten calibrations and its errors exceed the 0.004 allowance in all three prediction families. Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_aging_line_memory.py` to check the two calculations, controls and isothermal limits. Add `--generate` to regenerate the data.
