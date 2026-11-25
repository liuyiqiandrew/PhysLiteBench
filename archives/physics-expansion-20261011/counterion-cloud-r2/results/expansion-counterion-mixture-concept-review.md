# Counterion mixture: proposed revision concept review

**Approved as a substantive revision concept, subject to full numerical validation before evaluation.** Keep the canonical PB apparatus and dilution order from counterion-cloud r1. Add monovalent and trivalent negative point ions with separately conserved numbers. The known fraction `f` is a fraction of neutralizing charge, not a particle fraction: with `lambda=u*tau0`, `N1=(1-f)*lambda` and `N3=f*lambda/3` per unit axial length. Normalize pressure by the total ion number density, using `Ntotal=lambda*g`, `g=1-2*f/3`. Define both absent-species endpoints explicitly.

The physical species laws are `n1 proportional to exp(psi)` and `n3 proportional to exp(3*psi)`, coupled to one potential and the two fixed population constraints. This is the standard multicomponent PB construction in [Deserno and Holm, Section 5](https://arxiv.org/pdf/cond-mat/0112096). Correlations and specific binding remain excluded even for trivalent ions.

Let `xi=lB*lambda`. In dimensionless free-ion counts `v1=lB*N1_free`, `v3=lB*N3_free`, the leading logarithmic PB free-energy coefficient is

```text
E = (v1+3*v3)^2 - 2*(v1+v3),
0 <= v1 <= (1-f)*xi,  0 <= v3 <= f*xi/3.
```

Electrostatic energy penalizes residual charge squared, while radial entropy rewards each free particle. The minimum selects trivalent condensation first. For general valence `Z>1`, the three thresholds are `1/Z`, `1/[Z*(1-f)]`, and `1/(1-f)`. Writing `xeff=v1+Z*vZ` and `vfree=v1+vZ`, the four regimes are

| Regime | `xeff` | `vfree` |
|---|---|---|
| No condensation | `xi` | `xi*(1-f+f/Z)` |
| Partial multivalent condensation | `1/Z` | `xi*(1-f)*(1-1/Z)+1/Z²` |
| All multivalents condensed, monovalents free | `(1-f)*xi` | `(1-f)*xi` |
| All multivalents condensed, partial monovalent condensation | `1` | `1` |

There is no regime with both valences partially condensed: their marginal conditions would require incompatible residual charges. The endpoint cases `f=0,1` reduce to the appropriate pure species, and the intermediate intervals can disappear there.

The pressure follows independently from the PB virial identity applied to the condensed layer and the full neutral cell:

```text
phi = [vfree-xeff²/2]/[xi*(1-f+f/Z)] = -min(E)/[2*xi*(1-f+f/Z)].
```

This includes the electrostatic contribution of the free cloud; replacing its pressure by its ideal free-ion count alone would be a different error. The four expressions are continuous with continuous first derivatives at their boundaries.

The proposed control replaces the mixture by ions of mean valence `zbar=1/g` and applies the full single-species Manning law at `zbar*xi`. This is a complete mean-valence PB closure preserving total ion number and charge. It is exact for pure species and for all weak preparations with `Z*xi<=1`, and it already repairs r1's basic condensation omission. At `u<=.15`, `Z=3`, `tau0<=1.4` and `lB<=1.25`, `Z*xi<=.7875`; mixed calibration is exactly shared and identifies `tau0` from `1-xi/(2*g)`.

There are substantial allowed gaps. For `(xi,f,Z)=(.7,.8,3)`, `(1.5,.6,3)` and `(1.6,.2,3)`, exact pressures are .4557823, .4666667 and .3605769, with mean-valence relative errors 26.87%, 57.14% and 24.89%. These respectively probe partial trivalent condensation, exhausted trivalents, and partial monovalent condensation. Keep frozen hidden inputs away from all three crossover surfaces and retain pure-species/shared anchors.

I independently minimized the bounded quadratic free energy at five settings; it agrees with the four-regime formula within `4.45e-16`. I also ran the prototype's separate species-conserving PB solver at log radii 80 and 160 for three nontrivial regimes. Partial-condensation errors decreased from `3.14e-4` to `8.47e-5`, and from `4.98e-4` to `1.31e-4`; the exhausted-trivalent case agreed within `3.84e-11`. These support the limit, but do not replace a packaged convergence and domain check.

Describe convex minimization as an independent implementation of the limiting free-energy law. The finite-cell PB BVP supplies the independent physical check. Validate separate species counts, total neutrality, both pure endpoints, all crossovers, and extrapolation/refinement before freezing truth; convergence becomes nonuniform near crossover or absent-species limits. Prototype log-density variables need explicit pure-species branches at `f=0,1`.

This changes the omitted physics from whether condensation occurs to which conserved species condenses. It is a justified r2 rather than an unchanged resample. Existing electroviscous PB tasks test conduction, while binary gravity tests attractive mass segregation and stability; neither implements this valence-selective dilution law. Related primary literature is Ramanathan, *Polyelectrolyte solutions with ionic mixtures*, J. Chem. Phys. 85, 2957–2960 (1986), DOI [10.1063/1.451816](https://doi.org/10.1063/1.451816). Its full text was unavailable during this review; the formulas above were checked by derivation and PB numerics, not attributed to an unread formula in that paper.
