# Driven optical binding: concept review

Verdict: reject the proposed backup as written. The apparatus is physically valid, but the proposed completed shortcut does not yet specify a consistent conservative approximation. I would not develop it merely by choosing a pair-energy expression and differentiating it. No implementation or model evaluation was performed.

## Sound physical apparatus

Lossless electric point dipoles with a specified outgoing Green tensor and radiation-dressed polarizability define a consistent open scattering problem. In vacuum units, with the `exp(-i*omega*t)` convention, write

```
alpha = 1 / (alpha0^-1 - i*k^3/(6*pi)),
p_i = alpha [E_inc(r_i) + sum_{j != i} G(r_i-r_j) p_j].
```

Here `G` includes the full electric dipole field normalization and all near-, intermediate-, and far-field terms. The force follows from the local Lorentz force,

```
F_i,a = Re { p_i* dot [partial_a E_inc(r_i)
                      + sum_{j != i} partial_a G(r_i-r_j) p_j] } / 2.
```

The field derivative acts at the observation point with source moments held fixed; the coupled moments are solved anew for each apparatus configuration. This is not a total configuration derivative of an eliminated mechanical potential. The self-field is excluded from the exciting field, with radiation reaction included once through `alpha`. A pure electric point dipole has no net self-radiation recoil force; interference with radiation from other dipoles is present through the full exciting field and momentum balance.

Single-particle plane-wave pressure is nonzero because of radiative damping and can identify a positive static response scale on a suitable bounded interval. Multiple fixed dipoles under sustained illumination can exchange momentum with outgoing radiation, so their mechanical interactions need not obey pairwise action and reaction or derive from a shared potential. Separate support-force measurements would expose that physics. Independent Maxwell-stress integration around each particle would be an appropriate reference.

## Why the proposed shortcut is not ready

“Solve coupled dipoles exactly, then take the conservative pair-energy gradient and add isolated incident pressure” leaves unresolved what energy is meant and which variables are held fixed. This is a physical ambiguity, not merely a missing derivative implementation.

For example, stack the dipoles into `p`, and let `G_R` be the real, reciprocal off-diagonal coupling matrix. A conventional reactive quadratic functional is

```
V(p,r) = p† [alpha0^-1 I - G_R(r)] p / 4
         - Re[p† E_inc(r)] / 2.
```

Stationarity with respect to `p*` produces the reactive response
`[alpha0^-1 I-G_R]p=E_inc`. The proposed exact response instead includes
`-i[k^3/(6*pi) I+Im G]p`. Its moments generally do not make `V` stationary. Substituting those moments into `V` and differentiating therefore adds response-derivative terms with no established interpretation as the proposed conservative force. Holding the moments fixed instead avoids those terms but does not produce the configuration gradient of `V(p(r),r)`. Retaining just a mutual pair term further omits the polarization self-energy and drive contribution needed even in the reactive variational problem.

One can define an arbitrary scalar of the exact moments and call its gradient a phenomenological force. That is mathematically complete, but it does not by itself establish the defensible physical approximation requested here. Likewise, adding isolated incident pressure does not account for the interaction-induced changes in incident-field extinction when the moments are dressed by the other particles.

A controlled alternative exists at a consistently truncated weak-coupling order: retain conservative optical binding and single-particle scattering while deleting collective scattering terms. [Rudolph et al., *Quantum theory of non-Hermitian optical binding between nanoparticles*](https://www.klaus-hornberger.de/pdf/pra30.pdf), especially equations 72–76, separates a reactive interaction potential from collective radiative contributions. That supports the phenomenon and a possible effective-model comparison. It does not justify replacing their perturbative induced moments by an arbitrary exact multiple-scattering solution inside the potential. Developing that alternative would be a new proposal, not completion of the current control.

## Archive overlap and recommendation

I read the author descriptions and public specifications for archived `electric-dipole-force`, `optical-torque`, `dipole-recoil`, and `induced-dipole`. The proposed collective configuration forces differ from the first task's electric canonical-momentum versus Poynting-pressure distinction, and from the static induced-polarization energy factor. However, the recoil and torque tasks already retain radiation-dressed responses while omitting momentum or angular momentum carried by coherent outgoing radiation. Optical binding adds spatial coupling and nonreciprocity, but the intended omission remains close to those existing radiative-force tasks.

This is not an exact apparatus duplicate. The novelty is nevertheless insufficient to compensate for the current control ambiguity. Drop this version. Reconsider only if a short, explicitly derived effective model supplies the conservative control and a diagnostic tied specifically to nonreciprocal configuration forces; do not treat differentiation of a guessed interaction energy as an approved physical baseline.
