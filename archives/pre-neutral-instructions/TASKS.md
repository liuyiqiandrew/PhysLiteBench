# Task physics

The twelve retained tasks below have current-source unhinted scores of at most 1/3; ten score 0/3 with reviewed physical-model failures. [DASHBOARD.md](DASHBOARD.md) distinguishes physical failures from other causes. Each public apparatus description states the assumptions that determine the answer; author notes and private verifiers are not shown to the solving agent.

## Thermal bodies

A free internal piston imposes equal chamber pressures, not constant pressure over time. The starter uses a constant-pressure heat capacity for each chamber. Calibration heats one side while cooling the other so that the temperature sum stays fixed; pressure remains constant on that restricted preparation. General heating changes pressure and couples both first-law balances through piston work. All three unhinted agents fit heat-transfer parameters but retain the wrong constraint.

[Public apparatus](tasks/thermal-bodies/environment/README.md). Revision 9, 0/3 unhinted and 3/3 hinted. The detailed original derivation is preserved in [the earlier task guide](archives/pre-curation-docs/TASKS.md).

## Reaction diffusion

Two salts share one anion. Local electroneutrality and zero net charge flux determine one electric field, which acts on every species. The supplied independent-salt model is exact for pure-salt calibration but cannot represent coupled motion in a mixture. An initially uniform species can redistribute when the other salt creates an electric field. Correctly fitting the binary diffusivity does not determine the mixed-salt transport law.

[Public apparatus](tasks/reaction-diffusion/environment/README.md). Revision 6, 0/3 unhinted and 2/3 hinted. The hinted failure selects the correct physics but has a one-coefficient algebra error, documented separately in the dashboard.

## Ternary gas

A mixture-averaged Fick model can conserve every species and total density while violating pairwise molecular momentum balance. The supplied model subtracts the total flux correctly and reduces to exact binary diffusion. Every calibration uses a binary pair; hidden preparations contain all three components. The correct Maxwell–Stefan fluxes solve a coupled resistance system with zero total molar flux. Replacing only this physical closure repairs all three unhinted submissions at their unchanged fitted diffusivity.

[Public apparatus](tasks/ternary-gas/environment/README.md), [derivation and independent validation](tasks/ternary-gas/AUTHOR.md). Revision 1, 0/3 unhinted and 3/3 hinted.

## Insulating MHD duct

An imposed magnetic field induces current in pressure-driven conducting fluid. Insulating walls require charge-conserving currents that close through the finite duct cross section. The supplied local magnetic drag sets the interior electric potential to a constant. It solves a stable velocity problem but omits the induced field's redistribution of current and momentum. Zero-field startup calibration identifies viscosity without probing this constraint. Hidden magnetic startup experiments require solving charge balance and flow together.

[Public apparatus](tasks/insulating-mhd-duct/environment/README.md), [derivation and independent validation](tasks/insulating-mhd-duct/AUTHOR.md). Revision 1, 0/3 unhinted and 3/3 hinted.

## Thermoelectric rod

The starter predicts Seebeck voltage correctly and solves Fourier conduction plus Joule heating. Temperature-dependent thermopower also transports reversible heat. With the stated conventions, energy conservation gives

    c T_t = k T_xx + rho J^2 - T S'(T) J T_x.

The last term is Thomson heat. Zero-current calibration cannot reveal it; correct terminal voltage alone cannot validate it either. Hidden nonzero-current temperature profiles expose the missing term. Fixed reservoir temperatures do not remove bulk Thomson heat.

[Public apparatus](tasks/thermoelectric-rod/environment/README.md), [derivation and independent validation](tasks/thermoelectric-rod/AUTHOR.md). Revision 1, 0/3 unhinted and 3/3 hinted.

## Thermoelastic rod

A rod has spatially varying thermal expansion and fixed total length. Mechanical equilibrium gives uniform reaction stress, not zero strain at every point or zero stress everywhere. Eliminating that stress gives a weighted nonlocal heat capacity:

    c_sigma(x) theta_t - T0 E alpha(x) mean(alpha theta_t) = k theta_xx.

The starter solves the full spatially varying local capacity problem but omits the second term. Reflection-symmetric material coefficients and antisymmetric calibration temperatures make the weighted mean vanish exactly. General heating and finite-body contact experiments reveal the clamp's global thermal coupling. Both models remain stable; the issue is the physical constraint.

[Public apparatus](tasks/thermoelastic-rod/environment/README.md), [derivation and independent validation](tasks/thermoelastic-rod/AUTHOR.md). Revision 2, 0/3 unhinted and 3/3 hinted. Revision 1 scored 2/3 and is preserved under [its archive](archives/thermoelastic-rod-r1).

## Magnetic tracer

A charged Brownian tracer moves through a magnetic field that can vary in space. In the controlled small-mass limit, its mobility is the full inverse drag–Lorentz matrix. The scalar part gives the familiar suppressed diffusivity; the antisymmetric part vanishes from density evolution only for uniform field. Every calibration has uniform field, so a conservative scalar-diffusion model fits exactly. In a field gradient, the missing antisymmetric term transports density modes. Both models conserve probability, preserve uniform equilibrium and dissipate density variations, so those checks alone do not distinguish them.

[Public apparatus](tasks/magnetic-tracer/environment/README.md), [derivation and independent validation](tasks/magnetic-tracer/AUTHOR.md). Revision 1, 0/3 unhinted and 3/3 hinted. Only density observables are used; trajectory-current ambiguities of the overdamped limit are excluded by the stated measurement.

## Entropy anomaly

An anisotropically damped Brownian particle moves through a spatial temperature gradient in a magnetic field. The supplied model already treats its magnetic position transport correctly and includes the zero-field kinetic entropy anomaly. A Lorentz force does no work, but rotates velocity fluctuations before they thermalize, changing their heat transport. The correct anomaly comes from the cubic velocity-relaxation operator with magnetic rotation. Zero-field calibration includes anisotropic drag and nonuniform temperature, so it validates the supplied kinetic coefficient without testing its magnetic dependence.

All three unhinted submissions retain the zero-field heat coefficient after fitting friction correctly. The physical correction alone repairs their hidden predictions. An independent finite-mass Kramers reference validates the small-mass limit and a separate isotropic magnetic formula checks the coefficient.

[Public apparatus](tasks/entropy-anomaly/environment/README.md), [derivation and validation](tasks/entropy-anomaly/AUTHOR.md). Revision 3, 0/3 unhinted and 3/3 hinted. Earlier revisions remain in [r1](archives/entropy-anomaly-r1) and [r2](archives/entropy-anomaly-r2).

## Capillary mixture

A binary liquid diffuses while its composition gradients exert reversible stress. The starter computes the chemical potential and diffusive transport correctly, but its fluid force contains only an isotropic pressure contribution. Incompressibility absorbs that contribution into pressure. A single composition stripe drives no bulk motion, so the calibrated mobility is correct. Mixed two-dimensional patterns create an anisotropic capillary stress whose divergence drives flow and advects composition.

Both the supplied model and the full model conserve composition and reduce the same free energy. Those global checks do not expose the omitted mechanical coupling. Replacing only the two force components with the free-energy variation repairs all three unhinted submissions without changing mobility or calibration predictions.

[Public apparatus](tasks/capillary-mixture/environment/README.md), [derivation and validation](tasks/capillary-mixture/AUTHOR.md). Revision 1, 0/3 unhinted and 3/3 hinted.

## Nonreciprocal cavity

A passive three-mode cavity couples to external ports and internal thermal baths. Its static bias breaks reciprocity. The supplied scattering matrix is correct, but the emission model applies incoming absorptivity, `I - S† S`, to outgoing thermal noise. Propagating the independent internal baths gives `I - S S†` instead. These positive matrices have the same trace, so total-power calibration identifies the loss rate while leaving this physical distinction untested. Port-resolved measurements expose it; equal-temperature equilibrium also requires the correct outgoing matrix.

All three plain agents fit the loss rate and keep the absorption-based emission closure. Replacing only that closure repairs their predictions at the same fitted rate and leaves calibration unchanged. The hinted failure states the correct physics but sums the wrong axis of a batched array.

[Public apparatus](tasks/nonreciprocal-cavity/environment/README.md), [derivation and validation](tasks/nonreciprocal-cavity/AUTHOR.md). Revision 1, 0/3 unhinted and 2/3 hinted.

## Qubit control

Two probes see the same fluctuating field. Single-probe calibration fixes the dephasing rate but cannot determine joint outcomes. The product of separately averaged probabilities misses correlations from their shared phase. The correct model averages the joint response over that same field realization.

[Public apparatus](tasks/qubit-control/environment/README.md). Revision 6, 1/3 unhinted and 3/3 hinted. One unhinted failure is cleanly physical; the other uses an erroneous mathematical diagnostic when choosing between physical models. It is not counted among the required zero-score tasks.

## Hall bar

Insulating sidewalls impose zero transverse current. The electric field adjusts to satisfy that condition. The supplied model instead imposes zero transverse electric field, which agrees with zero-field calibration but changes longitudinal magnetotransport. For this single isotropic carrier population, the Hall field cancels the transverse magnetic deflection and the longitudinal current has no classical magnetoresistance.

[Public apparatus](tasks/hall-bar/environment/README.md). Revision 1, 1/3 unhinted and 3/3 hinted, with two reviewed physical failures.

## Screened alternatives

Candidates that exceed 1/3 unhinted passes are preserved under [archives/screened](archives/screened/README.md), with their controls, validators and source manifest. The archive includes valid physics tasks that Luna solved consistently; they do not count toward this difficulty target. [The complete ledger](results/candidates.json) retains every attempted revision and outcome.
