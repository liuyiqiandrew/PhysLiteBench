# Task physics

All active tasks use a neutral instruction that permits replacing all implementation code. Current outcomes are in [DASHBOARD.md](DASHBOARD.md); archived scores do not describe revised sources. The descriptions below are author material and are excluded from agent images.

| Active task | Physical issue | Public apparatus | Derivation and validation |
|---|---|---|---|
| finite-layer-phoresis | Navier slip changes the transmission of solute interaction force as well as translational drag. | [README](tasks/finite-layer-phoresis/environment/README.md) | [AUTHOR](tasks/finite-layer-phoresis/AUTHOR.md) |
| entropy-anomaly | A finite contact-position population lag survives spatially weighted calorimetry after measured-residence background subtraction. | [README](tasks/entropy-anomaly/environment/README.md) | [AUTHOR](tasks/entropy-anomaly/AUTHOR.md) |
| hydrodynamic-heating | Regional heating follows carrier drag dissipation while pressure transports conservative energy. | [README](tasks/hydrodynamic-heating/environment/README.md) | [AUTHOR](tasks/hydrodynamic-heating/AUTHOR.md) |
| insulating-mhd-duct | Oblique magnetic fields share one electrical current-closure problem. | [README](tasks/insulating-mhd-duct/environment/README.md) | [AUTHOR](tasks/insulating-mhd-duct/AUTHOR.md) |
| magnetic-tracer | Mechanical signed area retains a contribution lost by the limiting position SDE. | [README](tasks/magnetic-tracer/environment/README.md) | [AUTHOR](tasks/magnetic-tracer/AUTHOR.md) |
| reaction-diffusion | The ionic conductivity in the coupled transport operator depends on local composition. | [README](tasks/reaction-diffusion/environment/README.md) | [AUTHOR](tasks/reaction-diffusion/AUTHOR.md) |
| rotating-reservoir | Thermal heat in a moving reservoir differs from laboratory-frame mechanical energy exchange. | [README](tasks/rotating-reservoir/environment/README.md) | [AUTHOR](tasks/rotating-reservoir/AUTHOR.md) |
| single-file-memory | A fixed particle preparation retains two-time memory that stationary increments miss. | [README](tasks/single-file-memory/environment/README.md) | [AUTHOR](tasks/single-file-memory/AUTHOR.md) |
| superconducting-heat | Thermal oscillator energy exchange conditions electron–hole coherence, with different static singlet and driven equal-spin pair interference. | [README](tasks/superconducting-heat/environment/README.md) | [AUTHOR](tasks/superconducting-heat/AUTHOR.md) |
| thermoelastic-rod | Mechanical equilibration requires the viscosity-weighted common-stress constraint. | [README](tasks/thermoelastic-rod/environment/README.md) | [AUTHOR](tasks/thermoelastic-rod/AUTHOR.md) |

Phoresis revision 2 has three reviewed physical failures in three completed unhinted Luna-high trials. Its [promotion audit](results/zero-three-phoresis-r2-promotion.json) also preserves one started outage interruption and two unstarted setup failures; those are not physics failures. It replaces [electric-dipole-force revision 1](archives/electric-dipole-force-r1), which passed the fresh confirmation 3/3. Source author notes describe the frozen pre-evaluation revision; current outcomes are in the dashboard.

Single-file revision 1 also has three reviewed physical failures in three completed unhinted trials. All three fit the one-time variance but retain a stationary-increment covariance inappropriate for the fixed initial particle lattice. Its [promotion audit](results/zero-three-singlefile-r1-physics-promotion-audit.json) keeps the one unstarted TLS setup failure separate. It replaces [spin-strain-response revision 1](archives/spin-strain-response-r1), which passed the fresh confirmation 3/3.

Entropy anomaly revision 7 has three completed, causally reviewed physical failures with no execution errors. Its [promotion audit](results/entropy-anomaly-r7-promotion.json) verifies the frozen sources, unchanged gates, independent finite-mass calorimeter and single-effect diagnostic repairs. It replaces the earlier entropy revision within the same family.

Superconducting heat revision 6 has three completed, independently reviewed physical failures and no execution errors. Its [promotion record](results/superconducting-heat-r6-promotion.json) preserves the previous canonical source and all development outcomes, verifies single-effect repairs at unchanged fitted conductance, and documents validation inheritance.

Weak candidates and their complete prior results are linked from the dashboard. Earlier task explanations, scores and hint comparisons remain in the [pre-neutral archive](archives/pre-neutral-instructions).

Each public task must define its initial and boundary conditions, preparation, observables and limit order. These determine the physical problem. The missing correction, derivation, fitted true value, hidden cases and diagnostic hint stay outside the agent image. No particular solver, discretization or oracle representation is required.

Each completed shortcut must fit calibration and identify the unknown parameter before failing an allowed prediction. Independent references, repeated-noise checks and oracle/shortcut controls establish scientific validity. Only fresh frozen agent runs establish observed difficulty. Earlier outcomes stay attached to their evaluated sources.
