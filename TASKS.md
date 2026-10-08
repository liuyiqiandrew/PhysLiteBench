# Task physics

The current publication has ten source-matched qualified tasks; see the [qualified roster](README.md) and [result manifest](results/qualified-tasks.json). Entropy anomaly r7, rotating reservoir r8 and superconducting heat r6 replace their earlier canonical revisions. Hierarchical glass overlaps r2 is the final addition. For current apparatus definitions and APIs, use each task's environment/README.md; its AUTHOR.md records derivations and validation. Earlier discussions below describe historical revisions where indicated.

The hierarchical-glass task compares a quenched two-level Gaussian tree with an independent-coordinate approximation that has the same warm caloric law. It measures common-parent, common-state and exact labelled leaf partitions for two through six replicas. Selection of a parent reweights its descendants by a fractional power of the subtree partition sum; ignoring this correlation caused all three reviewed failures.

All active tasks use a neutral instruction that permits replacing all implementation code. Current outcomes are in [DASHBOARD.md](DASHBOARD.md); archived scores do not describe revised sources. The descriptions below are author material and are excluded from agent images.

| Active task | Physical issue | Public apparatus | Derivation and validation |
|---|---|---|---|
| finite-layer-phoresis | Navier slip changes the transmission of solute interaction force as well as translational drag. | [README](tasks/finite-layer-phoresis/environment/README.md) | [AUTHOR](tasks/finite-layer-phoresis/AUTHOR.md) |
| entropy-anomaly | Magnetic rotation changes kinetic heat transport even when the magnetic force does no work. | [README](tasks/entropy-anomaly/environment/README.md) | [AUTHOR](tasks/entropy-anomaly/AUTHOR.md) |
| hydrodynamic-heating | Regional heating follows carrier drag dissipation while pressure transports conservative energy. | [README](tasks/hydrodynamic-heating/environment/README.md) | [AUTHOR](tasks/hydrodynamic-heating/AUTHOR.md) |
| insulating-mhd-duct | Oblique magnetic fields share one electrical current-closure problem. | [README](tasks/insulating-mhd-duct/environment/README.md) | [AUTHOR](tasks/insulating-mhd-duct/AUTHOR.md) |
| magnetic-tracer | Mechanical signed area retains a contribution lost by the limiting position SDE. | [README](tasks/magnetic-tracer/environment/README.md) | [AUTHOR](tasks/magnetic-tracer/AUTHOR.md) |
| reaction-diffusion | The ionic conductivity in the coupled transport operator depends on local composition. | [README](tasks/reaction-diffusion/environment/README.md) | [AUTHOR](tasks/reaction-diffusion/AUTHOR.md) |
| rotating-reservoir | Thermal heat in a moving reservoir differs from laboratory-frame mechanical energy exchange. | [README](tasks/rotating-reservoir/environment/README.md) | [AUTHOR](tasks/rotating-reservoir/AUTHOR.md) |
| single-file-memory | A fixed particle preparation retains two-time memory that stationary increments miss. | [README](tasks/single-file-memory/environment/README.md) | [AUTHOR](tasks/single-file-memory/AUTHOR.md) |
| superconducting-heat | The electron–hole heat coherence factor varies with quasiparticle energy. | [README](tasks/superconducting-heat/environment/README.md) | [AUTHOR](tasks/superconducting-heat/AUTHOR.md) |
| thermoelastic-rod | Mechanical equilibration requires the viscosity-weighted common-stress constraint. | [README](tasks/thermoelastic-rod/environment/README.md) | [AUTHOR](tasks/thermoelastic-rod/AUTHOR.md) |

Phoresis revision 2 has three reviewed physical failures in three completed unhinted Luna-high trials. Its [promotion audit](results/zero-three-phoresis-r2-promotion.json) also preserves one started outage interruption and two unstarted setup failures; those are not physics failures. It replaces [electric-dipole-force revision 1](archives/electric-dipole-force-r1), which passed the fresh confirmation 3/3. Source author notes describe the frozen pre-evaluation revision; current outcomes are in the dashboard.

Single-file revision 1 also has three reviewed physical failures in three completed unhinted trials. All three fit the one-time variance but retain a stationary-increment covariance inappropriate for the fixed initial particle lattice. Its [promotion audit](results/zero-three-singlefile-r1-physics-promotion-audit.json) keeps the one unstarted TLS setup failure separate. It replaces [spin-strain-response revision 1](archives/spin-strain-response-r1), which passed the fresh confirmation 3/3.

Weak candidates and their complete prior results are linked from the dashboard. Earlier task explanations, scores and hint comparisons remain in the [pre-neutral archive](archives/pre-neutral-instructions).

Each public task must define its initial and boundary conditions, preparation, observables and limit order. These determine the physical problem. The missing correction, derivation, fitted true value, hidden cases and diagnostic hint stay outside the agent image. No particular solver, discretization or oracle representation is required.

Each completed shortcut must fit calibration and identify the unknown parameter before failing an allowed prediction. Independent references, repeated-noise checks and oracle/shortcut controls establish scientific validity. Only fresh frozen agent runs establish observed difficulty. Earlier outcomes stay attached to their evaluated sources.
