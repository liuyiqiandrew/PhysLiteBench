# Task physics

All active tasks use a neutral instruction that permits replacing all implementation code. Current outcomes are in [DASHBOARD.md](DASHBOARD.md); archived scores do not describe revised sources. The descriptions below are author material and are excluded from agent images.

| Active task | Physical issue | Public apparatus | Derivation and validation |
|---|---|---|---|
| electric-dipole-force | The particle force differs from pressure inferred from the total local Poynting flux. | [README](tasks/electric-dipole-force/environment/README.md) | [AUTHOR](tasks/electric-dipole-force/AUTHOR.md) |
| entropy-anomaly | Magnetic rotation changes kinetic heat transport even when the magnetic force does no work. | [README](tasks/entropy-anomaly/environment/README.md) | [AUTHOR](tasks/entropy-anomaly/AUTHOR.md) |
| hydrodynamic-heating | Regional heating follows carrier drag dissipation while pressure transports conservative energy. | [README](tasks/hydrodynamic-heating/environment/README.md) | [AUTHOR](tasks/hydrodynamic-heating/AUTHOR.md) |
| insulating-mhd-duct | Oblique magnetic fields share one electrical current-closure problem. | [README](tasks/insulating-mhd-duct/environment/README.md) | [AUTHOR](tasks/insulating-mhd-duct/AUTHOR.md) |
| magnetic-tracer | Mechanical signed area retains a contribution lost by the limiting position SDE. | [README](tasks/magnetic-tracer/environment/README.md) | [AUTHOR](tasks/magnetic-tracer/AUTHOR.md) |
| reaction-diffusion | The ionic conductivity in the coupled transport operator depends on local composition. | [README](tasks/reaction-diffusion/environment/README.md) | [AUTHOR](tasks/reaction-diffusion/AUTHOR.md) |
| rotating-reservoir | Thermal heat in a moving reservoir differs from laboratory-frame mechanical energy exchange. | [README](tasks/rotating-reservoir/environment/README.md) | [AUTHOR](tasks/rotating-reservoir/AUTHOR.md) |
| spin-strain-response | Quantum equilibrium force fluctuations differ from the fully equilibrated displacement response. | [README](tasks/spin-strain-response/environment/README.md) | [AUTHOR](tasks/spin-strain-response/AUTHOR.md) |
| superconducting-heat | The electron–hole heat coherence factor varies with quasiparticle energy. | [README](tasks/superconducting-heat/environment/README.md) | [AUTHOR](tasks/superconducting-heat/AUTHOR.md) |
| thermoelastic-rod | Mechanical equilibration requires the viscosity-weighted common-stress constraint. | [README](tasks/thermoelastic-rod/environment/README.md) | [AUTHOR](tasks/thermoelastic-rod/AUTHOR.md) |

Weak candidates and their complete prior results are linked from the dashboard. Earlier task explanations, scores and hint comparisons remain in the [pre-neutral archive](archives/pre-neutral-instructions).

Each public task must define its initial and boundary conditions, preparation, observables and limit order. These determine the physical problem. The missing correction, derivation, fitted true value, hidden cases and diagnostic hint stay outside the agent image. No particular solver, discretization or oracle representation is required.

Each completed shortcut must fit calibration and identify the unknown parameter before failing an allowed prediction. Independent references, repeated-noise checks and oracle/shortcut controls establish scientific validity. Only fresh frozen agent runs establish observed difficulty. Earlier outcomes stay attached to their evaluated sources.
