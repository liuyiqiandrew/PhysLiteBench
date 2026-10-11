# Task physics

The current publication has ten qualified tasks; see the [qualified roster](README.md) and [result manifest](results/qualified-tasks.json). Four of them have had [post-qualification maintenance](docs/BENCHMARK.md#post-qualification-maintenance) without new trials. Entropy anomaly r7, rotating reservoir r8 and superconducting heat r6 replace their earlier canonical revisions. Hierarchical glass overlaps r2 is the final addition. For current apparatus definitions and APIs, use each task's environment/README.md; its AUTHOR.md records derivations and validation. Earlier discussions below describe historical revisions where indicated.

The hierarchical-glass task compares a quenched two-level Gaussian tree with an independent-coordinate approximation that has the same warm caloric law. It measures common-parent, common-state and exact labelled leaf partitions for two through six replicas. Selection of a parent reweights its descendants by a fractional power of the subtree partition sum; ignoring this correlation caused all three reviewed failures.

Every task's instruction permits replacing all implementation code; all except hierarchical-glass-overlaps use the same neutral instruction. Current outcomes are in [DASHBOARD.md](DASHBOARD.md); archived scores do not describe revised sources. The descriptions below are author material and are excluded from agent images.

| Qualified task | Physical issue | Public apparatus | Derivation and validation |
|---|---|---|---|
| entropy-anomaly | A spatially weighted calorimeter detects a small lag in local contact populations that equal local contact occupancy omits. | [README](tasks/entropy-anomaly/environment/README.md) | [AUTHOR](tasks/entropy-anomaly/AUTHOR.md) |
| finite-layer-phoresis | Navier slip changes the transmission of solute interaction force as well as translational drag. | [README](tasks/finite-layer-phoresis/environment/README.md) | [AUTHOR](tasks/finite-layer-phoresis/AUTHOR.md) |
| hierarchical-glass-overlaps | Selecting a parent reweights its descendants by a fractional power of the subtree partition sum; independent coordinates miss this correlation. | [README](tasks/hierarchical-glass-overlaps/environment/README.md) | [AUTHOR](tasks/hierarchical-glass-overlaps/AUTHOR.md) |
| insulating-mhd-duct | Oblique magnetic fields share one electrical current-closure problem. | [README](tasks/insulating-mhd-duct/environment/README.md) | [AUTHOR](tasks/insulating-mhd-duct/AUTHOR.md) |
| magnetic-tracer | Mechanical signed area retains a contribution lost by the limiting position SDE. | [README](tasks/magnetic-tracer/environment/README.md) | [AUTHOR](tasks/magnetic-tracer/AUTHOR.md) |
| reaction-diffusion | The ionic conductivity in the coupled transport operator depends on local composition. | [README](tasks/reaction-diffusion/environment/README.md) | [AUTHOR](tasks/reaction-diffusion/AUTHOR.md) |
| ring-current-fluctuations | Strongly suppressed currents in a driven half-filled exclusion ring favor a nonuniform density profile, not the homogeneous one. | [README](tasks/ring-current-fluctuations/environment/README.md) | [AUTHOR](tasks/ring-current-fluctuations/AUTHOR.md) |
| rotating-reservoir | A retained gas contact develops non-Gaussian energy correlations that change bath heat noise; a Gaussian moment closure misses them while still fitting the mean calibration. | [README](tasks/rotating-reservoir/environment/README.md) | [AUTHOR](tasks/rotating-reservoir/AUTHOR.md) |
| single-file-memory | A fixed particle preparation retains two-time memory that stationary increments miss. | [README](tasks/single-file-memory/environment/README.md) | [AUTHOR](tasks/single-file-memory/AUTHOR.md) |
| superconducting-heat | Quantum charge-displacement coherence is correlated with the oscillator energy exchanged in tunneling; a factorized covariance loses this. | [README](tasks/superconducting-heat/environment/README.md) | [AUTHOR](tasks/superconducting-heat/AUTHOR.md) |

The rows describe the qualified revisions. Earlier one-line descriptions of entropy anomaly, rotating reservoir and superconducting heat are preserved in the [pre-single-file documentation archive](archives/pre-singlefile-promotion/TASKS.md).

Two folders under `tasks/` are inherited from the base branch and are not in the qualified roster:

| Inherited folder | Physical issue | Public apparatus | Derivation and validation |
|---|---|---|---|
| hydrodynamic-heating | Regional heating follows carrier drag dissipation while pressure transports conservative energy. | [README](tasks/hydrodynamic-heating/environment/README.md) | [AUTHOR](tasks/hydrodynamic-heating/AUTHOR.md) |
| thermoelastic-rod | Mechanical equilibration requires the viscosity-weighted common-stress constraint. | [README](tasks/thermoelastic-rod/environment/README.md) | [AUTHOR](tasks/thermoelastic-rod/AUTHOR.md) |

Phoresis revision 2 has three reviewed physical failures in three completed unhinted Luna-high trials. Its [promotion audit](results/zero-three-phoresis-r2-promotion.json) also preserves one started outage interruption and two unstarted setup failures; those are not physics failures. It replaces [electric-dipole-force revision 1](archives/electric-dipole-force-r1), which passed the fresh confirmation 3/3. Source author notes describe the frozen pre-evaluation revision; current outcomes are in the dashboard.

Single-file revision 1 also has three reviewed physical failures in three completed unhinted trials. All three fit the one-time variance but retain a stationary-increment covariance inappropriate for the fixed initial particle lattice. Its [promotion audit](results/zero-three-singlefile-r1-physics-promotion-audit.json) keeps the one unstarted TLS setup failure separate. It replaces [spin-strain-response revision 1](archives/spin-strain-response-r1), which passed the fresh confirmation 3/3.

Weak and superseded candidates and their prior results are preserved under `archives/`; the [screened archive](archives/screened/README.md) holds tasks that missed the earlier cutoff of at most one pass in three. Earlier task explanations, scores and hint comparisons remain in the [pre-neutral archive](archives/pre-neutral-instructions).

Each public task must define its initial and boundary conditions, preparation, observables and limit order. These determine the physical problem. The missing correction, derivation, fitted true value, hidden cases and diagnostic hint stay outside the agent image. No particular solver, discretization or oracle representation is required.

Each completed shortcut must fit calibration and identify the unknown parameter before failing an allowed prediction. Independent references, repeated-noise checks and oracle/shortcut controls establish scientific validity. Only fresh frozen agent runs establish observed difficulty. Earlier outcomes stay attached to their evaluated sources.
