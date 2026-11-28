# Task physics

The [active roster](results/task-roster.json) contains 50 evaluated snapshots. Each public README defines its apparatus and API, and AUTHOR.md gives the derivation. October outcomes below refer to their original batches.

The hierarchical-glass task compares a quenched two-level Gaussian tree with an independent-coordinate approximation that has the same warm caloric law. It measures common-parent, common-state and exact labelled leaf partitions for two through six replicas. Selection of a parent reweights its descendants by a fractional power of the subtree partition sum; ignoring this correlation caused all three reviewed failures.

The original ten task instructions permit replacing all implementation code; nine share a neutral instruction and hierarchical-glass-overlaps has its own. The three additions use task-specific instructions that permit changing the response model while preserving the public API. Current outcomes are in [DASHBOARD.md](DASHBOARD.md); archived scores do not describe revised sources. The descriptions below are author material and are excluded from agent images.

| Roster task | Physical issue | Public apparatus | Derivation and validation |
|---|---|---|---|
| entropy-anomaly | A spatially weighted calorimeter detects a small lag in local contact populations that equal local contact occupancy omits. | [README](tasks/entropy-anomaly/environment/README.md) | [AUTHOR](tasks/entropy-anomaly/AUTHOR.md) |
| finite-layer-phoresis | Navier slip changes the transmission of solute interaction force as well as translational drag. | [README](tasks/finite-layer-phoresis/environment/README.md) | [AUTHOR](tasks/finite-layer-phoresis/AUTHOR.md) |
| hierarchical-glass-overlaps | Selecting a parent reweights its descendants by a fractional power of the subtree partition sum; independent coordinates miss this correlation. | [README](tasks/hierarchical-glass-overlaps/environment/README.md) | [AUTHOR](tasks/hierarchical-glass-overlaps/AUTHOR.md) |
| insulating-mhd-duct | Oblique magnetic fields share one electrical current-closure problem. | [README](tasks/insulating-mhd-duct/environment/README.md) | [AUTHOR](tasks/insulating-mhd-duct/AUTHOR.md) |
| magnetic-tracer | Directional fluid baths correlate longitudinal motion with transverse circulation. The mixed diffusion term changes circulation noise and skewness while density and mean circulation remain correct. | [README](tasks/magnetic-tracer/environment/README.md) | [AUTHOR](tasks/magnetic-tracer/AUTHOR.md) |
| reaction-diffusion | Equilibrium association and charge constraints correlate conserved constituent totals; independent Poisson totals give the wrong fluctuation covariance even with correct mean transport. | [README](tasks/reaction-diffusion/environment/README.md) | [AUTHOR](tasks/reaction-diffusion/AUTHOR.md) |
| ring-current-fluctuations | Strongly suppressed currents in a driven half-filled exclusion ring favor a nonuniform density profile, not the homogeneous one. | [README](tasks/ring-current-fluctuations/environment/README.md) | [AUTHOR](tasks/ring-current-fluctuations/AUTHOR.md) |
| rotating-reservoir | Angular detector gain remains correlated with kinetic heat exchange; averaging detector phase before computing noise and higher cumulants discards those correlations. | [README](tasks/rotating-reservoir/environment/README.md) | [AUTHOR](tasks/rotating-reservoir/AUTHOR.md) |
| single-file-memory | A fixed particle preparation retains two-time memory that stationary increments miss. | [README](tasks/single-file-memory/environment/README.md) | [AUTHOR](tasks/single-file-memory/AUTHOR.md) |
| superconducting-heat | State-dependent spin feedback changes the third cumulant of gate work. A fixed-rate Poisson model with linearly screened work marks omits the nonlinear rate response. | [README](tasks/superconducting-heat/environment/README.md) | [AUTHOR](tasks/superconducting-heat/AUTHOR.md) |
| swirling-heat-tails | Rare initial and final trap energies restrict the heat generating function even at long times; an unrestricted work rate misses both heat tails. | [README](tasks/swirling-heat-tails/environment/README.md) | [AUTHOR](tasks/swirling-heat-tails/AUTHOR.md) |
| hard-rod-current | Rare currents select an atypically short block of crossed rods; using only global mean composition loses the annealed mixture fluctuation. | [README](tasks/hard-rod-current/environment/README.md) | [AUTHOR](tasks/hard-rod-current/AUTHOR.md) |
| finite-ion-cloud | Three pair angles share three particle positions; independently averaging equilibrium pair angles misses angular frustration and finite-cell pressure. | [README](tasks/finite-ion-cloud/environment/README.md) | [AUTHOR](tasks/finite-ion-cloud/AUTHOR.md) |

## November 27 additions

| Task | Physical issue | Derivation |
|---|---|---|
| [Weak-field condensate coincidences](tasks/weak-field-condensate-coincidences/environment/README.md) | A volume-scaled splitting biases the occupied condensate modes, whose coherent rotations affect same-shot coincidences. | [Author notes](tasks/weak-field-condensate-coincidences/AUTHOR.md) |
| [Magnetization-constrained condensate](tasks/magnetization-constrained-condensate/environment/README.md) | Exact magnetization couples thermal saturation to the populations and phases seen after a mixing pulse. | [Author notes](tasks/magnetization-constrained-condensate/AUTHOR.md) |
| [Switched axis heat tails](tasks/switched-axis-heat-tails/environment/README.md) | Ordered trap changes and endpoint heat terms determine the calorimeter's tails. | [Author notes](tasks/switched-axis-heat-tails/AUTHOR.md) |
| [Two-charge condensate fringes](tasks/two-charge-condensate-fringes/environment/README.md) | Two fixed charges constrain the ground-mode sharing after thermal excitation. | [Author notes](tasks/two-charge-condensate-fringes/AUTHOR.md) |
| [Band-edge emitter memory](tasks/band-edge-emitter-memory/environment/README.md) | Band-edge continuum and discrete spectral components jointly determine the return amplitude. | [Author notes](tasks/band-edge-emitter-memory/AUTHOR.md) |
| [Marked branch glass overlap](tasks/marked-branch-glass-overlap/environment/README.md) | Shared branching labels propagate the Gibbs selection bias to both overlap depths. | [Author notes](tasks/marked-branch-glass-overlap/AUTHOR.md) |
| [Shared-field glass families](tasks/shared-field-glass-families/environment/README.md) | Two fields change weights of the same frozen families and retain correlated draw probabilities. | [Author notes](tasks/shared-field-glass-families/AUTHOR.md) |
| [Merged depth glass marks](tasks/merged-depth-glass-marks/environment/README.md) | Depths that freeze jointly change the effective random measure of the descendant branches. | [Author notes](tasks/merged-depth-glass-marks/AUTHOR.md) |
| [Critical glass bridge marks](tasks/critical-glass-bridge-marks/environment/README.md) | Gaussian bridge feasibility and Gibbs tilting constrain paired critical-state marks. | [Author notes](tasks/critical-glass-bridge-marks/AUTHOR.md) |
| [Coexisting frozen landscapes](tasks/coexisting-frozen-landscapes/environment/README.md) | Both phase partition sums compete in the same quenched normalization. | [Author notes](tasks/coexisting-frozen-landscapes/AUTHOR.md) |
| [Two phase sites in selected families](tasks/two-phase-sites-in-selected-families/environment/README.md) | Phase mass within a selected family governs repeated-site probabilities as well as single-draw rates. | [Author notes](tasks/two-phase-sites-in-selected-families/AUTHOR.md) |
| [Spectral hole feedback](tasks/spectral-hole-feedback/environment/README.md) | A Gibbs draw changes the posterior family populations that govern the next pulse. | [Author notes](tasks/spectral-hole-feedback/AUTHOR.md) |
| [Double critical sibling cloud](tasks/double-critical-sibling-cloud/environment/README.md) | The ordinary cloud obeys cumulative Gaussian constraints and shares a normalizer with a rare family. | [Author notes](tasks/double-critical-sibling-cloud/AUTHOR.md) |
| [Critical crossed address echo](tasks/critical-crossed-address-echo/environment/README.md) | A rare row and column contribute both strips and their intersection to the Gibbs mass. | [Author notes](tasks/critical-crossed-address-echo/AUTHOR.md) |
| [Commensurate critical domain echo](tasks/commensurate-critical-domain-echo/environment/README.md) | Discrete addressed energies coexist with a cumulatively constrained ordinary Gaussian cloud. | [Author notes](tasks/commensurate-critical-domain-echo/AUTHOR.md) |
| [Microcanonical sibling spectroscopy](tasks/microcanonical-sibling-spectroscopy/environment/README.md) | The count-normalized spectrum includes only ordinary extreme sites on the feasible bridge half-line. | [Author notes](tasks/microcanonical-sibling-spectroscopy/AUTHOR.md) |
| [Aperture-selected critical roots](tasks/aperture-selected-critical-roots/environment/README.md) | Separate aperture-restricted Gibbs laws share a rare root and allow negative increments within feasible paths. | [Author notes](tasks/aperture-selected-critical-roots/AUTHOR.md) |

## November 28 additions

| Task | Physical issue | Derivation |
|---|---|---|
| [Entropy-shifted branch echo](tasks/entropy-shifted-branch-echo/environment/README.md) | A rare two-edge prefix contributes a conditioned mass within the cumulative Gaussian cone. | [Author notes](tasks/entropy-shifted-branch-echo/AUTHOR.md) |
| [Critical gamma cavity recurrence](tasks/critical-gamma-cavity-recurrence/environment/README.md) | A non-Gaussian cavity line contributes finite Gibbs mass to its own denominator. | [Author notes](tasks/critical-gamma-cavity-recurrence/AUTHOR.md) |
| [Twin resonant family calorimeter](tasks/twin-resonant-family-calorimeter/environment/README.md) | A single tagged site changes both preparations while the opposite tag population keeps its original law. | [Author notes](tasks/twin-resonant-family-calorimeter/AUTHOR.md) |
| [Conjugate child resonance](tasks/conjugate-child-resonance/environment/README.md) | Opposite child energies share a root whose conditioned descendants enter both thermal weights. | [Author notes](tasks/conjugate-child-resonance/AUTHOR.md) |
| [Two-root heralded returns](tasks/two-root-heralded-returns/environment/README.md) | Two independent root-count records condition the diffuse clouds and the addressed site. | [Author notes](tasks/two-root-heralded-returns/AUTHOR.md) |
| [Antipodal color echo](tasks/antipodal-color-echo/environment/README.md) | Opposing energy windows couple the allowed root range to both color normalizers. | [Author notes](tasks/antipodal-color-echo/AUTHOR.md) |
| [Dual shell count herald](tasks/dual-shell-count-herald/environment/README.md) | Overlapping shells share one count class in the marked Poisson process. | [Author notes](tasks/dual-shell-count-herald/AUTHOR.md) |
| [Middle inversion site lock](tasks/middle-inversion-site-lock/environment/README.md) | Inverting the shared middle increment couples the ordinary and conditioned feasibility cones. | [Author notes](tasks/middle-inversion-site-lock/AUTHOR.md) |
| [Diluted common catalog return](tasks/diluted-common-catalog-return/environment/README.md) | Shared label entropy and unequal cutoffs set the selected mass in partially shared catalogs. | [Author notes](tasks/diluted-common-catalog-return/AUTHOR.md) |
| [Row-column bond echo](tasks/row-column-bond-echo/environment/README.md) | An entropy surplus in one coordinate permits a pooled saddle and distinct row and bond corrections. | [Author notes](tasks/row-column-bond-echo/AUTHOR.md) |
| [Feedback-selected family aperture](tasks/feedback-selected-family-aperture/environment/README.md) | The first recorded draw determines a later aperture and the weight of its retained family. | [Author notes](tasks/feedback-selected-family-aperture/AUTHOR.md) |
| [Quantized root liquid line](tasks/quantized-root-liquid-line/environment/README.md) | A lattice root intensity couples to a continuous Gamma leaf intensity. | [Author notes](tasks/quantized-root-liquid-line/AUTHOR.md) |
| [Tangent aperture recurrence](tasks/tangent-aperture-recurrence/environment/README.md) | Two tangent apertures constrain the same microscopic line and retain its finite mass. | [Author notes](tasks/tangent-aperture-recurrence/AUTHOR.md) |
| [Thin oblique critical aperture](tasks/thin-oblique-critical-aperture/environment/README.md) | The limiting ordinary gate changes from a shrinking wedge to an oblique segment as its center moves off zero. | [Author notes](tasks/thin-oblique-critical-aperture/AUTHOR.md) |
| [Phase-tagged Bell return](tasks/phase-tagged-bell-return/environment/README.md) | A phase-sensitive Bell projection acts on spins whose full thermal traces determine the site probabilities. | [Author notes](tasks/phase-tagged-bell-return/AUTHOR.md) |
| [Deep root photon herald](tasks/deep-root-photon-herald/environment/README.md) | The thinned deep family has a conditional compound Poisson population in the second gate. | [Author notes](tasks/deep-root-photon-herald/AUTHOR.md) |
| [Current coexistence dwell](tasks/current-coexistence-dwell/environment/README.md) | Exact counter conditioning gives half weight to a first-switch record at its limiting window edge. | [Author notes](tasks/current-coexistence-dwell/AUTHOR.md) |
| [Frozen root critical leaf echo](tasks/frozen-root-critical-leaf-echo/environment/README.md) | A stable cold root partition shares a selected leaf with a critical thermal replica. | [Author notes](tasks/frozen-root-critical-leaf-echo/AUTHOR.md) |
| [Coherent dual-sector line lock](tasks/coherent-dual-sector-line-lock/environment/README.md) | Finite coherent mixing survives in the rare dual-sector block and vanishes in the ordinary Gibbs mass. | [Author notes](tasks/coherent-dual-sector-line-lock/AUTHOR.md) |
| [Crossbar finite-site latch](tasks/crossbar-finite-site-latch/environment/README.md) | Two strips, a diffuse intersection and the finite tagged site share the ordinary Gibbs denominator. | [Author notes](tasks/crossbar-finite-site-latch/AUTHOR.md) |

The original roster's October outcomes below distinguish the initial qualification from fresh confirmation. The three additions originally scored 0/3 each, while their fresh confirmation scored 1/3 for swirling heat tails and 0/3 for the other two. One finite-ion-cloud failure combines physical and mathematical errors. Earlier one-line descriptions of entropy anomaly, rotating reservoir and superconducting heat are preserved in the [pre-single-file documentation archive](archives/pre-singlefile-promotion/TASKS.md).

The following inherited candidate folders are retained in the archive.

| Inherited folder | Physical issue | Public apparatus | Derivation and validation |
|---|---|---|---|
| hydrodynamic-heating | Regional heating follows carrier drag dissipation while pressure transports conservative energy. | [README](archives/excluded-inherited-20261127/hydrodynamic-heating/environment/README.md) | [AUTHOR](archives/excluded-inherited-20261127/hydrodynamic-heating/AUTHOR.md) |
| thermoelastic-rod | Mechanical equilibration requires the viscosity-weighted common-stress constraint. | [README](archives/excluded-inherited-20261127/thermoelastic-rod/environment/README.md) | [AUTHOR](archives/excluded-inherited-20261127/thermoelastic-rod/AUTHOR.md) |

Phoresis revision 2 has three reviewed physical failures in three completed unhinted Luna-high trials. Its [promotion audit](results/zero-three-phoresis-r2-promotion.json) also preserves one started outage interruption and two unstarted setup failures; those are not physics failures. It replaces [electric-dipole-force revision 1](archives/electric-dipole-force-r1), which passed the fresh confirmation 3/3. Source author notes describe the frozen pre-evaluation revision; current outcomes are in the dashboard.

Single-file revision 1 also has three reviewed physical failures in three completed unhinted trials. All three fit the one-time variance but retain a stationary-increment covariance inappropriate for the fixed initial particle lattice. Its [promotion audit](results/zero-three-singlefile-r1-physics-promotion-audit.json) keeps the one unstarted TLS setup failure separate. It replaces [spin-strain-response revision 1](archives/spin-strain-response-r1), which passed the fresh confirmation 3/3.

Weak and superseded candidates and their prior results are preserved under `archives/`; the [screened archive](archives/screened/README.md) holds tasks that missed the earlier cutoff of at most one pass in three. Earlier task explanations, scores and hint comparisons remain in the [pre-neutral archive](archives/pre-neutral-instructions).

Each public task must define its initial and boundary conditions, preparation, observables and limit order. These determine the physical problem. The missing correction, derivation, fitted true value, hidden cases and diagnostic hint stay outside the agent image. No particular solver, discretization or oracle representation is required.

Each completed shortcut must fit calibration and identify the unknown parameter before failing an allowed prediction. Independent references, repeated-noise checks and oracle/shortcut controls establish scientific validity. Only fresh frozen agent runs establish observed difficulty. Earlier outcomes stay attached to their evaluated sources.
