# PhysLiteBench evaluation dashboard

The active target is now **ten tasks at 0/3 each** in unhinted Luna-high trials, with predominantly physical-model failures. **5 of 10 currently qualify**: finite-layer-phoresis, insulating-mhd-duct, magnetic-tracer, reaction-diffusion and single-file-memory. The other five require revision or replacement. [Current goal status](results/zero-three-goal-status.json) records source checks and the stricter protocol; the completed batches below remain historical evidence.

The earlier confirmation batch on the previous roster is complete: **6 of 10 tasks meet the at-most-1/3 pass cutoff**. There were **13 passes in 30 new unhinted Luna-high trials**. 3 tasks scored 0/3. Above cutoff: `electric-dipole-force` (3/3), `entropy-anomaly` (2/3), `spin-strain-response` (3/3), `superconducting-heat` (2/3).

| Task | Fresh passes/trials | At most 1/3? |
|---|---:|---|
| electric-dipole-force | 3/3 | No |
| entropy-anomaly | 2/3 | No |
| hydrodynamic-heating | 1/3 | Yes |
| insulating-mhd-duct | 0/3 | Yes |
| magnetic-tracer | 0/3 | Yes |
| reaction-diffusion | 0/3 | Yes |
| rotating-reservoir | 1/3 | Yes |
| spin-strain-response | 3/3 | No |
| superconducting-heat | 2/3 | No |
| thermoelastic-rod | 1/3 | Yes |

These are three fresh trials per task on the previous roster. The current roster replaces electric-dipole-force with finite-layer-phoresis and spin-strain-response with single-file-memory; their results are listed below. The earlier 0/1 screening result for each task is kept separate. All 30 outcomes and prior revisions are retained; there was no outcome-dependent stopping, source revision or resampling during this batch. These observed counts do not establish population success probabilities.

The frozen files match the [batch plan](results/retained-three-r1-plan.json). Eight of those revisions remain canonical; [electric-dipole-force](archives/electric-dipole-force-r1) and [spin-strain-response](archives/spin-strain-response-r1) are archived unchanged. All native sessions use `gpt-5.6-luna`, `high`, and Codex CLI 0.154.0, with the exact neutral task instruction and reviewed infrastructure inputs. Full counts, metrics, classifications and evidence are in [the fresh-batch report](results/retained-three-r1-results.json).

## Current 0/3 hardening candidates

These are new scientific revisions under the stricter target. Each gets Docker controls and exactly three fresh unhinted trials. A candidate is retained only after source/input checks and causal review; passing results remain recorded.

| Candidate | Fresh passes/trials | Status |
|---|---:|---|
| radiative-angular-closure r1 | 3/3 | [Archived: three physical passes](results/zero-three-radiative-r1-results.json) |
| prestrained-solid r1 | 2/3 | [Archived: two physical passes, one mixed failure](results/zero-three-prestrained-r1-results.json) |
| hydrodynamic-heating r2 | 2/3 | [Archived: two physical passes, one physical failure](results/zero-three-hydro-r2-results.json) |
| terminal-current-noise r1 | 3/3 | [Archived: three physical passes](results/zero-three-terminal-r1-results.json) |
| ionic-current-loops r2 | 1/3 | [Archived: two physical failures, one physical pass; one unstarted setup failure preserved](results/zero-three-ionic-r2-results.json) |
| poroelastic-relaxation r1 | 3/3 | [Archived: three physical passes](results/zero-three-poroelastic-r1-results.json) |
| isolated-spheres r1 | 3/3 | [Archived: three physical passes](results/zero-three-spheres-r1-results.json) |
| hot-brownian-rotation r1 | 3/3 | [Archived: three physical passes](results/zero-three-hotrotation-r1-results.json) |
| pressure-surface-waves r1 | 2/3 | [Archived: two physical passes; one algebra-influenced regression](results/zero-three-pressurewaves-r1-results.json) |
| entropy-anomaly r4 | 2/3 | [Archived: two physical passes; one physical failure](results/zero-three-entropy-r4-results.json) |
| finite-layer-phoresis r1 | 1/3 | [Archived: two physical failures; one physical pass](results/zero-three-phoresis-r1-results.json) |
| finite-band-reservoir r1 | 2/3 | [Archived: two physical passes; one physical model-selection failure](results/zero-three-finiteband-r1-results.json) |
| ionic-current-loops r3 | 1/3 | [Archived: two physical failures; one physical pass](results/zero-three-ionic-r3-results.json) |
| finite-layer-phoresis r2 | 0/3 completed | [Three physical failures; one started interruption and two setup attempts preserved; promoted; replaced electric-dipole-force r1](results/zero-three-phoresis-r2-results.json) |
| hardcore-coherence r1 | 3/3 completed | [Archived: three physical passes; two started infrastructure interruptions preserved](results/zero-three-hardcore-r1-results.json) |
| hardcore-ring r1 | 3/3 | [Archived: three physical passes](results/zero-three-ring-r1-results.json) |
| acoustic-cavity-pressure r1 | 3/3 | [Archived: three physical passes](results/zero-three-acoustic-r1-results.json) |
| single-file-memory r1 | 0/3 completed | [Promoted: three physical failures; one unstarted setup failure preserved; replaced spin-strain-response r1](results/zero-three-singlefile-r1-results.json) |
| nematic-wall-torque r1 | 3/3 | [Archived: three physical passes](results/zero-three-nematic-r1-results.json) |
| dielectric-flow r2 | 3/3 | [Archived: three physical passes](results/zero-three-dielectric-r2-results.json) |
| active-wall-pressure r1 | 3/3 | [Archived: three physical passes](results/zero-three-activewall-r1-results.json) |
| fene-stress r1 | 1/3 | [Archived: one physical pass, two physical failures](results/zero-three-fene-r1-results.json) |
| relativistic-snapshot r1 | 2/3 | [Archived: two physical passes, one physical failure](results/zero-three-relativistic-r1-results.json) |
| fene-stress r2 | 1/3 | [Archived: one pass, one clean physical failure, one mixed failure](results/zero-three-fene-r2-results.json) |
| resonance-fluorescence r1 | 3/3 | [Archived: three physical passes](results/zero-three-fluorescence-r1-results.json) |
| gated-capture r1 | 3/3 | [Archived: three physical passes](results/zero-three-gated-r1-results.json) |
| thermal-unsteady-sphere r1 | 3/3 | [Archived: three physical passes](results/zero-three-thermal-sphere-r1-results.json) |
| fixed-trap-memory r1 | 3/3 | [Archived: three physical passes](results/zero-three-fixed-trap-r1-results.json) |
| survivor-transport r1 | 2/3 | [Archived: 2 pass, 1 clean physical failure, 0 mixed failure](results/zero-three-survivor-r1-results.json) |
| collisionless-screening r1 | 1/3 | [Archived: 1 pass, 0 clean physical failure, 1 mixed failure, 1 normalization failure; 2 interruptions preserved](results/zero-three-collisionless-r1-results.json) |
| waving-sheet r1 | 2/3 | [Archived: 2 pass, 1 clean physical failure, 0 mixed failure, 0 normalization failure; 3 interruptions preserved](results/zero-three-waving-r1-results.json) |
| chemical-route-power r1 | 3/3 | [Archived: three physical passes](results/zero-three-chemical-route-r1-results.json) |
| fixed-number-band r1 | 3/3 | [Archived: 3 pass, 0 clean physical failure, 0 mixed failure](results/zero-three-band-r1-results.json) |
| waving-sheet r2 | 1/3 | [Archived: 1 pass, 1 clean physical failure, 1 mixed failure](results/zero-three-waving-r2-results.json) |
| adiabatic-capture r1 | Pending | [Six unstarted setup failures preserved; three replacements running with reviewed setup allowance](results/zero-three-adiabatic-coherent-setup-budget-plan.json) |
| coherent-array r1 | Pending | [Science and input review complete; frozen serial evaluation plan](results/zero-three-coherent-r1-plan.json) |
| finite-chain-clamp r1 | 3/3 | [Archived: three physical passes; three unstarted setup failures preserved](results/zero-three-chain-r1-results.json) |

The [deforming-crystal current prototype](results/zero-three-deforming-crystal-feasibility-source-review.json) passed bounded scientific and input review. Its full task is under construction; no model difficulty result is available.

The [Docker outage review](results/docker-outage-20261004-080748.json) predeclares exact replacements for five unscored attempts lost in a host-wide shutdown. Every interruption remains recorded; no completed score is replaced and no infrastructure error counts as a physics failure.

The [later Docker shutdown review](results/docker-outage-20261004-140542.json) preserves two plasma and three waving-sheet interruptions with null rewards. Exactly five replacements completed the frozen batches; the original completed plasma failure was retained. Neither candidate meets the zero-pass target, and the plasma failures also miss the required physical failure mode.

## Prior conditional screen

Updated 2026-10-03 05:38 UTC. The requested conditional screen is complete. **The ten tasks selected at that time each had a reviewed physical failure on their first run.** The old ten-task 0/3 result applies only to [archived instructions and sources](archives/pre-neutral-instructions).

That screen used one unhinted trial per changed revision, followed by exactly two more if the first passed. An initial failure stops at **0/1**. These counts do not establish a new 0/3 result or guarantee future failures. Every initial result and follow-up is retained. Reviewed setup failures before model execution remain recorded as attempts and are excluded from model-trial counts only when the strict unstarted-retry criteria are met. The original target of at least ten 0/3 tasks has not been re-established under the neutral instruction.

| Historical screened task | Scientific revision at that screen | Unhinted passes/trials | Status |
|---|---|---:|---|
| electric-dipole-force | Optical force on an electric dipole, r1 | 0/1 | Reviewed physical failure |
| entropy-anomaly | Magnetic kinetic heat transport, r3 | 0/1 | Reviewed physical failure |
| hydrodynamic-heating | Electron-fluid calorimetry, r1 | 0/1 | Reviewed physical failure |
| insulating-mhd-duct | Oblique magnetic field, r2 | 0/1 | Reviewed physical failure |
| magnetic-tracer | Finite-mass mechanical area, r2 | 0/1 | Reviewed physical failure |
| reaction-diffusion | Coupled local conductivity, r7 | 0/1 | Reviewed physical failure |
| rotating-reservoir | Calorimetry of a rotating bath, r1 | 0/1 | Reviewed physical failure |
| spin-strain-response | Static molecular force response, r1 | 0/1 | Reviewed physical failure |
| superconducting-heat | Phase-biased BCS heat transfer, r1 | 0/1 | Reviewed physical failure |
| thermoelastic-rod | Heterogeneous Kelvin–Voigt rod, r3 | 0/1 | Reviewed physical failure |

For each reviewed physical failure, the submission fits calibration and recovers the parameter. A diagnostic correction at that same fitted parameter passes the hidden predictions. Reviews separate physical mistakes from algebra, implementation and infrastructure failures.

The following revisions were archived after review because they exceeded the pass-rate cutoff or missed the required physical failure mode. Prior revisions and every passing result remain available; none were replaced with unchanged retries.

| Archived family | Latest unhinted passes/trials | Archive |
|---|---:|---|
| magnetic-bath-transfer | 3/3 | [Latest source and evidence](archives/magnetic-bath-transfer-r1) |
| geometric-rotor r2 | 0/1 (implementation error) | [Latest source and evidence](archives/geometric-rotor-r2) |
| geometric-rotor r1 | 3/3 | [Latest source and evidence](archives/geometric-rotor-r1) |
| paired-spin-noise | 3/3 | [Latest source and evidence](archives/paired-spin-noise-r1) |
| spin-body-torque | 3/3 | [Latest source and evidence](archives/spin-body-torque-r1) |
| delta-box-force | 3/3 | [Latest source and evidence](archives/delta-box-force-r1) |
| quantum-work r2 | 3/3 | [Latest source and evidence](archives/quantum-work-r2) |
| quantum-work r1 | 2/3 (one implementation failure) | [Latest source and evidence](archives/quantum-work-r1) |
| paired-resonators | 3/3 | [Latest source and evidence](archives/paired-resonators-r1) |
| quantum-calorimetry | 3/3 | [Latest source and evidence](archives/screened/quantum-calorimetry-r1) |
| gyroscopic-noise | 3/3 | [Latest source and evidence](archives/gyroscopic-noise-r1) |
| boson-hall | 0/1 (sign error) | [Latest source and evidence](archives/screened/boson-hall-r1) |
| resonator-energy | 3/3 | [Latest source and evidence](archives/resonator-energy-neutral-r1) |
| optical-torque | 3/3 | [Latest source and evidence](archives/screened/optical-torque-r1) |
| bogoliubov-momentum | 3/3 | [Latest source and evidence](archives/bogoliubov-momentum-r1) |
| dressed-photodetection | 3/3 | [Latest source and evidence](archives/screened/dressed-photodetection-r1) |
| actuator-frame-current | 3/3 | [Latest source and evidence](archives/actuator-frame-current-r1) |
| circular-detector | 3/3 | [Latest source and evidence](archives/screened/circular-detector-r1) |
| magnetic-bilayer | 2/3 | [Latest source and evidence](archives/magnetic-bilayer-neutral-r1) |
| boson-transfer-noise | 3/3 | [Latest source and evidence](archives/boson-transfer-noise-r1) |
| emitter-lifetime | 3/3 | [Latest source and evidence](archives/screened/emitter-lifetime-r1) |
| elastic-ring | 3/3 | [Latest source and evidence](archives/elastic-ring-neutral-r1) |
| metal-plate-force | 3/3 | [Latest source and evidence](archives/metal-plate-force-r1) |
| active-bath-work | 3/3 | [Latest source and evidence](archives/active-bath-work-r1) |
| quantum-residence | 3/3 | [Latest source and evidence](archives/screened/quantum-residence-r1) |
| capillary-mixture | 3/3 | [Latest source and evidence](archives/capillary-mixture-neutral-r2) |
| thermoelectric-rod | 3/3 | [Latest source and evidence](archives/thermoelectric-rod-neutral-r2) |
| ternary-gas | 3/3 | [Latest source and evidence](archives/ternary-gas-neutral-r2) |
| rotating-layer | 2/3 | [Latest source and evidence](archives/rotating-layer-neutral-r1) |
| hall-bar | 3/3 | [Latest source and evidence](archives/screened/hall-bar-neutral-r3) |
| nonreciprocal-cavity | 3/3 | [Latest source and evidence](archives/nonreciprocal-cavity-neutral-r3) |
| qubit-control | 3/3 | [Latest source and evidence](archives/qubit-control-r9) |
| superconducting-ring | 3/3 | [Latest source and evidence](archives/superconducting-ring-neutral-r1) |
| thermal-bodies | 3/3 | [Latest source and evidence](archives/screened/thermal-bodies-r12) |

The first neutral screen produced 29 passing rewards in 32 trials: nine tasks returned 3/3, qubit-control returned 2/3, and entropy-anomaly and nonreciprocal-cavity returned 0/1. One MHD passing reward also had an agent timeout; both the solution and exception are recorded. All 32 submissions were reviewed. See [the complete initial screen](results/neutrality-initial-screen.json).

The cavity audit found a calibration side channel: uncertainty encoded the noiseless response. After replacing it with uncertainty determined by known instrument controls, that revision passed 2/3. A further physical revision passed 3/3 and was archived. Its earlier 0/1 is historical.

Runs use GPT-5.6 Luna with high reasoning effort, Harbor 0.21.0, Codex CLI 0.154.0 and 600/60-second agent/verifier limits. Native sessions verify the setup. Before agent evaluation, each scientific revision receives an independent physics review, repeated-noise validation, and oracle/shortcut controls. Inputs and grading stay frozen throughout each batch.

[Source-matched conditional-screen status](results/neutrality-status.json) is generated from [the complete candidate ledger](results/candidates.json). Reviews under `results/*trial-reviews.json` retain source hashes, public reasoning, verifier metrics and diagnostic repairs. Frozen tasks, native sessions and submissions remain in ignored `jobs/` directories.

The [historical instruction audit](results/instruction-audit.json) identified the fit-only scope ambiguity and cavity side channel. The [conditional-screen input audit](results/neutrality-current-input-audit.json) records exact neutral prompts, source hashes, public image boundaries and native inputs. Public apparatus definitions retain necessary physical assumptions without prescribing the missing correction or a numerical method. Draft entries are explicitly marked pending review.
