# PhysLiteBench evaluation dashboard

Updated 2026-10-03 05:38 UTC. The requested conditional screen is complete. **10 current tasks have a reviewed physical failure on their first run.** The old ten-task 0/3 result applies only to [archived instructions and sources](archives/pre-neutral-instructions).

The approved protocol is one unhinted trial per changed revision, followed by exactly two more if the first passes. An initial failure stops at **0/1**. These counts do not establish a new 0/3 result or guarantee future failures. Every initial result and follow-up is retained. Reviewed setup failures before model execution remain recorded as attempts and are excluded from model-trial counts only when the strict unstarted-retry criteria are met. The original target of at least ten 0/3 tasks has not been re-established under the neutral instruction.

| Active task | Current scientific revision | Unhinted passes/trials | Status |
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

[Source-matched current status](results/neutrality-status.json) is generated from [the complete candidate ledger](results/candidates.json). Reviews under `results/*trial-reviews.json` retain source hashes, public reasoning, verifier metrics and diagnostic repairs. Frozen tasks, native sessions and submissions remain in ignored `jobs/` directories.

The [historical instruction audit](results/instruction-audit.json) identified the fit-only scope ambiguity and cavity side channel. The [current input audit](results/neutrality-current-input-audit.json) records exact neutral prompts, source hashes, public image boundaries and native inputs. Public apparatus definitions retain necessary physical assumptions without prescribing the missing correction or a numerical method. Draft entries are explicitly marked pending review.
