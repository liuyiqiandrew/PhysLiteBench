# PhysLiteBench evaluation dashboard

Updated 2026-10-02. **Ten retained tasks score 0/3 unhinted, and the other two score 1/3.** All thirty failures in the ten zero-score tasks have reviewed physical-model causes. Candidates exceeding the cutoff are archived with their prior revisions and results.

All new agent batches use GPT-5.6 Luna, high reasoning effort, Harbor 0.21.0, Codex CLI 0.154.0, and 600/60-second agent/verifier limits. Each ordinary/hinted comparison has three trials per condition. Native session logs verify the model and effort; frozen source hashes identify the exact revision. Only the latest completed revision of a task contributes to selection. Earlier outcomes are never replaced by rerunning unchanged files until a favorable batch appears.

| Task | Revision | Unhinted passes | Hinted passes | Reviewed cause of unhinted failures |
|---|---:|---:|---:|---|
| thermal-bodies | 9 | 0/3 | 3/3 | Holds pressure constant despite a free internal piston. |
| reaction-diffusion | 6 | 0/3 | 2/3 | Uses independent salt diffusion instead of the shared electric field. |
| ternary-gas | 1 | 0/3 | 3/3 | Mixture-averaged flux does not satisfy pairwise friction balance. |
| insulating-mhd-duct | 1 | 0/3 | 3/3 | Local magnetic drag omits cross-sectional current closure. |
| thermoelectric-rod | 1 | 0/3 | 3/3 | Fourier/Joule heating omits the Thomson term. |
| thermoelastic-rod | 2 | 0/3 | 3/3 | Local heat capacity omits the weighted global clamp constraint. |
| magnetic-tracer | 1 | 0/3 | 3/3 | Scalar magnetic diffusivity omits antisymmetric transport. |
| entropy-anomaly | 3 | 0/3 | 3/3 | Magnetic rotation changes the unresolved velocity heat response. |
| capillary-mixture | 1 | 0/3 | 3/3 | Isotropic pressure-only forcing omits the capillary stress that drives flow. |
| nonreciprocal-cavity | 1 | 0/3 | 2/3 | Uses incoming absorptivity for outgoing thermal emission despite broken reciprocity. |
| qubit-control | 6 | 1/3 | 3/3 | One clean shared-noise failure and one mixed math/model-selection failure. |
| hall-bar | 1 | 1/3 | 3/3 | Two failures impose zero transverse field at insulating sidewalls. |

All ten zero-score tasks fail for reviewed physical-model causes after passing calibration and parameter recovery. For the new tasks, fixed-parameter diagnostic repairs restore the intended physical term without changing the fitted parameter or the recorded agent result. The corresponding reviews and repairs are linked from [the full trial ledger](results/candidates.json).

The original thermal and reaction unhinted batches are historical, not fresh reruns. Current public environments and grading match their frozen snapshots exactly. Their new hints use the same setup and change only the instruction: [source audit](results/hardening-original-audit.json), [paired audit](results/original-hint-paired-audit.json), and [hint reviews](results/original-hint-trial-reviews.json).

Reaction's one hinted failure selects the correct common-field physics but implements one diffusion-matrix coefficient incorrectly. Repairing only that coefficient at its retained fitted diffusivity passes: [diagnostic](results/reaction-original-hint-coefficient-repair.json). The qubit mixed failure remains explicitly classified as such and contributes nothing to the ten required zero-score tasks.

Entropy revision 3 retains correct magnetic position transport but incorrectly reuses the zero-field kinetic heat coefficient. All three plain agents kept that closure. Replacing only the cubic velocity-relaxation coefficient at the fitted friction repairs every hidden prediction and leaves calibration unchanged: [repair evidence](results/entropy-anomaly-r3-fixed-friction-repair.json). Its initial oracle control timed out before any agent trial. The unchanged physics and 60-second limit passed after limiting BLAS/OMP to one thread; [timeout provenance](archives/entropy-anomaly-r3-control-timeout/manifest.json) is preserved separately.

The nonreciprocal cavity has the correct scattering response and total thermal power but uses the incoming absorption defect for port-resolved emission. All three plain agents retain that physical closure. The hinted failure states the correct outgoing defect but sums the wrong array axis; it is an implementation failure, not a physical failure. Its author notes link the fixed-loss-rate repairs and full trial reviews.

## Screening and evidence

[Current source-matched status](results/hardening-status.json) lists every latest task result. The [final source and control audit](results/final-curation-audit.json) verifies all twelve retained tasks; an [independent paired audit](results/brownian-retained-provenance-audit.json) covers all sixty trials in the ten zero-score families. [All batches and paired checks](results/candidates.json) retain all controls, agent outcomes, source hashes, native setup checks, final artifacts and trajectory reviews. [The archive manifest](archives/screened/manifest.json) records the exact sources removed from the retained task directory. The complete original three-model comparison and first fourteen-candidate screen remain in [the earlier dashboard](archives/pre-curation-docs/DASHBOARD.md).

Scientific validity and measured difficulty are separate. Each candidate must first pass an independent physical reference, parameter-identification/noise checks, and actual oracle/shortcut controls. An unfinished fit or numerical failure is never counted as the intended physical failure. A candidate that passes all unhinted trials is archived even when its physics is valid.

An [instruction audit](results/instruction-audit.json) found no direct disclosure of the missing physical corrections, but identified a soft fit-only scope ambiguity in the starter framing and a calibration-uncertainty side channel in the cavity task. A [proposed neutral instruction patch](results/instruction-neutrality-proposed.patch) explicitly permits replacing implementation code without naming the missing physics. Evaluated task files are unchanged. The current results do not isolate physical misunderstanding from deference to the supplied code; revised instructions require new batches.

These tasks were developed against Luna using small three-trial batches. The reported counts describe those runs, not an unbiased estimate of model accuracy or a guarantee of 3/3 failures on future runs. No held-out confirmation claim is made.
