# Finite ion cloud, revision 3: independent promotion review

**Approve promotion on the stated criterion: 0/3 passes and three genuine physical failures.** This describes the predeclared frozen batch of gpt-5.6-luna at high reasoning effort, not a universal failure-rate estimate. No task source was changed and no model trial was rerun for this review.

I read all three final models and diffs, the full emitted trajectory messages and tool commands, their observations, native session/configuration/results, and verifier evidence for `jobs/finite-ion-cloud-expansion-r3-plain-20261011-1030`. The apparatus, completed controls and independent reference were approved in the [source review](expansion-finite-ion-cloud-r3-physics-review.md).

All 91 provenance comparisons passed: the 17 planned source files against current and frozen copies; the prior review and its 20 source/report hashes; all 12 preserved per-trial evidence files against their recorded hashes and original artifacts; the three trajectory hashes; and each trial's public README, test and calibration against frozen inputs. Each emitted instruction equals the frozen neutral instruction. All three native task checksums are `7db806267e935057f6c1584346ac9502516564490115b3a5934c3105b85bd06d`. Native session records exactly match the nine tool calls per trajectory. The configuration specifies three concurrent plain trials, no added hint, gpt-5.6-luna, high effort, and Codex CLI 0.154.0. The job records three completed trials, zero errors and zero retries; each trial has no exception.

Built-in web search was disabled. Shell networking remained available; no HTTP retrieval or external physics lookup was observed in the emitted commands. This batch must not be described as having all network access disabled.

Every model passes public calibration, private calibration, parameter recovery and shared one-ion/two-ion anchors. Every model fails only `three_ion_loop`, `strong_loop` and `wide_cell`, with errors 10.9818%, 19.0037% and 29.5393% against the 4% gate. All retained the exact verifier coupling `1.1199334360355244`, with relative parameter error `5.9432111e-5` and reduced calibration chi-square `1.068747208068`.

The causal omission is the independence of the three equilibrium pair angles. For actual particle positions, theta13=theta12+theta23 modulo 2pi. The exact angular Gibbs factor therefore contains the linked two-angle average, or all matching Fourier harmonics around the triangle. The submitted product of three separate pair averages retains only the zero harmonic. It is the completed pair-cluster closure reviewed before evaluation, and is exact for calibration. AST comparison confirms that every final submitted response is unchanged from the starter; only fitting code was added.

| Trial | Explicit physical endorsement in emitted trajectory |
| --- | --- |
| `DabdUSJ` | Step 8 calls the supplied response consistent with the exact ordered-coordinate partition function and all pair interactions. |
| `USUxuDY` | Step 9 says the supplied integrals already match the canonical partition derivative, including interactions for N=2,3. |
| `uDNYxi6` | Step 7 accepts the existing finite-annulus formulas as the stated partition-function structure and proceeds to fitting and three-ion numerical checks. |

All three refine the same pair-product quadrature and verify finite outputs. None derives the linked-angle response and then loses it through coding or numerical error. Their incidental `git diff` failures outside a Git repository do not affect the submitted model. Scratch residual-normalization and convergence-index labels are also immaterial to the fit or prediction law. These are physical-model failures, not unfinished fits, timeouts, infrastructure failures or unconverged integrations.

For an independent response-only repair, I imported each exact preserved model, assigned its coupling from its verifier metrics without calling `fit`, reproduced its original group errors, and replaced only global `three_ion_coefficient` in memory with the frozen reference's positive linked-angle Gibbs quadrature at 40 radial and 160 angular nodes. I cleared the response cache and retained the same model instance and prediction path. This uses the independent microscopic integration, not the oracle's Fourier implementation. Calibration predictions change by exactly zero in every trial.

The repaired group errors are identical across trials: shared anchors `3.381563213030455e-5`, three-ion loop `1.5601194214845426e-4`, strong loop `2.242533274501389e-4`, and wide cell `3.341819976162791e-4`. Every group passes; the largest error is 0.03342%. The [independent repair evidence](expansion-finite-ion-cloud-r3-independent-repair.json) records the method, unchanged parameters and source hashes. This agrees with the root report's separate oracle response-only repair and isolates the missing loop correlation as the cause.

Revisions 1 and 2 remain separate archived 3/3 passing batches. Revision 3 adds three-ion preparations and exact two-ion calibration, with a control that restores all pair angular responses. Its promotion does not pool or resample the unchanged earlier revisions.

The [result report](expansion-finite-ion-cloud-r3-results.json) contains the verified per-trial model, metrics, reward, stdout and trajectory digests. Final review anchors:

```text
bc0969c11888565ba51298590bcee6b3c2d39278539d7e46facda81fcd4aef69  results/expansion-finite-ion-cloud-r3-plan.json
186bc046fedad8d433e8fcfd2bff87f252a8c7b55021fec2960f14d5154de7b9  results/expansion-finite-ion-cloud-r3-results.json
30ef5972504b6afed3d1574880be61100ccd60128afbbab6fe4b5a3fc8dba737  results/expansion-finite-ion-cloud-r3-physics-review.md
fbebccd2e0a9235e69b999e014a4b4cb77b4111a578bc6bea4403bfc65011efc  results/expansion-finite-ion-cloud-r3-independent-repair.json
cb5751e6ca040cbbcdc9235a90751a00f9846bf83e6dbcf47e8994311a5129ef  jobs/finite-ion-cloud-expansion-r3-plain-20261011-1030/config.json
99ec493e6730c9a6970b4e22f60fbcfd126cdfe8229c730804c3ed69cbd6b33e  jobs/finite-ion-cloud-expansion-r3-plain-20261011-1030/result.json
6f0e2515a4cd118533f6b224307f98019282f591bae0811991f6b7ac0f074d55  jobs/finite-ion-cloud-expansion-r3-plain-20261011-1030/frozen-task/tests/reference.py
e27da413affc2857e3399c43d6a425a92b5982ddc5c98f923928e79df6f46e33  jobs/finite-ion-cloud-expansion-r3-plain-20261011-1030/finite-ion-cloud__DabdUSJ/artifacts/app/model.py
358ccc4bc173bfe419f97a623782d925782b48de51d9a2877b233a2f6899c34f  jobs/finite-ion-cloud-expansion-r3-plain-20261011-1030/finite-ion-cloud__USUxuDY/artifacts/app/model.py
db5887022079c418d4246674b564dad8b7ed5de3a9d90a3f239a1d577559f4cd  jobs/finite-ion-cloud-expansion-r3-plain-20261011-1030/finite-ion-cloud__uDNYxi6/artifacts/app/model.py
```
