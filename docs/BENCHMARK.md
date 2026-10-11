# Benchmark interface and validation

The current publication has thirteen tasks selected by their original qualification batches; see the [qualified roster](../README.md) and the [result manifest](../results/qualified-tasks.json), which is the canonical record of the roster, the qualified source hashes and the trial IDs. Four tasks have since had [post-qualification maintenance](#post-qualification-maintenance) without new trials. Entropy anomaly r7, rotating reservoir r8 and superconducting heat r6 replace their earlier canonical revisions. Swirling heat tails r1, hard-rod current r2 and finite-ion cloud r3 were selected in the original expansion. For current apparatus definitions and APIs, use each task's environment/README.md; its AUTHOR.md records derivations and validation. Earlier discussions below describe historical revisions where indicated.

Hierarchical glass uses Model.fit(records), predict_calibration(records, width=None), and predict(experiments, width=None). Prediction records supply beta, replicas and observable; leaf_partition additionally supplies positive block sizes summing to replicas. The 58 graded outputs form six groups, each with relative Euclidean error below4%; parameter error must be below3%, and calibration chi-square per record below1.5. It keeps600-second agent and60-second verifier limits. Its validator reads checked-in data and requires --task, --baseline and --output paths.

The target was **ten tasks at 0/3 each** in unhinted Luna-high trials, with predominantly physical-model failures. All ten now qualify. Historical status: when the goal status record was last updated (2026-10-06), 5 of 10 qualified, namely finite-layer-phoresis, insulating-mhd-duct, magnetic-tracer, reaction-diffusion and single-file-memory, and the other five required revision or replacement. That [goal status record](../results/zero-three-goal-status.json) keeps its source checks and the stricter protocol but is superseded by the result manifest; the completed batches below remain historical evidence.

Historically, the [existing-family hardening plan](../results/existing-family-hardening-plan.json) froze the five sources that qualified at that time and prioritized rotating reservoir, thermoelastic rod and hydrodynamic heating, followed by entropy anomaly and superconducting heat. Entropy anomaly r7, rotating reservoir r8 and superconducting heat r6 later qualified. Thermoelastic rod r4 passed 3/3 and hydrodynamic heating r4 passed 2/3, so neither qualified. Ring-current fluctuations r1 and hierarchical glass overlaps r2 completed the ten. The plan's rules still apply to new revisions. Review successful as well as failed trajectories before revising the physics. A replacement requires a documented explanation of why further hardening of the existing family is unlikely to help. Every revision still needs scientific validation, frozen inputs/grading and exactly three fresh unhinted trials; unfinished candidates and single-run failures do not count toward the ten.

PhysLiteBench tests whether a coding agent revises a physical approximation that fits calibration but fails elsewhere in the stated apparatus. An earlier roster was selected through a conditional screen with one reviewed physical failure per task (0/1). Historically, a separate fresh three-trial confirmation of that roster ([retained-three r1](../results/retained-three-r1-results.json), 2026-10-04) found 6 of its ten tasks at or below one pass. Phoresis revision 2 and single-file revision 1 then replaced two of those tasks, each evaluated at 0/3 completed trials with separate infrastructure accounting. [The dashboard](../DASHBOARD.md) reports the current roster; archived scores stay with their archived sources.

The new expansion targets three additional tasks under the same 0/3 criterion, with `gpt-5.6-luna` High explicitly authorized after exact `gpt-6-luna` access failed. Swirling heat tails r1 is the first qualified addition. Its `Model.fit(records)` stores `mobility`; `predict(experiments)` reads stiffness, drive_ratio and relative_heat and returns one rate per input. Four prediction groups use relative Euclidean error below 4%, parameter error below 3%, and reduced calibration chi-square below 1.5. The validator is `scripts/validate_swirling_heat_tails.py`; the three trials used disabled web search and standard 600/60-second limits. [Results and response-only repairs](../results/expansion-swirling-heat-tails-r1-results.json) preserve all outcomes. [Nineteen excluded revisions](../archives/physics-expansion-20261011/README.md) retain their sources and evidence.

Hard-rod current r2 uses `Model.fit(records)` to store `speed`, then `predict(experiments)` for mean current, variance growth, or a scaled cumulant generating function. Inputs are density, mean length, length contrast, counting field, and statistic. It uses the same four-group 4% prediction gate, 3% parameter gate, and 1.5 reduced chi-square gate. Run `scripts/validate_hard_rod_current.py` for the independent flow/composition contraction, finite-time rank calculation, and noise checks. [Results and response-only repairs](../results/expansion-hard-rod-current-r2-results.json) cover all three trials. Revision 1 is preserved in the expansion archive.

Finite-ion cloud r3 uses `Model.fit(records)` to store `coupling`, then `predict(experiments)` for finite-cell osmotic pressure. Inputs are ion_count (1, 2 or 3), temperature_multiplier, radius_ratio and disk_radius. One-ion and two-ion calibration identifies the shared coupling; three-ion predictions test consistency of the relative angles. It uses the same 4% prediction, 3% parameter and 1.5 reduced chi-square gates. `scripts/validate_finite_ion_cloud.py` checks positive microscopic Gibbs quadrature, determinant anchors, domain convergence and 256 noise draws. [Results and response-only repairs](../results/expansion-finite-ion-cloud-r3-results.json) cover all three physical failures. Revisions 1 and 2 each passed 3/3 and remain archived. This completes the three requested additions.

The user requested a fresh confirmation after development selection. It returned 1/3 passes for swirling heat tails, 0/3 for hard-rod current and 0/3 for finite-ion cloud. All nine used new Docker environments and Codex sessions, unchanged task files and grading, no hint, and the same pinned model settings. The fixed nine-trial plan was saved before outcomes; the passing heat solution is retained. Original qualification counts are historical, and this confirmation does not establish 0/3 for all three tasks. Seven failures are clean physical-model errors; one ion failure combines the angular closure with an incorrect radial exponent. Separate interventions show that either remaining error alone fails the hidden groups. Only hard-rod current repeats zero passes with three clean physical failures. [Plan](../results/expansion-confirmation-20261011-1709-plan.json), [results](../results/expansion-confirmation-20261011-1709-results.json) and [independent review](../results/expansion-confirmation-20261011-1709-review.md) preserve provenance.

## Agent task

Each task supplies an `instruction.md`, a public Docker environment, calibration data, a Python predictor and public tests. Its README defines the apparatus, allowed preparations, observables, units, parameter ranges and API. The instruction explicitly permits modifying or replacing every prediction function and helper while preserving the documented API. Private cases use the same stated apparatus with preparations outside the calibration subset.

The ordinary image contains only `environment/`. Private `tests/`, `solution/`, `AUTHOR.md` and `hint.md` stay outside it. The runner adds the hint only to the instruction when `--hint` is requested. The frozen environment, grading and configuration must remain byte-identical between paired ordinary and hinted batches.

Each task uses a model with `fit` returning itself and a prediction method documented in its public README. Inputs and shapes vary by apparatus. Follow each task's interface rather than assuming a shared prediction signature. Public tests check interface and calibration; the private verifier owns its own calibration copy and physical reference.

## Reward and diagnosis

A trial receives reward 1 only if every required check passes. The verifier tests interface, finite outputs, calibration goodness of fit, parameter recovery and multiple physical predictions. Most new tasks require reduced calibration chi-square below 1.5 and relative parameter error below 3%; hidden tolerances and normalization are task-specific. Their author notes explain numerical convergence and separation from measurement noise.

A failed unfinished fit is not the intended shortcut control. Every candidate includes a completed calibration fit for both a correct oracle and the supplied physical approximation. Both must recover the parameter and fit calibration; only the wrong physical approximation should fail the distinguishing experiments.

Review final code, public messages, commands and verifier metrics before classifying an agent failure. Separate incorrect physical assumptions from algebra, discretization, coding, optimizer and infrastructure failures. Diagnostic repairs run in memory at the retained fitted parameter; they never modify the original submission or its recorded reward.

## Reproduction

Use Docker and uv. From the repository root:

```bash
python3 scripts/run_candidate_matrix.py magnetic-tracer \
  --conditions oracle shortcut plain --workers 1 --label fresh-three
```

The matrix runner checks the oracle and completed shortcut, then runs exactly three unhinted Luna high trials. The oracle reward should be 1 and the completed shortcut reward 0, with only hidden prediction tests failing. The runner checks task directories, requested hints and shortcut files before launching anything; ring-current fluctuations has no `hint.md`, so omit the `hint` condition for it. Infrastructure errors stop that task for review. New job names must be unique; never overwrite an existing result. Staged candidates can use `--task-root staging/REVISION/tasks --control-root staging/REVISION/scripts`.

The expansion batches explicitly disabled web search using Harbor's agent
setting `web_search=disabled`. This disables the built-in web tool, not shell
network access; the archived annihilating-gas trials retrieved primary
literature through shell HTTP requests. These are not offline runs.
The working-tree runner already exposed this
setting; its unrelated changes are not part of the expansion. The same pinned
agent setup can be invoked directly, with `CODEX_AUTH_JSON_PATH` pointing to
an existing authentication file outside the repository:

```bash
uvx --python 3.13 harbor@0.21.0 run -p tasks/hard-rod-current \
  -a codex --env docker -n 3 -k 3 -m gpt-5.6-luna \
  --ak version=0.154.0 --ak reasoning_effort=high --ak web_search=disabled \
  --ae "CODEX_AUTH_JSON_PATH=$CODEX_AUTH_JSON_PATH" \
  --job-name "hard-rod-current-$(date -u +%Y%m%d-%H%M%S)"
```

The historical conditional screen used `run_neutrality_screen.py`: an initial pass triggered two more trials, while an initial failure stopped at 0/1. The initial outcome was never discarded. The fresh confirmation used `run_candidate_matrix.py --conditions plain` for all ten retained tasks; it did not use conditional stopping or hints.

An entirely unstarted initial or follow-up batch in the conditional runner may be replaced with `--retry-unstarted-job JOB` after review confirms a network setup failure, no agent execution, no verifier result and no native session. The runner verifies that source, grading, instruction and configuration still match the frozen task and preserves the failed attempts. A replacement initial trial follows the same conditional rule; replacement follow-ups run only the missing trials. Reports distinguish total attempts from actual model trials. A failure after the agent starts is never excluded by this rule.

The later fixed-three evaluation has a separately documented [host-wide Docker outage recovery](../results/docker-outage-20261004-080748.json). Exactly five unfinished, unscored attempts were replaced under a plan recorded before any replacement outcome. The existing scored pass remained, sources and grading stayed frozen, and every interruption is reported separately. Phoresis revision 2 therefore has 0/3 completed scored trials across six total attempts; its three infrastructure attempts are not physics failures. This event-specific recovery does not authorize replacing completed scores or resampling an unchanged task.

Single-file revision 1 has zero passes in three completed trials across four attempts. One original NVM TLS setup failure had no model execution or verifier; an explicit plan replaced exactly that unstarted attempt. Both original completed scores were retained. The replacement was declared after those two outcomes were terminal and before the replacement began. See the [four-attempt audit](../results/zero-three-singlefile-r1-physics-promotion-audit.json).

Pinned setup: Python 3.13, Harbor 0.21.0, Codex CLI 0.154.0, NumPy 2.3.3, SciPy 1.16.3 and pytest 8.4.2 in new task images. The agent/verifier timeouts are 600/60 seconds. Authentication remains outside the repository.

The optional `--agent-setup-timeout-multiplier` extends tool installation time only. It leaves the default behavior, task files, model settings and 600/60-second solving/grading limits unchanged; matrix controls do not receive it. After six unstarted adiabatic setup failures, the [setup-budget plan](../results/zero-three-adiabatic-coherent-setup-budget-plan.json) predeclares three replacements and the first coherent-array batch with multiplier 3 (1080 seconds for setup). All prior null outcomes remain recorded, and no completed score is replaced.

Those batches are complete: [adiabatic r1](../results/zero-three-adiabatic-r1-results.json) passed 3/3, including one saved passing model with a scored 600-second execution timeout; [coherent-array r1](../results/zero-three-coherent-r1-results.json) passed 3/3 without exceptions. Both are archived. The adiabatic timeout remains a scored outcome and was not replaced.

Scientific validators run independently of agent trials:

```bash
uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 \
  --with pytest==8.4.2 python scripts/validate_magnetic_tracer.py
```

Check each task's author notes for its exact command and report. Ordinary validation reads fixed data; `--generate` intentionally replaces both public and private calibration copies. New candidate validators use 256 noisy calibration fits. Some also evaluate every hidden case on every draw; others check hidden sensitivity at the observed parameter extrema. These are different checks and are labeled as such in the reports.

Independent references use a different derivation or representation from the oracle: examples include constrained pair-friction elimination, conservative heat-flux balance, entropy/strain constitutive systems, finite-volume current closure and Fourier-moment evolution. Check relevant conservation laws, stable/positive limits and numerical refinement as well as prediction agreement.

## Evidence and selection

The runner preserves a frozen task snapshot, native sessions, trajectory, final `/app` artifacts, source diffs, verifier metrics and timing under ignored `jobs/`. These raw jobs are not published. On the machine that holds them, `python3 scripts/collect_trial_evidence.py` copies only each trial's final `model.py`, `metrics.json`, `reward.txt` and `test-stdout.txt` into `results/<task>-r<revision>-evidence/`, adds `model.diff` (the submitted `model.py` against the job's frozen starter), with a manifest that hashes the copies and compares each job's frozen task with the qualified source hashes. By default it covers entropy anomaly r7, rotating reservoir r8, superconducting heat r6 and hierarchical glass overlaps r2, whose trials have no published review or evidence files yet; `--task` selects other qualified tasks. [The full candidate ledger](../results/candidates.json) summarizes all indexed batches and merges separate per-trial reviews. Regenerate the ledger with:

```bash
python3 scripts/summarize_candidates.py
```

The historical conditional-screen generator is `scripts/summarize_neutrality.py`; its saved report describes the earlier roster. The canonical roster, qualified source hashes, trial IDs and post-qualification changes are recorded in `results/qualified-tasks.json`. `results/zero-three-goal-status.json` is a superseded historical status record.

Selection includes every neutral-instruction batch matching the current environment, grading, instruction and configuration hashes. For the four tasks with [post-qualification maintenance](#post-qualification-maintenance), compare the qualifying batches with the qualified `source_sha256` in the result manifest, not with the current files. Do not rerun unchanged candidates until a favorable zero appears. A scientific revision is evaluated separately and its predecessor preserved. The original results and paired hint audits are historical and preserved in the pre-neutral archive. A current 0/1 failure is not interchangeable with an archived 0/3.

The user authorized archiving tasks above the 1/3 cutoff. [Screened tasks](../archives/screened/README.md) remain runnable by passing their archive path to `run_science.py`; their source hashes and evaluation IDs are in the archive manifest. Their individual validators and baselines moved with them. The shared first-wave validator resolves retained and archived paths.

One or three conditional trials measure a development outcome, not a population success probability. Selection against Luna and reuse of calibration observations are disclosed. No held-out generalization or future 3/3-failure guarantee is claimed.

For a new task's files, assumptions, review and validation requirements, use [ADDING_TASKS.md](ADDING_TASKS.md). Prior versions of this guide are preserved in the documentation archive.

## Post-qualification maintenance

On 2026-10-10, after qualification, the changes below were made without new model trials. The [result manifest](../results/qualified-tasks.json) keeps each task's qualified `source_sha256` and lists every changed task file, with its qualified and current hash, under `post_qualification_changes`.

- **Frozen hidden truth.** Entropy anomaly, insulating MHD duct, rotating reservoir and superconducting heat recomputed their independent reference on every verifier run. `scripts/freeze_hidden_truth.py` now writes `tests/data/hidden_truth.json` with each hidden group's inputs and `reference.predict(hidden_inputs, TRUE_PARAMETER)`. `tests/test_hidden.py` checks the stored inputs against `reference.hidden_inputs()`, allowing last-bit differences between CPUs (the MHD inputs come from `np.geomspace`, which differs by one unit in the last place on AVX-512 processors), and grades against the stored values. In the local Harbor-verifier emulator, oracle verifier time fell from 40.4 to 1.3 s (entropy anomaly), 34.1 to 3.9 s (MHD duct), 27 to 4.9 s (rotating reservoir) and 58.6 to 7.2 s (superconducting heat), against the 60-second limit. Old and new tests were run on the oracle, the completed shortcut, the starter, archived real submissions where available, and synthetic submissions just inside and just outside the thresholds; they gave identical rewards and per-test outcomes and bit-identical metrics. Truth recomputed inside the verifier image matched the stored values bit for bit. The change is grading-equivalent for submissions that finish within the old verifier budget and leave NumPy's global error settings alone. A submission that timed out only because of the live reference computation, or whose global `np.seterr` made that computation fail, could now pass. No Luna submission for these revisions is published, so none was regraded. After running the evidence collector, regrade each qualifying trial under the current tests, for example `python3 scripts/run_science.py tasks/entropy-anomaly --agent oracle --trials 1 --solution-model results/entropy-anomaly-r7-evidence/entropy-anomaly__AbnYJDw/model.py`.
- **Superconducting-heat starter.** `stationary()` in the public starter, the completed shortcut and the oracle had the default `conductance=45.`, which is the true value. The default was removed; every caller already passed the conductance, and predictions are bit-identical. The change is agent-visible. It removes only the true value, so it is not expected to make the task easier, but the task was not re-evaluated. `tests/metadata.json` now says revision 6 instead of 5; no test reads that field.
- **Validators.** The entropy-anomaly, rotating-reservoir and superconducting-heat validators read inputs that are not published on this branch: `archives/entropy-anomaly-r6`, `archives/rotating-reservoir-r7`, `results/prototypes/rotating-reservoir-r8-root/` and `archives/superconducting-heat-r4`. They now skip only the checks that need those inputs, print a warning and record each one under `skipped_checks` in their report; all other checks run as before. The validators of all four changed tasks also recompute the frozen truth and fail if it drifts from `tests/reference.py` by more than 1e-7 of the group RMS. All four passed with default arguments. The rotating-reservoir and superconducting-heat runs took about 17 and 47 minutes on one CPU.
- **Matrix runner.** `scripts/run_candidate_matrix.py` creates `jobs/` when it is missing and uses `scripts/reaction_baseline.py` as the reaction-diffusion shortcut. It checks task directories, requested `hint.md` files and shortcut files before launching anything. The shortcut control must now fail only hidden prediction tests, read from the verifier's `test-stdout.txt`; the old check read metric keys that hierarchical glass overlaps and reaction-diffusion do not write.
- **Evidence collector.** `scripts/collect_trial_evidence.py` (see [Evidence and selection](#evidence-and-selection)) has been added but not yet run on the raw jobs. It does not copy the raw `final.diff`, which covers every text file the agent left in `/app`; it writes a `model.py`-only diff instead.
