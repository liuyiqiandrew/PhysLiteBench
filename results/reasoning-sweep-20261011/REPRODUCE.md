# Reproduce the figure

The portable plotting script reads the saved `results.json` summary and writes PNG, PDF and SVG figures under `output/reasoning-sweep-20261011/`. It also rebuilds this comparison’s README. It does not call any model API or launch benchmark attempts.

From the repository root, with uv installed:

```bash
uv run --python 3.13 --with matplotlib==3.10.7 --with numpy==2.5.4 python results/reasoning-sweep-20261011/tooling/plot.py
```

The benchmark source commit, per-file hashes, runtime versions, task order and settings are in `plan.json`. The original frozen schedule is retained; the operational priority amendment and concurrency guard are in `scheduling.json`. The copied task roster retains its historical ten-task comparison reference; `results.json` is the new thirteen-task comparison.

`evidence/` preserves each new submission, its diff, verifier output and metrics. `runs/` records every batch and original trial ID. `reviews/` records reviewed execution failures without replacing their outcomes. Native trajectories and full frozen runtime copies remain in the local ignored `jobs/` directories; they are needed to repeat the detailed collection audit.

OpenAI costs use recorded tokens and the rates in `pricing.json`, with the qualifications stated in the report. Fresh DeepSeek costs use the catalog snapshot and cache-aware rates in `prime/pricing-v2.json`; the older Prime rate record remains unchanged for historical results. Prime batch ledgers reconcile requests completed after trial timeouts, and missing usage retains a conservative reserve. The original spending decision is preserved in `prime/budget-decision.json`; the continuation authorization removes that spending cap for the finite four-effort schedule.

API credentials and runtime authentication configuration are excluded from these artifacts.

`per-task-scores.csv` and `trial-outcomes.csv` expose the score counts without assigning unattributed post-timeout Prime costs to individual trials. `model-inventory.json` distinguishes tested settings from planned settings and the historical GPT-5.6 Luna baseline. Regenerate these tables with `python results/reasoning-sweep-20261011/tooling/export_tables.py`.

The October 12 continuation is authorized in `continuation-20261012.json`. Its fresh DeepSeek schedule is frozen in `prime/transport-v2-plan.json`; `prime/transport-v2-amendment.json` documents the reproduced network-sidecar header timeout and the correction. The primary comparison includes only this new DeepSeek cohort. All 48 older DeepSeek observations remain in `prime/legacy-transport-results.json` and their original run folders. New Prime batch ledgers record both client and provider reasoning settings, use catalog cache-read rates, and retain conservative cost reserves where usage is missing. OpenAI High reuse is unchanged.

Automated source and network keyword flags are retained. `source-flag-reviews.json` and `network-flag-reviews.json` document inspected false positives using exact model or trajectory hashes; only hash-matched reviews can clear those flags in the final audit.

`unstarted-setup-recovery-20261012.json` records six setup failures during host sleep and the two replacement batches declared before fresh outcomes. The originals have no agent execution, verifier result or native session; all six raw results and run records remain under their original jobs and `unstarted-setup-20261012/`. Primary panels count started model attempts; these setup failures are reported separately.

`timing-flag-reviews.json` preserves exact hashes and power-log evidence for three started DeepSeek High attempts affected by closed-lid sleep. Their outcomes remain in the primary data, with no replacement. Run `python results/reasoning-sweep-20261011/tooling/audit.py --checkpoint` to validate a settled partial dataset; omit `--checkpoint` only after all 624 primary observations are complete. The auditor requires exact reviewed evidence for timing flags and still rejects unreviewed outliers.

The current delivery is budget-limited: 15 complete points and 588 primary observations. DeepSeek None has 3/39 observations and is excluded from the figure. Its remaining 36 attempts are unrun. `checkpoint-audit.json` validates the recorded snapshot; `final-audit.json` explicitly records that the full 624-observation plan is incomplete. `account-usage-budget-20261012.json` preserves the conservative interpretation, window transition and stop decision.
