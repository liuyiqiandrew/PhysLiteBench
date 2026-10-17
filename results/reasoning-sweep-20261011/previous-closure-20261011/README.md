# Reasoning-effort comparison

Updated 2026-10-12T00:36:43.836949+00:00. Each complete point covers 13 frozen tasks and 39 attempts. Low, High and Extra High use the same Codex 0.160.0 client and 600/60-second agent/verifier limits.

| Model | Effort | Status | Recorded attempts | Completed passes | Passing checkpoints | Errors | Mean estimated API cost |
|---|---|---|---:|---:|---:|---:|---:|
| GPT-6 Luna | Low | Complete | 39/39 | 4 | 4 | 0 | $0.00402 |
| GPT-6 Luna | High | Complete | 39/39 | 1 | 1 | 0 | $0.00517 |
| GPT-6 Luna | Extra High | Complete | 39/39 | 12 | 13 | 4 | $0.01261 |
| GPT-6 Sol | Low | Complete | 39/39 | 37 | 37 | 0 | $0.07733 |
| GPT-6 Sol | High | Complete | 39/39 | 39 | 39 | 0 | $0.19754 |
| GPT-6 Sol | Extra High | Complete | 39/39 | 39 | 39 | 0 | $0.23061 |
| GPT-5.6 Terra | Low | Complete | 39/39 | 4 | 4 | 0 | $0.11071 |
| GPT-5.6 Terra | High | Complete | 39/39 | 9 | 9 | 0 | $0.19419 |
| GPT-5.6 Terra | Extra High | Complete | 39/39 | 19 | 20 | 4 | $0.31963 |
| GPT-6 Astra | Low | Complete | 39/39 | 39 | 39 | 0 | $0.44501 |
| GPT-6 Astra | High | Complete | 39/39 | 39 | 39 | 0 | $0.71323 |
| GPT-6 Astra | Extra High | Not run | 0/39 | — | — | — | — |
| DeepSeek V4.1 Flash | High | Complete | 39/39 | 1 | 1 | 26 | $0.19126 |
| DeepSeek V4.1 Flash | Low | Partial | 3/39 | 0 | 0 | 2 | — |

The sweep is closed with 12 complete model–effort points (39 attempts each), plus a three-attempt DeepSeek Low pilot. Reported Codex account usage was 46% at closure, below the user’s 50% ceiling; new launches were guarded at 47%. Astra Extra High was deferred because the remaining launch allowance was too small to assume a full 39-attempt panel would fit. Prime calls are stopped at a conservative cumulative $8.30 of the $10 cap, with no pending requests. No benchmark jobs remain active.

[Final run status and budget decision](run-status.json).

Only complete panels with reconciled costs enter the frontier. Missing attempts are not treated as failures. Execution errors remain recorded and are reported separately. The main score counts a pass only when the agent finishes without an exception; passing saved checkpoints after a timeout are also listed, without changing that predefined score. Low and Extra High are fresh runs; High reuses 30 matching observations per model from the original ten-task comparison and adds nine observations on the new tasks.

Costs are API-equivalent token estimates, not Pro subscription charges. Task selection against GPT-5.6 Luna and reuse of the original High observations are disclosed. Four historical Terra High trials were affected by host sleep; see the prior comparison timing audit. The selected GPT-5.6 Luna baseline remains in the old ten-task plot and is excluded from this thirteen-task frontier.

[Frozen plan](plan.json), [results](results.json), [CSV](summary.csv), [pricing](pricing.json), [evidence audit](evidence-audit.json), [task-source audit](task-source-audit.json), [final audit](final-audit.json), and [plot reproduction](REPRODUCE.md). [Per-task scores](per-task-scores.csv), [trial outcomes](trial-outcomes.csv), and [model inventory](model-inventory.json) provide the detailed counts.

Prime Low is partial (3/39 attempts) and excluded from the frontier. Its pilot cost $0.65; cumulative conservative Prime spending is $8.30 against the $10 cap. Further Prime calls were stopped because projected completion cost exceeds remaining funds. See [budget decision](prime/budget-decision.json).

Provider errors are execution failures. Token estimates may omit usage from interrupted responses; inspect the [saved reviews](reviews/) before treating small cost differences as meaningful.

![Reasoning-effort frontier](../../output/reasoning-sweep-20261011/pareto-reasoning.png)

[PDF](../../output/reasoning-sweep-20261011/pareto-reasoning.pdf) · [SVG](../../output/reasoning-sweep-20261011/pareto-reasoning.svg)
