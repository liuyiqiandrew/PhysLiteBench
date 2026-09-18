# Latest evaluation dashboard

For the physics and calibration loopholes behind the three tasks, see [TASKS.md](TASKS.md).

Updated 2026-09-17. **Only the newest completed three-trial batch for each task/model is included: 27 trials, nine batches.** Luna uses the fresh recheck; Terra and Sol use their latest comparison batches. Older results are excluded from these counts. Original raw artifacts are archived separately and are not included in this repository.

All models used **high reasoning effort**, Harbor 0.21.0, Codex CLI 0.154.0, and the same task files, data, and grading within each task. Agent/verifier limits were 600/60 seconds. The original audit checked native logs to confirm the requested models and effort. No new agent runs were made for this dashboard.

This repository includes the [per-trial statistics and spin diagnostic](results/latest.json) and a small [electrodiffusion endpoint diagnostic](results/electrodiffusion-endpoints.json). Full trajectories, command logs, final submissions, and batch audits are not bundled. The trajectory findings below summarize the original review of public messages, tool activity, and final code. Trial and batch IDs identify those archived records; they are not links to local files.

## Results

Cells show **passes out of three**.

| Task | Luna high | Terra high | Sol high |
|---|---:|---:|---:|
| Spin probes, r6 | 1/3 | 3/3 | 3/3 |
| Thermal piston, r9 | 0/3 | 1/3 | 3/3 |
| Electrodiffusion, r6 | 0/3 | 0/3 | 3/3 |
| **Total** | **1/9** | **4/9** | **9/9** |

All 27 submissions passed calibration and parameter recovery. Every failed submission failed only hidden predictions; there were **no timeouts or Harbor exceptions**. Trajectory review distinguishes the causes:

| Model | Pass | Physical-model failure | Mixed math diagnostic / model selection | Numerical implementation failure |
|---|---:|---:|---:|---:|
| Luna | 1 | 7 | 1 | 0 |
| Terra | 4 | 4 | 0 | 1 |
| Sol | 9 | 0 | 0 | 0 |

**Classification qualification:** Luna spin trial `vYnEEHB` submitted the wrong physical model, but its trajectory includes a mathematical error in the comparison used to select that model. Its raw failure stands; it is separated here from clean physical-model failures. Under a strict “no mathematical cause” criterion, the newest spin batch supplies **one clean physical failure, one mixed failure, and one pass**—so it does not establish two clean physical failures. Thermal and electrodiffusion each retain three clean Luna physical failures.

These are small batches on three tasks developed against Luna, not general model success rates.

## Interaction and resource statistics

Each row covers nine trials. Counts are totals; time is the median per trial.

| Model | Model rounds | Tool requests | Shell commands | Median agent time | Output tokens |
|---|---:|---:|---:|---:|---:|
| Luna | 111 | 102 | 104 | 123.4 s | 58,999 |
| Terra | 71 | 62 | 50 | 102.2 s | 43,966 |
| Sol | 89 | 80 | 64 | 159.1 s | 69,742 |

- **Model rounds / interactions:** sum of Harbor trajectory `llm_call_count` over agent steps, cross-checked against native token-usage records. These are model turns, not human conversations; no trial received a human follow-up.
- **Tool requests:** unique top-level tool-call IDs in the model transcript, cross-checked against native logs. An orchestration request can contain multiple shell commands or patches, so shell counts need not match tool counts.
- **Shell commands:** completed `command_execution` items, counted once by item ID. Setup and verifier commands are excluded.
- **Agent time:** Harbor's agent-execution interval, excluding container setup and grading. Output tokens are Harbor's recorded completion-token totals; reasoning tokens are not added again.

Luna used **111 model rounds versus Sol's 89**, yet passed one trial versus nine. More interaction was not sufficient to correct the physics in these runs. Nonzero shell exits occurred **11 / 12 / 8** times for Luna / Terra / Sol; these included fit checks, missing paths, and Git commands outside a repository. They are intermediate events, not trial verdicts.

## Interesting trajectories

**1. Recognizing the right issue did not prevent a wrong conclusion.** Luna spin `vYnEEHB` explicitly checked whether the probes' shared noise required correlation. Its temporary shared-noise candidate used incorrect preparation/readout rotations and reported reduced χ² **997**, versus **0.9924** for the supplied factorized model. Its next public update accepted the factorized model. A post-hoc recomputation corrects only those rotations at the retained fitted gamma and obtains **0.9924** for the shared-noise model too. This is a mixed diagnostic/model-selection failure, not simple forgetting. The recomputation is recorded in the [statistics data](results/latest.json), without changing the submission or running another agent.

**2. Sol changed its initial physical assumption.** Thermal `3k2BgxE` initially endorsed the supplied constant-pressure model, then noticed that calibration keeps the temperature sum fixed and cannot validate general heating. It corrected the piston dynamics and checked a new `[320,300] K` preparation against the coupled first law; derivative disagreement was **2.38×10⁻⁸ K/s**. Its final hidden error was below **0.065%**. The public trajectory records the correction directly.

**3. The longest Luna trajectory repaired fitting but left the physics wrong.** Thermal `QDbiKyo` used **26 model rounds, 25 tool requests, 20 shell commands, four patch events, 308 s, and 15,008 output tokens**. It fixed an intermediate golden-section-search bug and passed calibration, but kept the original constant-pressure predictor. Hidden errors remained **13.25–27.20%**. Successful numerical self-correction did not become a physical-model audit.

**4. A conservation check can validate the wrong discretization.** Terra electrodiffusion `CMRj2xx` included the correct shared electric field and tested a mixture, but checked the unweighted mean over grid nodes. Its incorrect full-width endpoint cells conserve that quantity, while the physical spatial mass requires half-weight endpoints. The check passed; hidden error reached **9.004%**. Correcting only endpoint volumes at fixed diffusivity reduced it to **0.1192%** in the retained [diagnostic](results/electrodiffusion-endpoints.json). All three Sol trials checked correctly weighted mass; `KWp96WN` also compared the coupled solver with the binary-salt limit and tested a mixed profile with two spatial harmonics.

**5. Several passing agents checked behavior outside calibration.** Terra spin `UAEzApH` compared its joint probability with one million shared-phase samples. Sol `yZc2yzY` used two million samples, agreeing within **0.000195**. In contrast, all three latest Luna electrodiffusion agents fitted the pure-salt data and left the independent-salt predictor unchanged. Their public tests passed, but they completely missed the initially uniform ion's induced redistribution in the hidden mixtures.

## Per-trial statistics

**P** = pass; **F** = physical-model failure; **M** = mixed mathematical diagnostic/model-selection failure; **N** = numerical implementation failure. Trial IDs identify the separately archived runs. All table values and classifications are included in the bundled [statistics](results/latest.json).

### Spin

| Model / trial | Result | Model rounds | Tool requests | Shell commands | Agent seconds | Output tokens |
|---|---:|---:|---:|---:|---:|---:|
| Luna / `gw836QL` | P | 16 | 15 | 15 | 211.7 | 9,150 |
| Luna / `vYnEEHB` | M | 9 | 8 | 11 | 123.4 | 5,378 |
| Luna / `vkqNyWU` | F | 11 | 10 | 17 | 114.8 | 5,091 |
| Terra / `B7hhTin` | P | 9 | 8 | 6 | 114.5 | 5,114 |
| Terra / `UAEzApH` | P | 10 | 9 | 7 | 133.7 | 6,354 |
| Terra / `qdAqWSC` | P | 8 | 7 | 6 | 102.2 | 4,503 |
| Sol / `4H7oWE6` | P | 7 | 6 | 5 | 99.5 | 4,416 |
| Sol / `Yq6wDKj` | P | 10 | 9 | 7 | 130.7 | 5,118 |
| Sol / `yZc2yzY` | P | 11 | 10 | 7 | 200.2 | 9,479 |

### Thermal

| Model / trial | Result | Model rounds | Tool requests | Shell commands | Agent seconds | Output tokens |
|---|---:|---:|---:|---:|---:|---:|
| Luna / `4wpYNZt` | F | 13 | 12 | 11 | 184.7 | 8,190 |
| Luna / `QDbiKyo` | F | 26 | 25 | 20 | 308.3 | 15,008 |
| Luna / `XiM8S5L` | F | 7 | 6 | 5 | 98.7 | 4,048 |
| Terra / `8ZeNNqP` | F | 8 | 7 | 6 | 95.8 | 4,360 |
| Terra / `DPmhFWA` | F | 7 | 6 | 5 | 83.1 | 3,744 |
| Terra / `Hq9qXxN` | P | 8 | 7 | 6 | 125.0 | 5,539 |
| Sol / `3k2BgxE` | P | 11 | 10 | 8 | 159.1 | 7,620 |
| Sol / `JfrrvPr` | P | 11 | 10 | 9 | 141.8 | 5,812 |
| Sol / `xgEJXCg` | P | 9 | 8 | 7 | 129.1 | 5,706 |

### Electrodiffusion

| Model / trial | Result | Model rounds | Tool requests | Shell commands | Agent seconds | Output tokens |
|---|---:|---:|---:|---:|---:|---:|
| Luna / `DQSz4mt` | F | 11 | 10 | 11 | 126.2 | 4,765 |
| Luna / `igafNCt` | F | 10 | 9 | 8 | 98.2 | 4,337 |
| Luna / `rWQHHqo` | F | 8 | 7 | 6 | 85.4 | 3,032 |
| Terra / `CMRj2xx` | N | 9 | 8 | 6 | 154.8 | 7,149 |
| Terra / `ZWrTrQ4` | F | 6 | 5 | 4 | 92.8 | 3,396 |
| Terra / `cyEhdFf` | F | 6 | 5 | 4 | 86.7 | 3,807 |
| Sol / `KWp96WN` | P | 13 | 12 | 9 | 350.0 | 14,351 |
| Sol / `SNjmsdi` | P | 9 | 8 | 6 | 269.2 | 11,175 |
| Sol / `bkh5WFN` | P | 8 | 7 | 6 | 166.0 | 6,065 |

## Selected batches and evidence

Only these nine batches contribute to this dashboard. These are archive identifiers, not bundled directories. The statistics preserve both the original final-artifact classification and the reviewed trajectory classification, including the mixed spin failure.

| Task | Latest Luna | Latest Terra | Latest Sol |
|---|---|---|---|
| Spin | `qubit-r6-luna-high-recheck` | `qubit-r6-terra-high` | `qubit-r6-sol-high` |
| Thermal | `thermal-v9-luna-high-recheck` | `thermal-v9-terra-high` | `thermal-v9-sol-high` |
| Electrodiffusion | `reaction-v6-luna-high-recheck` | `reaction-v6-terra-high` | `reaction-v6-sol-high` |

[Machine-readable statistics and diagnostic](results/latest.json). This summary supports recomputing the aggregate tables from the 27 per-trial rows. Independently checking interaction counts or trajectory interpretations requires the original raw traces, which are archived separately. When updating, replace each task/model cell with its newest completed batch; do not accumulate earlier batches into these totals.
