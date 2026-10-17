# Five-outcome breakdown: no hint, general hint and physics hint

This report covers **1,170 matrix attempts** on the same 13 task snapshots: 390 no-hint, 390 general-hint and 390 physics-hint attempts. Each model/effort/hint row contains 39 attempts (13 tasks × 3 independent repeats). It uses the strict, trajectory-reviewed definition of physics failure requested for this breakdown.

## Category definitions

| Outcome | Rule |
|---|---|
| Passed | Verifier reward 1, with no execution exception. |
| Only succeeded observed prediction (physics fail) | Completed reward-0 attempt; calibration and parameter-recovery checks both pass; at least one hidden prediction fails; trajectory and final-code review identify a **primary physical-model error**. |
| Other fail | Any other normally completed reward-0 attempt: mathematical/numerical or coding failures, failed calibration/parameter checks, or a verifier-caught model error without usable prediction metrics. |
| Timeout | Recorded agent or verifier timeout, regardless of any saved checkpoint reward. |
| Other execution error | Another terminal harness/API/agent-process exception, rather than an ordinary grading failure. |

Execution exceptions take precedence over grades. Recovered intermediate tool errors do not become separate outcomes. “Only succeeded observed prediction” means successful calibration/parameter checks and failed generalization; it does not mean that every hidden prediction failed. Mixed cases use the reviewed primary cause: a mathematical/coding primary remains Other fail even when a secondary physics issue is present.

## Counts by model, effort and hint

| Model | Effort | Hint | Passed | Observed only: physics fail | Other fail | Timeout | Other execution error | Total |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpt-5.6-terra | low | No hint | 5 | 34 | 0 | 0 | 0 | 39 |
| gpt-5.6-terra | low | General | 12 | 19 | 7 | 1 | 0 | 39 |
| gpt-5.6-terra | low | Physics | 25 | 2 | 10 | 2 | 0 | 39 |
| gpt-5.6-terra | medium | No hint | 5 | 34 | 0 | 0 | 0 | 39 |
| gpt-5.6-terra | medium | General | 24 | 9 | 3 | 2 | 1 | 39 |
| gpt-5.6-terra | medium | Physics | 32 | 2 | 4 | 1 | 0 | 39 |
| gpt-5.6-sol | low | No hint | 24 | 14 | 1 | 0 | 0 | 39 |
| gpt-5.6-sol | low | General | 32 | 3 | 4 | 0 | 0 | 39 |
| gpt-5.6-sol | low | Physics | 37 | 0 | 2 | 0 | 0 | 39 |
| gpt-5.6-sol | medium | No hint | 31 | 6 | 2 | 0 | 0 | 39 |
| gpt-5.6-sol | medium | General | 38 | 1 | 0 | 0 | 0 | 39 |
| gpt-5.6-sol | medium | Physics | 38 | 0 | 1 | 0 | 0 | 39 |
| gpt-5.6-luna | none | No hint | 1 | 38 | 0 | 0 | 0 | 39 |
| gpt-5.6-luna | none | General | 1 | 36 | 2 | 0 | 0 | 39 |
| gpt-5.6-luna | none | Physics | 3 | 26 | 8 | 2 | 0 | 39 |
| gpt-5.6-luna | low | No hint | 0 | 38 | 1 | 0 | 0 | 39 |
| gpt-5.6-luna | low | General | 0 | 34 | 5 | 0 | 0 | 39 |
| gpt-5.6-luna | low | Physics | 8 | 20 | 11 | 0 | 0 | 39 |
| gpt-5.6-luna | medium | No hint | 0 | 38 | 1 | 0 | 0 | 39 |
| gpt-5.6-luna | medium | General | 9 | 25 | 5 | 0 | 0 | 39 |
| gpt-5.6-luna | medium | Physics | 16 | 10 | 12 | 1 | 0 | 39 |
| gpt-5.6-luna | high | No hint | 4 | 33 | 1 | 0 | 1 | 39 |
| gpt-5.6-luna | high | General | 18 | 12 | 6 | 3 | 0 | 39 |
| gpt-5.6-luna | high | Physics | 30 | 2 | 7 | 0 | 0 | 39 |
| gpt-5.6-luna | xhigh | No hint | 6 | 32 | 0 | 1 | 0 | 39 |
| gpt-5.6-luna | xhigh | General | 22 | 12 | 2 | 3 | 0 | 39 |
| gpt-5.6-luna | xhigh | Physics | 33 | 3 | 2 | 1 | 0 | 39 |
| gpt-5.6-luna | max | No hint | 8 | 29 | 2 | 0 | 0 | 39 |
| gpt-5.6-luna | max | General | 27 | 7 | 0 | 5 | 0 | 39 |
| gpt-5.6-luna | max | Physics | 35 | 1 | 0 | 3 | 0 | 39 |

## Totals

| Hint | Passed | Observed only: physics fail | Other fail | Timeout | Other execution error | Total |
|---|---:|---:|---:|---:|---:|---:|
| No hint | 84 | 296 | 8 | 1 | 1 | 390 |
| General | 183 | 158 | 34 | 14 | 1 | 390 |
| Physics | 257 | 66 | 57 | 10 | 0 | 390 |
| **All conditions** | **524** | **520** | **99** | **25** | **2** | **1,170** |

## Evidence and edge cases

The 315 graded failures in the hinted batch reuse the existing full observable-trajectory and submitted-code reviews. The 304 historical no-hint graded failures received targeted reviews for this report: relevant observable agent statements/actions, submitted prediction functions, verifier logs and metrics. Exact function or algebraic identity was used where multiple attempts retained the same physical approximation; each attempt retains its own review record. This new historical review is not described as an exhaustive native-event audit. No new model runs or diagnostic numerical reruns were performed, and original grades were unchanged. Passed outcomes use the terminal verifier result; normal passing trajectories were not newly reviewed.

Calibration/parameter success plus a hidden-prediction failure alone would identify 604 attempts. Of those, 84 have a reviewed mathematical/numerical or coding primary cause and are counted as Other fail. Examples include detector-coordinate conversion, an anion charge sign, incorrect operator derivatives and a phase-wrap implementation error.

The calibration threshold is <1.5. Parameter error must be <3%, except reaction–diffusion (<5%). Hidden prediction error must be <4%, except magnetic tracer (<2.5%) and reaction–diffusion (<5%). These are the frozen verifier thresholds; public-test failures also prevent the observed-only classification.

Three hinted attempts have verifier-passing saved checkpoints but agent timeouts; all remain in Timeout. The no-hint Luna xhigh magnetic-tracer interruption is also a timeout. The two matrix-level other execution errors are the no-hint Luna high rotating-reservoir nonzero agent exit and the general-hint Terra medium rotating-reservoir API-capacity rejection. The latter has no observed model output but consumes its original attempt slot.

The five pre-agent DNS setup failures below were preserved and replaced only because no agent execution began. They are excluded from the 39-attempt matrix rows to avoid double-counting their replacements. Including those originals gives **1,175 orchestration records**, with **7 other execution errors** in total; the other four outcome totals are unchanged.

| Model | Effort | Hint | Additional preserved setup execution errors |
|---|---|---|---:|
| gpt-5.6-luna | low | Physics | 1 |
| gpt-5.6-luna | none | Physics | 1 |
| gpt-5.6-sol | medium | Physics | 1 |
| gpt-5.6-terra | low | Physics | 1 |
| gpt-5.6-terra | medium | Physics | 1 |

## Provenance

- No-hint sources: [Terra/Sol report](../sol-terra-low-medium-20261011-185155/report.md) and [Luna report](../luna-all-efforts-20261011-205240/report.md).
- Hinted source: [summary and costs](summary.md), with [independent verification](verification.json).
- Task bytes, private grading and base instructions match across these three conditions. The historical no-hint batches used four workers without the newer per-container CPU quotas; hinted runs used up to eight workers capped at 8.8 CPU cores including proxies. Comparisons are descriptive, especially for timeouts.
- The roster is the ten locally qualified snapshots plus three imported tasks; this count does not assert identical qualification status for all thirteen.
- Audit ledger and targeted review records remain local under `jobs/outcome-classification-20261015/`; 4,686 original artifact hashes were checked. No trajectories are embedded in this Markdown.
- Generated 2026-10-16T00:26:47.441842+00:00.

Classification ledger SHA-256: `b9faeffe2d4db9c983eab440f52b0eb19cf9222ead0e142fd0c9dfb2a2e5faca`.
