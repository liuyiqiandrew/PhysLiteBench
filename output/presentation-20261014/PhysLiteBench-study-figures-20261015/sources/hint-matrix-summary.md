# GPT-5.6 hint comparison summary

Completed October 15, 2026 on 13 frozen tasks: Terra and Sol at low/medium effort, and Luna at none/low/medium/high/xhigh/max. Every task/model/effort setting received three independent general-hint attempts and three independent physics-hint attempts, including after a pass: **780 matrix attempts** in total.

[Panel data](summary.csv) · [Frozen API prices](pricing.json) · [Verification record](verification.json)

## Overall results

| Condition | Passes | Pass rate | Recorded tokens | Recorded API-equivalent cost | Attempts with unknown full cost |
|---|---:|---:|---:|---:|---:|
| General hint | 183/390 | 46.9% | 162,355,051 | ≥$44.934539 | 15 |
| Physics hint | 257/390 | 65.9% | 174,841,357 | ≥$43.135639 | 4 |
| Combined | 440/780 | 56.4% | 337,196,408 | ≥$88.070178 | 19 |

The historical no-hint panels passed **84/390 (21.5%)**. General hints increased the observed pass rate by 25.4 percentage points and physics hints by 44.4 points. These are descriptive comparisons of independent batches, not matched random-seed pairs. Source, grading and base-prompt hashes match; worker concurrency and CPU limits differ.

A pass requires verifier reward 1 and no execution exception. The matrix contains **440 clean passes, 315 graded failures, 24 timeouts and one API-capacity rejection**. Five additional attempts failed during setup before agent execution; their original records were retained and only those five missing slots were filled. Thus there are **785 orchestration records for 780 matrix attempts**. No model/agent-phase failure was resampled.

## Pass rates by model and effort

| Model | Effort | No hint, historical | General hint | Physics hint |
|---|---|---:|---:|---:|
| Terra | low | 5/39 (12.8%) | 12/39 (30.8%) | 25/39 (64.1%) |
| Terra | medium | 5/39 (12.8%) | 24/39 (61.5%) | 32/39 (82.1%) |
| Sol | low | 24/39 (61.5%) | 32/39 (82.1%) | 37/39 (94.9%) |
| Sol | medium | 31/39 (79.5%) | 38/39 (97.4%) | 38/39 (97.4%) |
| Luna | none | 1/39 (2.6%) | 1/39 (2.6%) | 3/39 (7.7%) |
| Luna | low | 0/39 (0.0%) | 0/39 (0.0%) | 8/39 (20.5%) |
| Luna | medium | 0/39 (0.0%) | 9/39 (23.1%) | 16/39 (41.0%) |
| Luna | high | 4/39 (10.3%) | 18/39 (46.2%) | 30/39 (76.9%) |
| Luna | xhigh | 6/39 (15.4%) | 22/39 (56.4%) | 33/39 (84.6%) |
| Luna | max | 8/39 (20.5%) | 27/39 (69.2%) | 35/39 (89.7%) |

Baseline reports: [Terra/Sol](../sol-terra-low-medium-20261011-185155/report.md) and [Luna](../luna-all-efforts-20261011-205240/report.md).

## Recorded tokens and cost by panel

Each panel contains 39 attempts. Recorded totals include successful and unsuccessful attempts. An ≥ marker identifies a recorded dollar lower bound rather than a known full cost. Unknown values remain blank in the CSV; they are not zero.

| Model | Effort | Hint | Recorded tokens | Attempts with token records | API-equivalent cost | Attempts with unknown full cost |
|---|---|---|---:|---:|---:|---:|
| Terra | low | general | 7,683,766 | 39/39 | ≥$6.126900 | 1 |
| Terra | low | physics | 7,833,276 | 39/39 | $5.978328 | 0 |
| Terra | medium | general | 9,069,931 | 38/39 | ≥$7.075431 | 3 |
| Terra | medium | physics | 9,447,784 | 39/39 | ≥$7.065796 | 1 |
| Sol | low | general | 5,862,993 | 39/39 | $8.565079 | 0 |
| Sol | low | physics | 5,920,143 | 39/39 | $8.458303 | 0 |
| Sol | medium | general | 10,203,922 | 39/39 | $14.467490 | 0 |
| Sol | medium | physics | 8,484,001 | 39/39 | $12.513553 | 0 |
| Luna | none | general | 6,922,787 | 39/39 | $0.401831 | 0 |
| Luna | none | physics | 6,532,828 | 39/39 | $0.416286 | 0 |
| Luna | low | general | 5,085,814 | 39/39 | $0.386253 | 0 |
| Luna | low | physics | 6,206,633 | 39/39 | $0.428627 | 0 |
| Luna | medium | general | 9,672,429 | 39/39 | $0.708176 | 0 |
| Luna | medium | physics | 12,137,696 | 39/39 | $0.837414 | 0 |
| Luna | high | general | 27,919,882 | 39/39 | ≥$1.828375 | 3 |
| Luna | high | physics | 27,483,760 | 39/39 | $1.773035 | 0 |
| Luna | xhigh | general | 26,868,037 | 39/39 | ≥$2.017124 | 3 |
| Luna | xhigh | physics | 37,913,054 | 39/39 | ≥$2.404693 | 1 |
| Luna | max | general | 53,065,490 | 39/39 | ≥$3.357879 | 5 |
| Luna | max | physics | 52,882,182 | 39/39 | ≥$3.259606 | 2 |

Native usage is available for 779 of the 780 matrix attempts. The remaining attempt reached the agent phase but received an API-capacity rejection before model output or usage. Eighteen agent timeouts have recorded usage but unknown full cost; together these account for the 19 unknown full costs. The six verifier timeouts retain completed agent usage. The five pre-agent setup records also have no native usage and are excluded from panel denominators.

Costs apply the frozen October 15 Standard API prices to each request, including cache discounts and long-context multipliers. They are **API-equivalent estimates, not actual OAuth subscription charges**; local compute is excluded. Input tokens include cached reads/writes, and output tokens already include reasoning. Recorded token totals are lower bounds where sessions were interrupted.

## Passes by task

| Task | General hint | Physics hint |
|---|---:|---:|
| entropy-anomaly | 16/30 | 18/30 |
| finite-ion-cloud | 17/30 | 22/30 |
| finite-layer-phoresis | 21/30 | 24/30 |
| hard-rod-current | 16/30 | 18/30 |
| hierarchical-glass-overlaps | 11/30 | 16/30 |
| insulating-mhd-duct | 14/30 | 14/30 |
| magnetic-tracer | 14/30 | 19/30 |
| reaction-diffusion | 13/30 | 29/30 |
| ring-current-fluctuations | 12/30 | 20/30 |
| rotating-reservoir | 7/30 | 20/30 |
| single-file-memory | 23/30 | 23/30 |
| superconducting-heat | 2/30 | 15/30 |
| swirling-heat-tails | 17/30 | 19/30 |

## Failure review and validation

All 315 graded failures and all 30 exception records were reviewed. Primary graded-failure categories were **229 physics/modeling, 32 implementation and 54 mathematical/numerical**. Common failures retained an approximation that matched calibration but missed the hidden physics; other attempts understood the hint but introduced sign, normalization, indexing or angle errors. Exceptions, including the five pre-agent setup failures, are kept separate from scientific failure categories. Three verifier-passing checkpoints with agent timeouts remain exceptions, not clean passes.

The independent audit confirmed 260 task/model/effort/hint combinations with exactly three attempts each, 780 unique native sessions and exact prompts, 860 source/runtime hash checks, and 10,601 independently repriced request records. The evaluated source commit was `3bcd76f4e668927a48e8fb393b897da1e85e927e`; no-hint baselines used `34dedfbbd8ff86a7e6fb86593dff5a0c1ed25482` with matching task bytes.

Harbor 0.21.0 and Codex CLI 0.154.0 used 600-second agent and 60-second verifier limits. The new batch used up to eight workers, capped at 8.8 CPU cores including network proxies on a 16-core host, leaving 7.2 CPU equivalents outside trial quotas. The five setup replacements used at most 5.5 cores. No sampled quota violations occurred. Historical no-hint runs used four workers without these per-container CPU quotas, which limits timeout-sensitive comparisons.

The roster contains the original ten locally qualified tasks plus three imported tasks. This report does not claim that all thirteen share the original qualification result. Full trajectories, submitted code and the shareable evidence archive remain local; this Git summary contains aggregate results and verification metadata only.

Source result SHA-256: `f4a8c1b79c6c5edcfbecc1ce531299db7a438eeabced2c8e2da8ffb9757ea30a`.
Frozen protocol SHA-256: `c63a6c87fdbe974ca1b5b3e3505cdb566b5be21bc014684950b2c6a3eaed8c80`.
