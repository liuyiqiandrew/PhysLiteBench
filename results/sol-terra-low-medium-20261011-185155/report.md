# GPT-5.6 Sol, Terra and Luna effort comparison

Sol/Terra batch complete: 156/156 trials finished on the thirteen roster tasks at commit `34dedfb`. Each model runs at low and medium reasoning effort, three independent times per task, with no hints and no stopping after a pass.

[Trial data](trials.csv) · [Sol/Terra summary](summary.csv) · [Combined Pareto summary](pareto-summary.csv) · [Per-request costs](requests.csv) · [Verification](verification.json) · [Protocol and source hashes](protocol.json) · [Luna all-effort results](../luna-all-efforts-20261011-205240/report.md)

## Model and effort results

A pass requires verifier reward 1 and no execution exception. Timeouts and other errors remain in the planned denominator and are reported separately. Costs include unsuccessful attempts with recorded usage.

| Model | Effort | Finished | Passes | Timeouts | Other errors | Total tokens | Total API equivalent | Mean per trial |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-5.6-sol | low | 39/39 | 24 | 0 | 0 | 5,433,080 | $7.143462 | $0.183166 |
| gpt-5.6-sol | medium | 39/39 | 31 | 0 | 0 | 9,454,516 | $12.394237 | $0.317801 |
| gpt-5.6-terra | low | 39/39 | 5 | 0 | 0 | 6,657,344 | $4.610282 | $0.118212 |
| gpt-5.6-terra | medium | 39/39 | 5 | 0 | 0 | 6,498,189 | $4.889242 | $0.125365 |


## GPT-5.6 Luna: all efforts

Fresh unhinted sweep: 13 tasks × 6 efforts × 3 independent runs = 234 trials. Task and runtime hashes match the Sol/Terra batch above.

[Full Luna report](../luna-all-efforts-20261011-205240/report.md) · [Luna trial data](../luna-all-efforts-20261011-205240/trials.csv) · [Luna verification](../luna-all-efforts-20261011-205240/verification.json) · [Combined Pareto summary](pareto-summary.csv)

| Effort | Passes | Pass rate | Recorded tokens | Total API equivalent | Recorded API/run |
|---|---:|---:|---:|---:|---:|
| none | 1/39 | 2.6% | 5,231,761 | $0.325781 | $0.008353 |
| low | 0/39 | 0.0% | 4,291,442 | $0.289999 | $0.007436 |
| medium | 0/39 | 0.0% | 7,823,950 | $0.526987 | $0.013512 |
| high | 4/39 | 10.3% | 14,677,409 | unknown | ≥$0.026321 |
| xhigh | 6/39 | 15.4% | 17,766,584 | unknown | ≥$0.033133 |
| max | 8/39 | 20.5% | 32,325,294 | $1.999129 | $0.051260 |

Interrupted sessions may omit an unfinished request: their recorded tokens and costs are lower bounds, and full cost is unknown. The original ten tasks were selected using earlier Luna High failures. This sweep uses fresh sessions on that selected roster; it is not an independent held-out evaluation. Three runs per task give limited precision.

Six harmless compatibility checks verified the effort settings before the sweep. Their $0.015007 API-equivalent cost is recorded separately in the Luna access-check.json and excluded from the physics totals. Costs are estimates, not subscription charges.

## Sol/Terra task results

Cells show passes / finished trials; each cell is planned for three trials.

| Task | Revision | Sol low | Sol medium | Terra low | Terra medium |
|---|---:|---:|---:|---:|---:|
| entropy-anomaly | 9 | 2/3 | 2/3 | 0/3 | 0/3 |
| finite-layer-phoresis | 2 | 2/3 | 3/3 | 1/3 | 1/3 |
| hierarchical-glass-overlaps | 2 | 3/3 | 3/3 | 0/3 | 1/3 |
| insulating-mhd-duct | 2 | 3/3 | 3/3 | 2/3 | 1/3 |
| magnetic-tracer | 9 | 1/3 | 3/3 | 0/3 | 0/3 |
| reaction-diffusion | 10 | 0/3 | 0/3 | 0/3 | 0/3 |
| ring-current-fluctuations | 1 | 3/3 | 3/3 | 0/3 | 1/3 |
| rotating-reservoir | 11 | 0/3 | 2/3 | 0/3 | 0/3 |
| single-file-memory | 1 | 2/3 | 3/3 | 0/3 | 1/3 |
| superconducting-heat | 11 | 0/3 | 0/3 | 0/3 | 0/3 |
| swirling-heat-tails | 1 | 3/3 | 3/3 | 0/3 | 0/3 |
| hard-rod-current | 2 | 3/3 | 3/3 | 0/3 | 0/3 |
| finite-ion-cloud | 3 | 2/3 | 3/3 | 2/3 | 0/3 |

## Token accounting and API equivalent cost

Token counts come from native Codex session logs. Each unique cumulative token event contributes its last-request usage once; sums are checked against the final session totals. Input includes cached reads and cache writes. Output already includes reasoning tokens. Total tokens are input plus output, without adding caches or reasoning twice. Missing or inconsistent usage remains unknown. Harbor's own estimates remain in the local detailed records.

Standard USD rates per million tokens, verified on October 11, 2026:

| Model | Ordinary input | Cached input | Cache write | Output including reasoning |
|---|---:|---:|---:|---:|
| [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol) | $4.00 | $0.40 | $5.00 | $20.00 |
| [GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra) | $2.00 | $0.20 | $2.50 | $12.00 |

For each request, charge ordinary input = total input minus cache reads minus cache writes. Requests above 272,000 input tokens use twice the input/cache rates and 1.5 times the output rate for the whole request. Apply the threshold to each request, not trial totals. [Frozen pricing](pricing.json) records the rates; Sol's promotional rates are stated available at least through November 21, 2026.

These are API-equivalent estimates for OAuth-authenticated runs, not measured subscription charges. Local compute is excluded. Actual service tier and subscription billing are not inferred.

## Protocol and later Pareto analysis

Harbor 0.21.0, Codex CLI 0.154.0, four concurrent trials, 600 seconds to solve and 60 seconds to verify. Agent setup receives a separate 3× multiplier. Web search is disabled and agent networking is limited to OpenAI authentication/API hosts. Runtime task configuration adds only that network policy; all instructions, public files and private grading are frozen. Credentials stay outside the repository.

Plot each row of `summary.csv` using `mean_api_cost_usd_per_trial` against `pass_rate`. All four panels use the same 13 tasks with equal repetition. Use `trials.csv` for task-clustered uncertainty estimates and `requests.csv` to reprice later. Three trials per task give limited precision. Compare with historical panels only after matching task hashes, hint policy, model version and runtime limits. Timeouts and access errors measure end-to-end performance, not physics knowledge alone.

Published files contain scores, token accounting, API-equivalent costs, frozen pricing, task hashes and verification records. Raw transcripts and submitted files remain local in `jobs/sol-terra-low-medium-20261011-185155`; detailed evidence and full result JSON also remain local. The `evidence` column in `trials.csv` and the `source`/`line` columns in `requests.csv` identify those local records. Each session is fresh; no scored outcome is resampled.

## Individual trials

| Task | Model | Effort | Repeat | Result | Input | Cached | Output | Reasoning included | API equivalent | Trial ID |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| entropy-anomaly | sol | low | 1 | pass | 147,682 | 131,584 | 3,901 | 1,379 | $0.195046 | `entropy-anomaly__C5HmwJU` |
| entropy-anomaly | sol | low | 2 | pass | 165,318 | 148,352 | 3,784 | 1,962 | $0.202885 | `entropy-anomaly__ezhBhpG` |
| entropy-anomaly | sol | low | 3 | fail | 128,883 | 118,016 | 3,700 | 1,324 | $0.164674 | `entropy-anomaly__28UbhqX` |
| entropy-anomaly | sol | medium | 1 | pass | 295,453 | 261,376 | 8,266 | 4,360 | $0.406178 | `entropy-anomaly__sAXkbmq` |
| entropy-anomaly | sol | medium | 2 | pass | 267,411 | 241,920 | 7,019 | 4,987 | $0.339112 | `entropy-anomaly__dcMLmYs` |
| entropy-anomaly | sol | medium | 3 | fail | 180,500 | 165,120 | 4,647 | 2,179 | $0.220508 | `entropy-anomaly__AZnMdVL` |
| entropy-anomaly | terra | low | 1 | fail | 167,778 | 145,408 | 2,525 | 915 | $0.104122 | `entropy-anomaly__xtSPwCi` |
| entropy-anomaly | terra | low | 2 | fail | 160,791 | 138,496 | 2,186 | 799 | $0.098521 | `entropy-anomaly__fMCczDf` |
| entropy-anomaly | terra | low | 3 | fail | 205,693 | 180,992 | 2,932 | 1,065 | $0.120784 | `entropy-anomaly__TksDnf6` |
| entropy-anomaly | terra | medium | 1 | fail | 123,262 | 100,608 | 2,507 | 842 | $0.095514 | `entropy-anomaly__gCM4wHB` |
| entropy-anomaly | terra | medium | 2 | fail | 128,440 | 104,704 | 3,366 | 1,362 | $0.108805 | `entropy-anomaly__zv9WyrA` |
| entropy-anomaly | terra | medium | 3 | fail | 127,194 | 102,656 | 2,119 | 402 | $0.095035 | `entropy-anomaly__biBYKjh` |
| finite-layer-phoresis | sol | low | 1 | pass | 94,262 | 84,992 | 2,744 | 901 | $0.125957 | `finite-layer-phoresis__2o2qgBa` |
| finite-layer-phoresis | sol | low | 2 | fail | 91,811 | 83,584 | 1,829 | 325 | $0.102922 | `finite-layer-phoresis__uwYj9k7` |
| finite-layer-phoresis | sol | low | 3 | pass | 133,728 | 122,880 | 5,221 | 899 | $0.196964 | `finite-layer-phoresis__4wjCguQ` |
| finite-layer-phoresis | sol | medium | 1 | pass | 143,305 | 128,128 | 6,673 | 3,811 | $0.245419 | `finite-layer-phoresis__mayp5xT` |
| finite-layer-phoresis | sol | medium | 2 | pass | 126,663 | 113,792 | 4,857 | 2,774 | $0.194141 | `finite-layer-phoresis__J9K2kLC` |
| finite-layer-phoresis | sol | medium | 3 | pass | 150,145 | 129,152 | 7,451 | 2,272 | $0.284653 | `finite-layer-phoresis__x5BkpnN` |
| finite-layer-phoresis | terra | low | 1 | fail | 93,798 | 73,216 | 1,747 | 391 | $0.076771 | `finite-layer-phoresis__ZEDd5zD` |
| finite-layer-phoresis | terra | low | 2 | fail | 94,340 | 84,224 | 2,598 | 1,237 | $0.068253 | `finite-layer-phoresis__Kbawnif` |
| finite-layer-phoresis | terra | low | 3 | pass | 101,615 | 89,344 | 3,417 | 1,782 | $0.083415 | `finite-layer-phoresis__yDTVZxq` |
| finite-layer-phoresis | terra | medium | 1 | pass | 168,779 | 139,264 | 6,509 | 2,014 | $0.164991 | `finite-layer-phoresis__c6DrBso` |
| finite-layer-phoresis | terra | medium | 2 | fail | 139,445 | 112,896 | 2,504 | 884 | $0.105725 | `finite-layer-phoresis__WfSF5ry` |
| finite-layer-phoresis | terra | medium | 3 | fail | 115,518 | 98,560 | 2,139 | 448 | $0.079296 | `finite-layer-phoresis__sM4pHDG` |
| hierarchical-glass-overlaps | sol | low | 1 | pass | 155,806 | 136,192 | 2,611 | 1,260 | $0.185153 | `hierarchical-glass-overlaps__EZY98Tg` |
| hierarchical-glass-overlaps | sol | low | 2 | pass | 113,553 | 100,480 | 3,047 | 1,366 | $0.153424 | `hierarchical-glass-overlaps__6H6KJSV` |
| hierarchical-glass-overlaps | sol | low | 3 | pass | 102,320 | 85,504 | 3,254 | 1,590 | $0.166546 | `hierarchical-glass-overlaps__nzq3MSP` |
| hierarchical-glass-overlaps | sol | medium | 1 | pass | 162,520 | 139,264 | 4,487 | 2,387 | $0.238470 | `hierarchical-glass-overlaps__H7NDnyy` |
| hierarchical-glass-overlaps | sol | medium | 2 | pass | 75,991 | 67,072 | 3,635 | 2,273 | $0.135205 | `hierarchical-glass-overlaps__847jcz6` |
| hierarchical-glass-overlaps | sol | medium | 3 | pass | 96,733 | 85,376 | 3,506 | 1,751 | $0.149698 | `hierarchical-glass-overlaps__fM8ApjY` |
| hierarchical-glass-overlaps | terra | low | 1 | fail | 177,428 | 146,688 | 3,175 | 1,246 | $0.128918 | `hierarchical-glass-overlaps__MBBGYam` |
| hierarchical-glass-overlaps | terra | low | 2 | fail | 127,191 | 99,072 | 2,571 | 1,069 | $0.106904 | `hierarchical-glass-overlaps__UkTS6c9` |
| hierarchical-glass-overlaps | terra | low | 3 | fail | 104,059 | 86,784 | 2,544 | 1,208 | $0.082435 | `hierarchical-glass-overlaps__FALVeVS` |
| hierarchical-glass-overlaps | terra | medium | 1 | fail | 75,358 | 55,296 | 3,329 | 2,175 | $0.091131 | `hierarchical-glass-overlaps__KQMwSKe` |
| hierarchical-glass-overlaps | terra | medium | 2 | pass | 165,869 | 142,592 | 5,112 | 3,032 | $0.136416 | `hierarchical-glass-overlaps__vPgyGMy` |
| hierarchical-glass-overlaps | terra | medium | 3 | fail | 184,128 | 160,768 | 4,171 | 1,901 | $0.128926 | `hierarchical-glass-overlaps__4kWBi48` |
| insulating-mhd-duct | sol | low | 1 | pass | 118,007 | 106,880 | 3,850 | 1,803 | $0.164260 | `insulating-mhd-duct__4fpPtze` |
| insulating-mhd-duct | sol | low | 2 | pass | 149,092 | 135,168 | 3,876 | 1,641 | $0.187283 | `insulating-mhd-duct__KgRVAAB` |
| insulating-mhd-duct | sol | low | 3 | pass | 120,050 | 109,568 | 4,372 | 2,113 | $0.173195 | `insulating-mhd-duct__jxZJNAi` |
| insulating-mhd-duct | sol | medium | 1 | pass | 200,410 | 183,424 | 6,804 | 3,825 | $0.277394 | `insulating-mhd-duct__fxugPad` |
| insulating-mhd-duct | sol | medium | 2 | pass | 212,452 | 178,560 | 6,220 | 3,318 | $0.331392 | `insulating-mhd-duct__YQeU9TH` |
| insulating-mhd-duct | sol | medium | 3 | pass | 219,523 | 202,496 | 7,290 | 3,994 | $0.294906 | `insulating-mhd-duct__k8qHSr5` |
| insulating-mhd-duct | terra | low | 1 | fail | 101,008 | 88,320 | 2,107 | 749 | $0.068324 | `insulating-mhd-duct__8upr6fe` |
| insulating-mhd-duct | terra | low | 2 | pass | 141,139 | 123,648 | 5,275 | 3,270 | $0.123012 | `insulating-mhd-duct__667yyhj` |
| insulating-mhd-duct | terra | low | 3 | pass | 238,605 | 215,296 | 4,951 | 2,778 | $0.149089 | `insulating-mhd-duct__dW7ajD3` |
| insulating-mhd-duct | terra | medium | 1 | fail | 138,166 | 109,056 | 4,277 | 2,963 | $0.131355 | `insulating-mhd-duct__y3LzvKs` |
| insulating-mhd-duct | terra | medium | 2 | fail | 212,547 | 182,016 | 4,798 | 2,751 | $0.155041 | `insulating-mhd-duct__sVkqb2i` |
| insulating-mhd-duct | terra | medium | 3 | pass | 363,629 | 333,056 | 8,496 | 5,017 | $0.229709 | `insulating-mhd-duct__mu4UyAg` |
| magnetic-tracer | sol | low | 1 | fail | 108,429 | 94,208 | 1,668 | 740 | $0.127927 | `magnetic-tracer__cvnmY2f` |
| magnetic-tracer | sol | low | 2 | pass | 157,579 | 142,336 | 3,489 | 1,734 | $0.187686 | `magnetic-tracer__rMNgQ6m` |
| magnetic-tracer | sol | low | 3 | fail | 91,398 | 78,592 | 1,172 | 162 | $0.106101 | `magnetic-tracer__VEpTUjz` |
| magnetic-tracer | sol | medium | 1 | pass | 264,503 | 239,616 | 7,912 | 3,956 | $0.353634 | `magnetic-tracer__fx493wX` |
| magnetic-tracer | sol | medium | 2 | pass | 375,858 | 350,720 | 9,579 | 5,426 | $0.432420 | `magnetic-tracer__sNBhE49` |
| magnetic-tracer | sol | medium | 3 | pass | 589,779 | 559,232 | 11,807 | 5,359 | $0.582021 | `magnetic-tracer__AMHUfM8` |
| magnetic-tracer | terra | low | 1 | fail | 192,235 | 144,128 | 1,464 | 360 | $0.142608 | `magnetic-tracer__a8HToKy` |
| magnetic-tracer | terra | low | 2 | fail | 112,955 | 96,768 | 1,618 | 963 | $0.071144 | `magnetic-tracer__ZfB4RY8` |
| magnetic-tracer | terra | low | 3 | fail | 103,377 | 88,576 | 2,788 | 1,598 | $0.080773 | `magnetic-tracer__bk6tGar` |
| magnetic-tracer | terra | medium | 1 | fail | 124,352 | 98,816 | 2,563 | 1,505 | $0.101591 | `magnetic-tracer__c2NLGbs` |
| magnetic-tracer | terra | medium | 2 | fail | 178,517 | 160,512 | 3,221 | 1,118 | $0.106764 | `magnetic-tracer__d5vPrJf` |
| magnetic-tracer | terra | medium | 3 | fail | 194,697 | 169,984 | 3,127 | 1,639 | $0.120947 | `magnetic-tracer__4rJRtQi` |
| reaction-diffusion | sol | low | 1 | fail | 191,457 | 174,976 | 4,291 | 1,857 | $0.221734 | `reaction-diffusion__WDdvWn3` |
| reaction-diffusion | sol | low | 2 | fail | 162,455 | 145,920 | 8,111 | 1,194 | $0.286728 | `reaction-diffusion__9ZdMt6e` |
| reaction-diffusion | sol | low | 3 | fail | 208,546 | 189,824 | 8,230 | 1,602 | $0.315418 | `reaction-diffusion__VQtrfNb` |
| reaction-diffusion | sol | medium | 1 | fail | 218,783 | 198,912 | 10,991 | 2,461 | $0.378869 | `reaction-diffusion__UZApbmA` |
| reaction-diffusion | sol | medium | 2 | fail | 276,645 | 254,464 | 12,479 | 2,687 | $0.440090 | `reaction-diffusion__xdiES8Y` |
| reaction-diffusion | sol | medium | 3 | fail | 242,094 | 219,904 | 11,292 | 2,654 | $0.402562 | `reaction-diffusion__LEspJLW` |
| reaction-diffusion | terra | low | 1 | fail | 126,545 | 101,632 | 3,499 | 1,243 | $0.112140 | `reaction-diffusion__KZrQea7` |
| reaction-diffusion | terra | low | 2 | fail | 122,794 | 108,544 | 3,953 | 1,659 | $0.097645 | `reaction-diffusion__QtMMojU` |
| reaction-diffusion | terra | low | 3 | fail | 223,592 | 199,936 | 7,421 | 1,105 | $0.176351 | `reaction-diffusion__YQW3t7d` |
| reaction-diffusion | terra | medium | 1 | fail | 133,640 | 107,776 | 7,976 | 1,574 | $0.168995 | `reaction-diffusion__HexBrfD` |
| reaction-diffusion | terra | medium | 2 | fail | 216,807 | 172,544 | 7,802 | 2,265 | $0.216659 | `reaction-diffusion__2MwwPq6` |
| reaction-diffusion | terra | medium | 3 | fail | 134,786 | 107,776 | 6,469 | 969 | $0.153203 | `reaction-diffusion__dyCFPXs` |
| ring-current-fluctuations | sol | low | 1 | pass | 117,627 | 98,048 | 6,341 | 2,314 | $0.244355 | `ring-current-fluctuations__4WpFpSk` |
| ring-current-fluctuations | sol | low | 2 | pass | 120,061 | 109,440 | 4,867 | 1,941 | $0.183600 | `ring-current-fluctuations__CcjG6yD` |
| ring-current-fluctuations | sol | low | 3 | pass | 116,161 | 107,136 | 5,050 | 2,063 | $0.179954 | `ring-current-fluctuations__BJgvczR` |
| ring-current-fluctuations | sol | medium | 1 | pass | 204,291 | 185,984 | 9,376 | 4,180 | $0.335142 | `ring-current-fluctuations__rHjarUm` |
| ring-current-fluctuations | sol | medium | 2 | pass | 230,822 | 211,584 | 8,675 | 3,515 | $0.335086 | `ring-current-fluctuations__HKjwVhN` |
| ring-current-fluctuations | sol | medium | 3 | pass | 269,131 | 250,240 | 11,024 | 4,472 | $0.396140 | `ring-current-fluctuations__KpeMvuz` |
| ring-current-fluctuations | terra | low | 1 | fail | 116,514 | 102,400 | 4,001 | 1,196 | $0.096720 | `ring-current-fluctuations__KPgVN2G` |
| ring-current-fluctuations | terra | low | 2 | fail | 141,425 | 125,696 | 5,219 | 2,503 | $0.119225 | `ring-current-fluctuations__tnY2p4r` |
| ring-current-fluctuations | terra | low | 3 | fail | 111,511 | 101,376 | 3,394 | 1,425 | $0.081273 | `ring-current-fluctuations__Ep3jAGt` |
| ring-current-fluctuations | terra | medium | 1 | fail | 184,590 | 151,552 | 3,881 | 1,356 | $0.142958 | `ring-current-fluctuations__YrEoJnP` |
| ring-current-fluctuations | terra | medium | 2 | pass | 353,984 | 322,560 | 10,534 | 4,053 | $0.253768 | `ring-current-fluctuations__k5wd2tp` |
| ring-current-fluctuations | terra | medium | 3 | fail | 110,871 | 101,376 | 3,762 | 930 | $0.084409 | `ring-current-fluctuations__yqTCrVm` |
| rotating-reservoir | sol | low | 1 | fail | 119,114 | 97,152 | 2,473 | 991 | $0.176169 | `rotating-reservoir__YqqBnyt` |
| rotating-reservoir | sol | low | 2 | fail | 96,827 | 86,400 | 1,511 | 308 | $0.106488 | `rotating-reservoir__Kuui3yt` |
| rotating-reservoir | sol | low | 3 | fail | 117,103 | 106,368 | 1,793 | 560 | $0.121347 | `rotating-reservoir__kQN3sVQ` |
| rotating-reservoir | sol | medium | 1 | fail | 145,155 | 129,920 | 3,513 | 1,551 | $0.183168 | `rotating-reservoir__3E5XDZZ` |
| rotating-reservoir | sol | medium | 2 | pass | 191,319 | 174,848 | 5,903 | 3,312 | $0.253883 | `rotating-reservoir__2TyHTqe` |
| rotating-reservoir | sol | medium | 3 | pass | 215,915 | 181,888 | 5,411 | 2,845 | $0.317083 | `rotating-reservoir__JsS6aH5` |
| rotating-reservoir | terra | low | 1 | fail | 425,354 | 382,464 | 9,328 | 2,151 | $0.274209 | `rotating-reservoir__GYKt6zm` |
| rotating-reservoir | terra | low | 2 | fail | 208,607 | 187,648 | 4,853 | 1,381 | $0.137684 | `rotating-reservoir__bmocDaF` |
| rotating-reservoir | terra | low | 3 | fail | 247,628 | 221,184 | 8,280 | 2,174 | $0.196485 | `rotating-reservoir__EDqbd3Q` |
| rotating-reservoir | terra | medium | 1 | fail | 241,316 | 217,088 | 7,512 | 3,695 | $0.182018 | `rotating-reservoir__bYdsSBH` |
| rotating-reservoir | terra | medium | 2 | fail | 360,223 | 342,784 | 6,573 | 626 | $0.182311 | `rotating-reservoir__3sP6hRX` |
| rotating-reservoir | terra | medium | 3 | fail | 97,091 | 85,504 | 2,396 | 880 | $0.069027 | `rotating-reservoir__KYX4q78` |
| single-file-memory | sol | low | 1 | pass | 114,724 | 105,600 | 4,319 | 868 | $0.165116 | `single-file-memory__rD8ENDo` |
| single-file-memory | sol | low | 2 | fail | 90,356 | 83,968 | 1,790 | 170 | $0.094939 | `single-file-memory__4yPMVbS` |
| single-file-memory | sol | low | 3 | pass | 125,447 | 115,584 | 4,671 | 1,149 | $0.179106 | `single-file-memory__4RLBCaE` |
| single-file-memory | sol | medium | 1 | pass | 138,747 | 127,872 | 4,124 | 879 | $0.177129 | `single-file-memory__brLRFhk` |
| single-file-memory | sol | medium | 2 | pass | 172,535 | 158,592 | 3,950 | 882 | $0.198209 | `single-file-memory__WmyZbnZ` |
| single-file-memory | sol | medium | 3 | pass | 105,164 | 95,104 | 4,477 | 1,370 | $0.167822 | `single-file-memory__vFqe66j` |
| single-file-memory | terra | low | 1 | fail | 112,094 | 88,576 | 2,038 | 504 | $0.089207 | `single-file-memory__jmqWh84` |
| single-file-memory | terra | low | 2 | fail | 134,509 | 113,920 | 3,375 | 466 | $0.104462 | `single-file-memory__GT2p2bn` |
| single-file-memory | terra | low | 3 | fail | 73,631 | 58,112 | 1,478 | 279 | $0.060396 | `single-file-memory__khTXfRB` |
| single-file-memory | terra | medium | 1 | fail | 107,396 | 92,416 | 1,817 | 405 | $0.070247 | `single-file-memory__j86X6fg` |
| single-file-memory | terra | medium | 2 | fail | 76,826 | 70,144 | 1,750 | 393 | $0.048393 | `single-file-memory__rHQw3Ze` |
| single-file-memory | terra | medium | 3 | pass | 124,282 | 111,360 | 4,463 | 1,200 | $0.101672 | `single-file-memory__LKvUUCo` |
| superconducting-heat | sol | low | 1 | fail | 277,666 | 240,000 | 4,141 | 1,651 | $0.329484 | `superconducting-heat__KZatyQU` |
| superconducting-heat | sol | low | 2 | fail | 209,741 | 177,664 | 3,337 | 1,129 | $0.266114 | `superconducting-heat__Sdxmb9v` |
| superconducting-heat | sol | low | 3 | fail | 190,781 | 163,200 | 2,648 | 1,379 | $0.228564 | `superconducting-heat__R8PphAp` |
| superconducting-heat | sol | medium | 1 | fail | 578,863 | 546,048 | 13,512 | 7,193 | $0.619919 | `superconducting-heat__LU4C8iu` |
| superconducting-heat | sol | medium | 2 | fail | 773,121 | 737,408 | 13,941 | 6,376 | $0.716635 | `superconducting-heat__swS9MoY` |
| superconducting-heat | sol | medium | 3 | fail | 551,257 | 504,832 | 11,422 | 5,429 | $0.616073 | `superconducting-heat__CGRVTem` |
| superconducting-heat | terra | low | 1 | fail | 256,454 | 219,904 | 2,717 | 590 | $0.149685 | `superconducting-heat__g5BfXgm` |
| superconducting-heat | terra | low | 2 | fail | 340,702 | 308,736 | 6,309 | 2,005 | $0.201387 | `superconducting-heat__Yt62zS6` |
| superconducting-heat | terra | low | 3 | fail | 262,009 | 231,168 | 3,422 | 827 | $0.148980 | `superconducting-heat__HCHjwrj` |
| superconducting-heat | terra | medium | 1 | fail | 261,188 | 224,000 | 3,632 | 1,353 | $0.162760 | `superconducting-heat__yq9PYsN` |
| superconducting-heat | terra | medium | 2 | fail | 179,950 | 146,688 | 3,843 | 1,886 | $0.141978 | `superconducting-heat__VVd5TfF` |
| superconducting-heat | terra | medium | 3 | fail | 251,668 | 211,712 | 2,861 | 1,054 | $0.156586 | `superconducting-heat__nU8QEmp` |
| swirling-heat-tails | sol | low | 1 | pass | 84,840 | 76,928 | 5,715 | 3,066 | $0.176719 | `swirling-heat-tails__gYcZsCS` |
| swirling-heat-tails | sol | low | 2 | pass | 102,580 | 90,240 | 4,263 | 2,642 | $0.170716 | `swirling-heat-tails__rshzvFB` |
| swirling-heat-tails | sol | low | 3 | pass | 119,879 | 110,080 | 4,908 | 2,953 | $0.181388 | `swirling-heat-tails__yNnfDMR` |
| swirling-heat-tails | sol | medium | 1 | pass | 112,682 | 91,520 | 7,730 | 5,583 | $0.275856 | `swirling-heat-tails__EYhNKPL` |
| swirling-heat-tails | sol | medium | 2 | pass | 124,898 | 113,664 | 7,533 | 4,511 | $0.241062 | `swirling-heat-tails__rkx9C9D` |
| swirling-heat-tails | sol | medium | 3 | pass | 144,745 | 134,144 | 6,934 | 3,365 | $0.234742 | `swirling-heat-tails__HHcTsuX` |
| swirling-heat-tails | terra | low | 1 | fail | 74,968 | 67,072 | 2,225 | 678 | $0.055906 | `swirling-heat-tails__WLKaZGZ` |
| swirling-heat-tails | terra | low | 2 | fail | 232,681 | 195,584 | 2,471 | 489 | $0.142963 | `swirling-heat-tails__QfBNMTR` |
| swirling-heat-tails | terra | low | 3 | fail | 75,946 | 68,096 | 1,948 | 478 | $0.052695 | `swirling-heat-tails__YhbC8Kx` |
| swirling-heat-tails | terra | medium | 1 | fail | 115,703 | 100,352 | 3,643 | 1,515 | $0.094488 | `swirling-heat-tails__A8mEB6c` |
| swirling-heat-tails | terra | medium | 2 | fail | 115,327 | 83,968 | 2,499 | 1,011 | $0.109500 | `swirling-heat-tails__FoTF6cP` |
| swirling-heat-tails | terra | medium | 3 | fail | 82,851 | 48,128 | 2,025 | 462 | $0.103372 | `swirling-heat-tails__fzpGBXU` |
| hard-rod-current | sol | low | 1 | pass | 177,952 | 159,488 | 4,811 | 2,886 | $0.233871 | `hard-rod-current__3wzVJog` |
| hard-rod-current | sol | low | 2 | pass | 132,723 | 115,840 | 4,653 | 2,800 | $0.206928 | `hard-rod-current__mR56vJh` |
| hard-rod-current | sol | low | 3 | pass | 152,025 | 135,296 | 3,905 | 1,777 | $0.199134 | `hard-rod-current__Lfo99Ft` |
| hard-rod-current | sol | medium | 1 | pass | 142,493 | 127,232 | 5,900 | 3,773 | $0.229937 | `hard-rod-current__KqCVbRk` |
| hard-rod-current | sol | medium | 2 | pass | 137,021 | 123,008 | 6,905 | 4,592 | $0.243355 | `hard-rod-current__rjPS9AY` |
| hard-rod-current | sol | medium | 3 | pass | 89,430 | 78,720 | 6,579 | 4,605 | $0.205908 | `hard-rod-current__GQakBB9` |
| hard-rod-current | terra | low | 1 | fail | 141,950 | 106,240 | 2,232 | 648 | $0.119452 | `hard-rod-current__GCSvFpJ` |
| hard-rod-current | terra | low | 2 | fail | 145,213 | 122,112 | 1,987 | 584 | $0.094468 | `hard-rod-current__Tgi3GDk` |
| hard-rod-current | terra | low | 3 | fail | 149,196 | 129,792 | 6,138 | 3,150 | $0.138422 | `hard-rod-current__Xw6VoxF` |
| hard-rod-current | terra | medium | 1 | fail | 127,057 | 108,800 | 2,662 | 1,213 | $0.090218 | `hard-rod-current__E2aDC3u` |
| hard-rod-current | terra | medium | 2 | fail | 137,234 | 114,176 | 2,224 | 455 | $0.095639 | `hard-rod-current__AzwJV6B` |
| hard-rod-current | terra | medium | 3 | fail | 144,016 | 117,248 | 5,044 | 3,346 | $0.137514 | `hard-rod-current__UdH4sXc` |
| finite-ion-cloud | sol | low | 1 | pass | 133,429 | 122,624 | 3,539 | 1,270 | $0.163050 | `finite-ion-cloud__p5gZ7xK` |
| finite-ion-cloud | sol | low | 2 | pass | 118,236 | 108,672 | 3,400 | 1,238 | $0.149725 | `finite-ion-cloud__ctKRdWp` |
| finite-ion-cloud | sol | low | 3 | fail | 126,042 | 117,632 | 2,105 | 352 | $0.122793 | `finite-ion-cloud__mqAiQ9n` |
| finite-ion-cloud | sol | medium | 1 | pass | 375,503 | 350,464 | 9,686 | 4,781 | $0.434062 | `finite-ion-cloud__NBS5vBe` |
| finite-ion-cloud | sol | medium | 2 | pass | 171,035 | 155,648 | 5,668 | 2,678 | $0.237167 | `finite-ion-cloud__Lu6iV9R` |
| finite-ion-cloud | sol | medium | 3 | pass | 187,551 | 171,904 | 6,892 | 3,811 | $0.269190 | `finite-ion-cloud__2H8cZCq` |
| finite-ion-cloud | terra | low | 1 | pass | 323,260 | 288,768 | 7,525 | 2,665 | $0.217038 | `finite-ion-cloud__aY3PPc5` |
| finite-ion-cloud | terra | low | 2 | fail | 131,186 | 110,848 | 2,670 | 1,059 | $0.094886 | `finite-ion-cloud__9GECbZu` |
| finite-ion-cloud | terra | low | 3 | pass | 214,474 | 189,952 | 4,708 | 2,339 | $0.143530 | `finite-ion-cloud__Hmg6chG` |
| finite-ion-cloud | terra | medium | 1 | fail | 109,414 | 97,280 | 2,698 | 1,074 | $0.076100 | `finite-ion-cloud__kooQiPf` |
| finite-ion-cloud | terra | medium | 2 | fail | 133,285 | 109,568 | 3,068 | 1,351 | $0.106164 | `finite-ion-cloud__dsrXTMH` |
| finite-ion-cloud | terra | medium | 3 | fail | 98,591 | 78,336 | 2,820 | 1,327 | $0.090017 | `finite-ion-cloud__yhMdWbN` |
