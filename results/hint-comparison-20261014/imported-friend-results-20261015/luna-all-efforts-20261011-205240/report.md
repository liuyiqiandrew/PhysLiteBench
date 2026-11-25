# GPT-5.6 Luna: all reasoning efforts

Complete: 234/234 trials finished on the 13 roster tasks at commit `34dedfb`. Models: gpt-5.6-luna. Efforts: none, low, medium, high, xhigh, max. Each combination runs 3 independent times per task, with no hints and no stopping after a pass.

[Trial data](trials.csv) · [Pareto summary](summary.csv) · [Per-request costs](requests.csv) · [Verification](verification.json) · [Protocol and source hashes](protocol.json)

## Model and effort results

A pass requires verifier reward 1 and no execution exception. Timeouts and other errors remain in the planned denominator and are reported separately. Costs include unsuccessful attempts with recorded usage. Agent execution exceptions have unknown full cost; recorded tokens and costs are lower bounds.

| Model | Effort | Finished | Passes | Timeouts | Other errors | Recorded tokens | Total API equivalent | Mean per trial | Recorded API equivalent |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-5.6-luna | none | 39/39 | 1 | 0 | 0 | 5,231,761 | $0.325781 | $0.008353 | $0.325781 |
| gpt-5.6-luna | low | 39/39 | 0 | 0 | 0 | 4,291,442 | $0.289999 | $0.007436 | $0.289999 |
| gpt-5.6-luna | medium | 39/39 | 0 | 0 | 0 | 7,823,950 | $0.526987 | $0.013512 | $0.526987 |
| gpt-5.6-luna | high | 39/39 | 4 | 0 | 1 | 14,677,409 | unknown | unknown | $1.026520 |
| gpt-5.6-luna | xhigh | 39/39 | 6 | 1 | 0 | 17,766,584 | unknown | unknown | $1.292186 |
| gpt-5.6-luna | max | 39/39 | 8 | 0 | 0 | 32,325,294 | $1.999129 | $0.051260 | $1.999129 |

## Interrupted runs

- `magnetic-tracer__NAbRPfe` (xhigh) reached the 600-second agent timeout.
- `rotating-reservoir__mTzBFYo` (high) ended after an API stream disconnected.

Both received verifier reward 0 and remain in the denominator. Their unfinished requests may have unrecorded usage, so full cost is unknown. They were not resampled or classified as ordinary physics failures.

## Task results

Cells show passes / finished trials; each cell is planned for 3 trials.

| Task | Revision | Luna none | Luna low | Luna medium | Luna high | Luna xhigh | Luna max |
|---|---:|---:|---:|---:|---:|---:|---:|
| entropy-anomaly | 9 | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 | 1/3 |
| finite-layer-phoresis | 2 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| hierarchical-glass-overlaps | 2 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| insulating-mhd-duct | 2 | 0/3 | 0/3 | 0/3 | 1/3 | 0/3 | 2/3 |
| magnetic-tracer | 9 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| reaction-diffusion | 10 | 1/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| ring-current-fluctuations | 1 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| rotating-reservoir | 11 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| single-file-memory | 1 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 1/3 |
| superconducting-heat | 11 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| swirling-heat-tails | 1 | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 | 1/3 |
| hard-rod-current | 2 | 0/3 | 0/3 | 0/3 | 1/3 | 1/3 | 1/3 |
| finite-ion-cloud | 3 | 0/3 | 0/3 | 0/3 | 2/3 | 1/3 | 2/3 |

## Token accounting and API equivalent cost

Token counts come from native Codex session logs. Each unique cumulative token event contributes its last-request usage once; sums are checked against the final session totals. Input includes cached reads and cache writes. Output already includes reasoning tokens. Total tokens are input plus output, without adding caches or reasoning twice. Missing or inconsistent usage remains unknown. For interrupted sessions, reconciled records can still omit an unfinished request; full cost remains unknown and recorded usage is a lower bound. Harbor's own estimates remain in the local detailed records.

Standard USD rates per million tokens, verified on 2026-10-11:

| Model | Ordinary input | Cached input | Cache write | Output including reasoning |
|---|---:|---:|---:|---:|
| [gpt-5.6-luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) | $0.20 | $0.02 | $0.25 | $1.20 |

For each request, charge ordinary input = total input minus cache reads minus cache writes. Requests above 272,000 input tokens use twice the input/cache rates and 1.5 times the output rate for the whole request. Apply the threshold to each request, not trial totals. [Frozen pricing](pricing.json) records the rates and source notes.

These are API-equivalent estimates for OAuth-authenticated runs, not measured subscription charges. Local compute is excluded. Actual service tier and subscription billing are not inferred.

## Protocol and later Pareto analysis

Harbor 0.21.0, Codex CLI 0.154.0, 4 concurrent trials, 600 seconds to solve and 60 seconds to verify. Agent setup receives a separate 3.0× multiplier. Web search is disabled and agent networking is limited to OpenAI authentication/API hosts. Runtime task configuration adds only that network policy; all instructions, public files and private grading are frozen. Credentials stay outside the repository.

Plot each row of `summary.csv` using `mean_api_cost_usd_per_trial` against `pass_rate`. All 6 panels use the same 13 tasks with equal repetition. Use `trials.csv` for task-clustered uncertainty estimates and `requests.csv` to reprice later. Three trials per task give limited precision. Compare with historical panels only after matching task hashes, hint policy, model version and runtime limits. Timeouts and access errors measure end-to-end performance, not physics knowledge alone.

Published files contain scores, token accounting, API-equivalent costs, frozen pricing, task hashes and verification records. Raw transcripts and submitted files remain local in `jobs/luna-all-efforts-20261011-205240`; detailed evidence and full result JSON also remain local. The `evidence` column in `trials.csv` and the `source`/`line` columns in `requests.csv` identify those local records. Each session is fresh; no scored outcome is resampled.

## Individual trials

| Task | Model | Effort | Repeat | Result | Input | Cached | Output | Reasoning included | API equivalent | Recorded cost | Trial ID |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| entropy-anomaly | luna | high | 1 | fail | 338,048 | 301,312 | 11,337 | 8,078 | $0.026978 | $0.026978 | `entropy-anomaly__JzJK9e6` |
| entropy-anomaly | luna | high | 2 | fail | 260,758 | 228,096 | 9,334 | 6,357 | $0.022295 | $0.022295 | `entropy-anomaly__6B9gFwF` |
| entropy-anomaly | luna | high | 3 | fail | 349,174 | 319,232 | 12,561 | 8,318 | $0.027446 | $0.027446 | `entropy-anomaly__GMygMPu` |
| entropy-anomaly | luna | low | 1 | fail | 93,059 | 75,008 | 1,613 | 285 | $0.007046 | $0.007046 | `entropy-anomaly__TokEY6Z` |
| entropy-anomaly | luna | low | 2 | fail | 97,257 | 87,296 | 1,523 | 139 | $0.005566 | $0.005566 | `entropy-anomaly__G5qGSzf` |
| entropy-anomaly | luna | low | 3 | fail | 97,207 | 85,248 | 1,088 | 166 | $0.005402 | $0.005402 | `entropy-anomaly__t2yopNc` |
| entropy-anomaly | luna | max | 1 | fail | 1,110,390 | 1,037,056 | 33,855 | 27,272 | $0.076034 | $0.076034 | `entropy-anomaly__8eSYdCw` |
| entropy-anomaly | luna | max | 2 | pass | 961,660 | 892,416 | 30,249 | 24,004 | $0.067996 | $0.067996 | `entropy-anomaly__2vfMbey` |
| entropy-anomaly | luna | max | 3 | fail | 358,782 | 314,880 | 14,063 | 10,735 | $0.031954 | $0.031954 | `entropy-anomaly__5nJKcwg` |
| entropy-anomaly | luna | medium | 1 | fail | 183,414 | 160,000 | 3,782 | 1,553 | $0.012421 | $0.012421 | `entropy-anomaly__sLn7F9q` |
| entropy-anomaly | luna | medium | 2 | fail | 166,387 | 148,992 | 3,126 | 966 | $0.010210 | $0.010210 | `entropy-anomaly__PMnkGhe` |
| entropy-anomaly | luna | medium | 3 | fail | 272,930 | 249,088 | 5,300 | 2,330 | $0.016110 | $0.016110 | `entropy-anomaly__QzyhdNr` |
| entropy-anomaly | luna | none | 1 | fail | 83,697 | 64,256 | 1,505 | 0 | $0.006979 | $0.006979 | `entropy-anomaly__VHVENBX` |
| entropy-anomaly | luna | none | 2 | fail | 187,953 | 163,072 | 2,116 | 0 | $0.010777 | $0.010777 | `entropy-anomaly__hosudmH` |
| entropy-anomaly | luna | none | 3 | fail | 81,464 | 70,144 | 853 | 0 | $0.004690 | $0.004690 | `entropy-anomaly__SQpvGkR` |
| entropy-anomaly | luna | xhigh | 1 | pass | 1,790,139 | 1,712,896 | 30,801 | 20,360 | $0.086668 | $0.086668 | `entropy-anomaly__82me5YP` |
| entropy-anomaly | luna | xhigh | 2 | pass | 1,431,456 | 1,343,744 | 38,256 | 24,951 | $0.090324 | $0.090324 | `entropy-anomaly__sDR4fRB` |
| entropy-anomaly | luna | xhigh | 3 | fail | 362,190 | 327,936 | 15,795 | 12,764 | $0.032364 | $0.032364 | `entropy-anomaly__cHq9Yn8` |
| finite-layer-phoresis | luna | high | 1 | fail | 127,116 | 97,280 | 2,921 | 1,092 | $0.011418 | $0.011418 | `finite-layer-phoresis__ozXvx5Z` |
| finite-layer-phoresis | luna | high | 2 | fail | 138,411 | 122,624 | 2,835 | 929 | $0.009012 | $0.009012 | `finite-layer-phoresis__XyWWqiU` |
| finite-layer-phoresis | luna | high | 3 | fail | 217,791 | 185,856 | 3,928 | 1,808 | $0.014818 | $0.014818 | `finite-layer-phoresis__pXVsvSH` |
| finite-layer-phoresis | luna | low | 1 | fail | 67,960 | 59,904 | 1,227 | 219 | $0.004282 | $0.004282 | `finite-layer-phoresis__yow49xw` |
| finite-layer-phoresis | luna | low | 2 | fail | 66,500 | 58,880 | 1,274 | 150 | $0.004230 | $0.004230 | `finite-layer-phoresis__pXdTS7f` |
| finite-layer-phoresis | luna | low | 3 | fail | 77,991 | 71,936 | 1,590 | 268 | $0.004558 | $0.004558 | `finite-layer-phoresis__XdCSxmf` |
| finite-layer-phoresis | luna | max | 1 | fail | 131,672 | 108,544 | 4,367 | 1,859 | $0.012037 | $0.012037 | `finite-layer-phoresis__6QVnbEa` |
| finite-layer-phoresis | luna | max | 2 | fail | 165,328 | 138,496 | 6,260 | 4,390 | $0.015648 | $0.015648 | `finite-layer-phoresis__WCogezW` |
| finite-layer-phoresis | luna | max | 3 | fail | 186,114 | 166,912 | 4,623 | 2,142 | $0.012726 | $0.012726 | `finite-layer-phoresis__EkhH4VP` |
| finite-layer-phoresis | luna | medium | 1 | fail | 88,322 | 71,168 | 1,952 | 524 | $0.007197 | $0.007197 | `finite-layer-phoresis__QSGzF42` |
| finite-layer-phoresis | luna | medium | 2 | fail | 122,182 | 104,192 | 2,424 | 675 | $0.008591 | $0.008591 | `finite-layer-phoresis__BGsRomU` |
| finite-layer-phoresis | luna | medium | 3 | fail | 116,670 | 94,464 | 2,157 | 605 | $0.008919 | $0.008919 | `finite-layer-phoresis__EwkXeBD` |
| finite-layer-phoresis | luna | none | 1 | fail | 145,339 | 123,648 | 1,522 | 0 | $0.008638 | $0.008638 | `finite-layer-phoresis__arUetjj` |
| finite-layer-phoresis | luna | none | 2 | fail | 72,638 | 56,064 | 1,279 | 0 | $0.005971 | $0.005971 | `finite-layer-phoresis__JAk2B5s` |
| finite-layer-phoresis | luna | none | 3 | fail | 107,014 | 88,576 | 1,175 | 0 | $0.006869 | $0.006869 | `finite-layer-phoresis__kPJCZgp` |
| finite-layer-phoresis | luna | xhigh | 1 | fail | 188,178 | 157,184 | 5,541 | 3,190 | $0.015992 | $0.015992 | `finite-layer-phoresis__erQxMkF` |
| finite-layer-phoresis | luna | xhigh | 2 | fail | 155,169 | 137,728 | 5,130 | 2,517 | $0.012399 | $0.012399 | `finite-layer-phoresis__a92CReT` |
| finite-layer-phoresis | luna | xhigh | 3 | fail | 143,378 | 124,416 | 5,825 | 3,420 | $0.013271 | $0.013271 | `finite-layer-phoresis__KYhHczU` |
| hierarchical-glass-overlaps | luna | high | 1 | fail | 92,309 | 71,168 | 6,004 | 4,443 | $0.012856 | $0.012856 | `hierarchical-glass-overlaps__dJeuNxE` |
| hierarchical-glass-overlaps | luna | high | 2 | fail | 125,264 | 100,608 | 5,244 | 3,398 | $0.013236 | $0.013236 | `hierarchical-glass-overlaps__AE5MVUJ` |
| hierarchical-glass-overlaps | luna | high | 3 | fail | 157,949 | 135,168 | 4,872 | 3,153 | $0.013106 | $0.013106 | `hierarchical-glass-overlaps__5EccBP4` |
| hierarchical-glass-overlaps | luna | low | 1 | fail | 80,349 | 69,120 | 2,080 | 768 | $0.006124 | $0.006124 | `hierarchical-glass-overlaps__i9dKcqd` |
| hierarchical-glass-overlaps | luna | low | 2 | fail | 77,960 | 68,096 | 2,232 | 601 | $0.006013 | $0.006013 | `hierarchical-glass-overlaps__m5TawPR` |
| hierarchical-glass-overlaps | luna | low | 3 | fail | 95,185 | 58,112 | 1,967 | 515 | $0.010937 | $0.010937 | `hierarchical-glass-overlaps__zxaa4SH` |
| hierarchical-glass-overlaps | luna | max | 1 | fail | 136,913 | 109,824 | 8,333 | 6,481 | $0.017614 | $0.017614 | `hierarchical-glass-overlaps__9mSpLZA` |
| hierarchical-glass-overlaps | luna | max | 2 | fail | 201,824 | 171,520 | 8,884 | 6,357 | $0.020152 | $0.020152 | `hierarchical-glass-overlaps__jjEN8sq` |
| hierarchical-glass-overlaps | luna | max | 3 | fail | 167,012 | 136,192 | 9,901 | 8,072 | $0.020769 | $0.020769 | `hierarchical-glass-overlaps__JRyS3Qq` |
| hierarchical-glass-overlaps | luna | medium | 1 | fail | 108,661 | 81,664 | 3,105 | 1,701 | $0.010759 | $0.010759 | `hierarchical-glass-overlaps__mHoTXFh` |
| hierarchical-glass-overlaps | luna | medium | 2 | fail | 103,799 | 88,320 | 3,666 | 1,936 | $0.009261 | $0.009261 | `hierarchical-glass-overlaps__E86awh6` |
| hierarchical-glass-overlaps | luna | medium | 3 | fail | 159,406 | 136,192 | 3,448 | 1,886 | $0.011504 | $0.011504 | `hierarchical-glass-overlaps__pPuNB5m` |
| hierarchical-glass-overlaps | luna | none | 1 | fail | 95,571 | 77,312 | 1,698 | 0 | $0.007236 | $0.007236 | `hierarchical-glass-overlaps__EvxEcVk` |
| hierarchical-glass-overlaps | luna | none | 2 | fail | 77,879 | 61,184 | 1,358 | 0 | $0.006192 | $0.006192 | `hierarchical-glass-overlaps__U6JT4Yv` |
| hierarchical-glass-overlaps | luna | none | 3 | fail | 112,459 | 94,720 | 1,034 | 0 | $0.006683 | $0.006683 | `hierarchical-glass-overlaps__uWXpfSD` |
| hierarchical-glass-overlaps | luna | xhigh | 1 | fail | 106,953 | 90,368 | 6,586 | 4,964 | $0.013028 | $0.013028 | `hierarchical-glass-overlaps__7YJKdqB` |
| hierarchical-glass-overlaps | luna | xhigh | 2 | fail | 169,287 | 109,056 | 7,891 | 5,435 | $0.023697 | $0.023697 | `hierarchical-glass-overlaps__gAWTvVY` |
| hierarchical-glass-overlaps | luna | xhigh | 3 | fail | 119,108 | 101,632 | 6,981 | 4,601 | $0.013905 | $0.013905 | `hierarchical-glass-overlaps__iYhywcA` |
| insulating-mhd-duct | luna | high | 1 | fail | 191,276 | 169,472 | 5,002 | 2,924 | $0.013753 | $0.013753 | `insulating-mhd-duct__qo3R4Eb` |
| insulating-mhd-duct | luna | high | 2 | pass | 390,983 | 334,848 | 15,044 | 11,370 | $0.035977 | $0.035977 | `insulating-mhd-duct__h58BEkF` |
| insulating-mhd-duct | luna | high | 3 | fail | 388,621 | 353,280 | 13,116 | 7,868 | $0.029873 | $0.029873 | `insulating-mhd-duct__TX9fzy8` |
| insulating-mhd-duct | luna | low | 1 | fail | 71,798 | 61,952 | 1,722 | 473 | $0.005275 | $0.005275 | `insulating-mhd-duct__fN4V6Jb` |
| insulating-mhd-duct | luna | low | 2 | fail | 72,120 | 64,000 | 1,743 | 397 | $0.004996 | $0.004996 | `insulating-mhd-duct__s9N3A6i` |
| insulating-mhd-duct | luna | low | 3 | fail | 130,383 | 109,056 | 897 | 128 | $0.007523 | $0.007523 | `insulating-mhd-duct__ghqJ64f` |
| insulating-mhd-duct | luna | max | 1 | pass | 571,092 | 496,896 | 27,616 | 22,165 | $0.057916 | $0.057916 | `insulating-mhd-duct__nLcgpjv` |
| insulating-mhd-duct | luna | max | 2 | pass | 401,480 | 353,280 | 17,559 | 14,093 | $0.037776 | $0.037776 | `insulating-mhd-duct__Zbe4KAk` |
| insulating-mhd-duct | luna | max | 3 | fail | 159,084 | 125,696 | 5,511 | 3,302 | $0.015805 | $0.015805 | `insulating-mhd-duct__zRfYMBJ` |
| insulating-mhd-duct | luna | medium | 1 | fail | 149,047 | 127,744 | 3,361 | 1,612 | $0.010849 | $0.010849 | `insulating-mhd-duct__rWMdTRy` |
| insulating-mhd-duct | luna | medium | 2 | fail | 100,956 | 89,088 | 2,398 | 987 | $0.007033 | $0.007033 | `insulating-mhd-duct__ZhKh7Z4` |
| insulating-mhd-duct | luna | medium | 3 | fail | 163,582 | 146,688 | 3,255 | 1,966 | $0.010219 | $0.010219 | `insulating-mhd-duct__xibS6xA` |
| insulating-mhd-duct | luna | none | 1 | fail | 123,940 | 102,400 | 1,346 | 0 | $0.007971 | $0.007971 | `insulating-mhd-duct__brPkSkJ` |
| insulating-mhd-duct | luna | none | 2 | fail | 164,438 | 139,008 | 1,667 | 0 | $0.009867 | $0.009867 | `insulating-mhd-duct__NX34XEq` |
| insulating-mhd-duct | luna | none | 3 | fail | 82,244 | 75,008 | 1,864 | 0 | $0.005184 | $0.005184 | `insulating-mhd-duct__2dgapNu` |
| insulating-mhd-duct | luna | xhigh | 1 | fail | 255,233 | 227,072 | 9,081 | 6,163 | $0.021071 | $0.021071 | `insulating-mhd-duct__AL2m5pS` |
| insulating-mhd-duct | luna | xhigh | 2 | fail | 226,990 | 190,976 | 6,507 | 4,293 | $0.018831 | $0.018831 | `insulating-mhd-duct__JS9oVB8` |
| insulating-mhd-duct | luna | xhigh | 3 | fail | 162,326 | 139,008 | 11,619 | 9,797 | $0.021387 | $0.021387 | `insulating-mhd-duct__2kizsAC` |
| magnetic-tracer | luna | high | 1 | fail | 507,963 | 470,272 | 15,037 | 10,349 | $0.034988 | $0.034988 | `magnetic-tracer__3A5RsJg` |
| magnetic-tracer | luna | high | 2 | fail | 1,384,064 | 1,314,048 | 26,485 | 15,028 | $0.072066 | $0.072066 | `magnetic-tracer__reKWkzb` |
| magnetic-tracer | luna | high | 3 | fail | 1,014,762 | 949,248 | 24,949 | 12,141 | $0.062027 | $0.062027 | `magnetic-tracer__AtXYXRv` |
| magnetic-tracer | luna | low | 1 | fail | 97,968 | 83,456 | 1,321 | 312 | $0.006157 | $0.006157 | `magnetic-tracer__vMUejxe` |
| magnetic-tracer | luna | low | 2 | fail | 81,820 | 70,400 | 967 | 249 | $0.004852 | $0.004852 | `magnetic-tracer__6SerJ2t` |
| magnetic-tracer | luna | low | 3 | fail | 163,093 | 129,280 | 2,039 | 425 | $0.011795 | $0.011795 | `magnetic-tracer__B8ubzjY` |
| magnetic-tracer | luna | max | 1 | fail | 4,232,522 | 4,062,720 | 63,529 | 31,616 | $0.191450 | $0.191450 | `magnetic-tracer__v5UseEw` |
| magnetic-tracer | luna | max | 2 | fail | 2,139,163 | 2,036,992 | 40,944 | 31,037 | $0.110307 | $0.110307 | `magnetic-tracer__7d6LE4g` |
| magnetic-tracer | luna | max | 3 | fail | 1,296,467 | 1,227,264 | 29,809 | 21,361 | $0.074157 | $0.074157 | `magnetic-tracer__fUvntSG` |
| magnetic-tracer | luna | medium | 1 | fail | 172,010 | 150,272 | 2,722 | 1,368 | $0.010619 | $0.010619 | `magnetic-tracer__pSUhgvF` |
| magnetic-tracer | luna | medium | 2 | fail | 306,635 | 281,856 | 6,150 | 3,518 | $0.017973 | $0.017973 | `magnetic-tracer__6naf9tK` |
| magnetic-tracer | luna | medium | 3 | fail | 280,040 | 256,512 | 5,157 | 2,254 | $0.016024 | $0.016024 | `magnetic-tracer__aSP7MQS` |
| magnetic-tracer | luna | none | 1 | fail | 99,765 | 78,592 | 769 | 0 | $0.006729 | $0.006729 | `magnetic-tracer__rHi6Pdb` |
| magnetic-tracer | luna | none | 2 | fail | 81,524 | 67,328 | 871 | 0 | $0.005231 | $0.005231 | `magnetic-tracer__VRKQMMt` |
| magnetic-tracer | luna | none | 3 | fail | 84,050 | 71,424 | 686 | 0 | $0.004777 | $0.004777 | `magnetic-tracer__Jk9yPeQ` |
| magnetic-tracer | luna | xhigh | 1 | timeout | 458,065 | 400,896 | 26,122 | 15,866 | unknown | $0.050798 | `magnetic-tracer__NAbRPfe` |
| magnetic-tracer | luna | xhigh | 2 | fail | 1,153,750 | 1,090,304 | 26,727 | 17,587 | $0.066568 | $0.066568 | `magnetic-tracer__cqfgYzY` |
| magnetic-tracer | luna | xhigh | 3 | fail | 987,892 | 930,560 | 25,016 | 17,222 | $0.060097 | $0.060097 | `magnetic-tracer__PmKTyNj` |
| reaction-diffusion | luna | high | 1 | fail | 249,264 | 215,040 | 12,528 | 3,300 | $0.026179 | $0.026179 | `reaction-diffusion__ZkonCNi` |
| reaction-diffusion | luna | high | 2 | fail | 538,318 | 474,880 | 14,951 | 9,002 | $0.040126 | $0.040126 | `reaction-diffusion__tiv5k5X` |
| reaction-diffusion | luna | high | 3 | fail | 197,845 | 155,136 | 10,043 | 5,798 | $0.023696 | $0.023696 | `reaction-diffusion__7dcJRSj` |
| reaction-diffusion | luna | low | 1 | fail | 102,288 | 90,112 | 2,164 | 592 | $0.006834 | $0.006834 | `reaction-diffusion__H2iigj8` |
| reaction-diffusion | luna | low | 2 | fail | 76,217 | 55,040 | 1,555 | 511 | $0.007202 | $0.007202 | `reaction-diffusion__RU3eMKh` |
| reaction-diffusion | luna | low | 3 | fail | 103,680 | 79,872 | 2,310 | 677 | $0.009131 | $0.009131 | `reaction-diffusion__g8S7KC2` |
| reaction-diffusion | luna | max | 1 | fail | 1,360,512 | 1,280,768 | 32,666 | 17,928 | $0.080763 | $0.080763 | `reaction-diffusion__z3stLCU` |
| reaction-diffusion | luna | max | 2 | fail | 749,869 | 699,392 | 27,053 | 13,613 | $0.056547 | $0.056547 | `reaction-diffusion__wjWSKsf` |
| reaction-diffusion | luna | max | 3 | fail | 649,816 | 590,080 | 25,736 | 12,780 | $0.054632 | $0.054632 | `reaction-diffusion__jxxqiMP` |
| reaction-diffusion | luna | medium | 1 | fail | 225,875 | 188,416 | 4,266 | 1,553 | $0.016379 | $0.016379 | `reaction-diffusion__hbaCAer` |
| reaction-diffusion | luna | medium | 2 | fail | 293,829 | 266,240 | 6,607 | 2,692 | $0.018771 | $0.018771 | `reaction-diffusion__nYp9KaR` |
| reaction-diffusion | luna | medium | 3 | fail | 266,826 | 233,728 | 6,642 | 2,965 | $0.019265 | $0.019265 | `reaction-diffusion__Emsjfu7` |
| reaction-diffusion | luna | none | 1 | pass | 409,189 | 372,736 | 5,967 | 0 | $0.021906 | $0.021906 | `reaction-diffusion__YTL66uH` |
| reaction-diffusion | luna | none | 2 | fail | 159,349 | 144,640 | 1,845 | 0 | $0.008049 | $0.008049 | `reaction-diffusion__mJKfgGJ` |
| reaction-diffusion | luna | none | 3 | fail | 62,711 | 52,992 | 2,840 | 0 | $0.006412 | $0.006412 | `reaction-diffusion__Pibubxm` |
| reaction-diffusion | luna | xhigh | 1 | fail | 556,514 | 513,536 | 17,951 | 8,661 | $0.040408 | $0.040408 | `reaction-diffusion__8vgXiK7` |
| reaction-diffusion | luna | xhigh | 2 | fail | 414,104 | 372,992 | 20,898 | 9,772 | $0.040760 | $0.040760 | `reaction-diffusion__XfNEhtn` |
| reaction-diffusion | luna | xhigh | 3 | fail | 803,065 | 744,704 | 23,404 | 10,657 | $0.054651 | $0.054651 | `reaction-diffusion__2omXpDQ` |
| ring-current-fluctuations | luna | high | 1 | fail | 136,301 | 112,640 | 6,246 | 4,185 | $0.014480 | $0.014480 | `ring-current-fluctuations__3UPgEAL` |
| ring-current-fluctuations | luna | high | 2 | fail | 146,434 | 131,584 | 5,019 | 2,347 | $0.011624 | $0.011624 | `ring-current-fluctuations__Sxrg8CY` |
| ring-current-fluctuations | luna | high | 3 | fail | 188,449 | 168,192 | 8,488 | 5,683 | $0.017601 | $0.017601 | `ring-current-fluctuations__cTpgfPd` |
| ring-current-fluctuations | luna | low | 1 | fail | 105,908 | 88,576 | 1,862 | 719 | $0.007472 | $0.007472 | `ring-current-fluctuations__Hr2Qf4m` |
| ring-current-fluctuations | luna | low | 2 | fail | 72,414 | 62,976 | 1,506 | 287 | $0.004954 | $0.004954 | `ring-current-fluctuations__zcoKqjj` |
| ring-current-fluctuations | luna | low | 3 | fail | 110,190 | 91,648 | 3,024 | 2,111 | $0.009170 | $0.009170 | `ring-current-fluctuations__eACdQ2i` |
| ring-current-fluctuations | luna | max | 1 | fail | 110,731 | 89,344 | 5,370 | 3,410 | $0.012508 | $0.012508 | `ring-current-fluctuations__gNBRLP4` |
| ring-current-fluctuations | luna | max | 2 | fail | 151,965 | 130,816 | 8,660 | 6,441 | $0.017238 | $0.017238 | `ring-current-fluctuations__eJ7Zu6H` |
| ring-current-fluctuations | luna | max | 3 | fail | 129,015 | 114,432 | 3,870 | 1,791 | $0.009849 | $0.009849 | `ring-current-fluctuations__hhLe4n9` |
| ring-current-fluctuations | luna | medium | 1 | fail | 96,621 | 84,224 | 5,310 | 3,746 | $0.010536 | $0.010536 | `ring-current-fluctuations__XXdM6qL` |
| ring-current-fluctuations | luna | medium | 2 | fail | 106,839 | 88,064 | 3,342 | 933 | $0.009527 | $0.009527 | `ring-current-fluctuations__fGdqRz5` |
| ring-current-fluctuations | luna | medium | 3 | fail | 151,822 | 126,464 | 5,738 | 2,638 | $0.014486 | $0.014486 | `ring-current-fluctuations__ECueDw9` |
| ring-current-fluctuations | luna | none | 1 | fail | 202,198 | 177,664 | 2,357 | 0 | $0.011288 | $0.011288 | `ring-current-fluctuations__RGNd9dS` |
| ring-current-fluctuations | luna | none | 2 | fail | 73,005 | 66,048 | 1,097 | 0 | $0.004029 | $0.004029 | `ring-current-fluctuations__dkMmDob` |
| ring-current-fluctuations | luna | none | 3 | fail | 132,007 | 109,568 | 1,867 | 0 | $0.008920 | $0.008920 | `ring-current-fluctuations__nmXGg3F` |
| ring-current-fluctuations | luna | xhigh | 1 | fail | 113,497 | 82,432 | 4,864 | 2,921 | $0.013698 | $0.013698 | `ring-current-fluctuations__FeV6xuJ` |
| ring-current-fluctuations | luna | xhigh | 2 | fail | 93,543 | 75,264 | 3,129 | 1,334 | $0.008916 | $0.008916 | `ring-current-fluctuations__CT9qRuE` |
| ring-current-fluctuations | luna | xhigh | 3 | fail | 135,998 | 100,608 | 6,763 | 4,295 | $0.017206 | $0.017206 | `ring-current-fluctuations__zmXmz8M` |
| rotating-reservoir | luna | high | 1 | execution_error | 55,723 | 47,104 | 9,868 | 4,856 | unknown | $0.014507 | `rotating-reservoir__mTzBFYo` |
| rotating-reservoir | luna | high | 2 | fail | 1,008,139 | 936,960 | 22,871 | 9,380 | $0.060420 | $0.060420 | `rotating-reservoir__TtN87vP` |
| rotating-reservoir | luna | high | 3 | fail | 430,690 | 394,496 | 10,193 | 6,927 | $0.027360 | $0.027360 | `rotating-reservoir__6GPg3Mq` |
| rotating-reservoir | luna | low | 1 | fail | 88,782 | 69,120 | 1,814 | 521 | $0.007492 | $0.007492 | `rotating-reservoir__Vw7ab8B` |
| rotating-reservoir | luna | low | 2 | fail | 87,283 | 75,264 | 1,632 | 279 | $0.005867 | $0.005867 | `rotating-reservoir__dDPPFCk` |
| rotating-reservoir | luna | low | 3 | fail | 104,266 | 89,344 | 1,582 | 330 | $0.006670 | $0.006670 | `rotating-reservoir__XaUp5GT` |
| rotating-reservoir | luna | max | 1 | fail | 577,407 | 523,264 | 18,048 | 13,858 | $0.042951 | $0.042951 | `rotating-reservoir__tE78iXe` |
| rotating-reservoir | luna | max | 2 | fail | 2,327,453 | 2,219,264 | 48,049 | 28,201 | $0.123682 | $0.123682 | `rotating-reservoir__Lz7LSHr` |
| rotating-reservoir | luna | max | 3 | fail | 543,804 | 487,936 | 18,366 | 14,721 | $0.042972 | $0.042972 | `rotating-reservoir__dJkXwAk` |
| rotating-reservoir | luna | medium | 1 | fail | 209,949 | 186,624 | 8,235 | 2,690 | $0.018279 | $0.018279 | `rotating-reservoir__Z9Hv3X8` |
| rotating-reservoir | luna | medium | 2 | fail | 285,100 | 251,648 | 8,702 | 2,511 | $0.022166 | $0.022166 | `rotating-reservoir__YD3RewU` |
| rotating-reservoir | luna | medium | 3 | fail | 240,631 | 218,624 | 3,109 | 1,420 | $0.012505 | $0.012505 | `rotating-reservoir__wr249wx` |
| rotating-reservoir | luna | none | 1 | fail | 86,380 | 68,352 | 891 | 0 | $0.006042 | $0.006042 | `rotating-reservoir__P9sycCj` |
| rotating-reservoir | luna | none | 2 | fail | 107,576 | 85,504 | 1,234 | 0 | $0.007605 | $0.007605 | `rotating-reservoir__iWETspA` |
| rotating-reservoir | luna | none | 3 | fail | 100,468 | 90,368 | 953 | 0 | $0.004971 | $0.004971 | `rotating-reservoir__FMmJX2N` |
| rotating-reservoir | luna | xhigh | 1 | fail | 333,327 | 302,592 | 14,311 | 10,471 | $0.029372 | $0.029372 | `rotating-reservoir__VqZpZoy` |
| rotating-reservoir | luna | xhigh | 2 | fail | 614,863 | 565,504 | 26,501 | 12,650 | $0.052983 | $0.052983 | `rotating-reservoir__JTzhq5Y` |
| rotating-reservoir | luna | xhigh | 3 | fail | 460,863 | 398,080 | 13,152 | 9,419 | $0.036301 | $0.036301 | `rotating-reservoir__MmPMagN` |
| single-file-memory | luna | high | 1 | fail | 93,495 | 80,128 | 3,907 | 2,148 | $0.008964 | $0.008964 | `single-file-memory__mhwoESJ` |
| single-file-memory | luna | high | 2 | fail | 136,121 | 110,080 | 3,927 | 1,714 | $0.012122 | $0.012122 | `single-file-memory__L3nS5iR` |
| single-file-memory | luna | high | 3 | fail | 104,325 | 92,160 | 3,175 | 1,202 | $0.008086 | $0.008086 | `single-file-memory__i4TTRA7` |
| single-file-memory | luna | low | 1 | fail | 78,480 | 70,912 | 1,998 | 302 | $0.005329 | $0.005329 | `single-file-memory__vumzgoy` |
| single-file-memory | luna | low | 2 | fail | 65,150 | 59,904 | 1,479 | 325 | $0.004022 | $0.004022 | `single-file-memory__n789yGq` |
| single-file-memory | luna | low | 3 | fail | 104,559 | 83,712 | 1,769 | 537 | $0.007966 | $0.007966 | `single-file-memory__TFBpp3A` |
| single-file-memory | luna | max | 1 | pass | 236,551 | 203,776 | 11,937 | 8,341 | $0.024955 | $0.024955 | `single-file-memory__7KR5Ymy` |
| single-file-memory | luna | max | 2 | fail | 86,193 | 71,168 | 8,623 | 6,489 | $0.014776 | $0.014776 | `single-file-memory__kEfSFmw` |
| single-file-memory | luna | max | 3 | fail | 110,421 | 96,256 | 4,130 | 1,980 | $0.009714 | $0.009714 | `single-file-memory__cPh2yuZ` |
| single-file-memory | luna | medium | 1 | fail | 98,478 | 72,448 | 2,193 | 486 | $0.009287 | $0.009287 | `single-file-memory__YMtBq7t` |
| single-file-memory | luna | medium | 2 | fail | 134,205 | 101,120 | 2,095 | 624 | $0.011153 | $0.011153 | `single-file-memory__e3rgz8z` |
| single-file-memory | luna | medium | 3 | fail | 98,242 | 80,128 | 2,183 | 625 | $0.007845 | $0.007845 | `single-file-memory__xkFui8u` |
| single-file-memory | luna | none | 1 | fail | 130,671 | 99,072 | 1,341 | 0 | $0.009910 | $0.009910 | `single-file-memory__hYTAGRY` |
| single-file-memory | luna | none | 2 | fail | 65,067 | 49,920 | 1,455 | 0 | $0.005774 | $0.005774 | `single-file-memory__whFhtum` |
| single-file-memory | luna | none | 3 | fail | 65,151 | 57,856 | 1,243 | 0 | $0.004108 | $0.004108 | `single-file-memory__3n3FV5y` |
| single-file-memory | luna | xhigh | 1 | fail | 111,980 | 90,368 | 4,061 | 1,668 | $0.011003 | $0.011003 | `single-file-memory__zRgWqh3` |
| single-file-memory | luna | xhigh | 2 | fail | 115,493 | 101,376 | 5,555 | 3,429 | $0.011517 | $0.011517 | `single-file-memory__gWte7tR` |
| single-file-memory | luna | xhigh | 3 | fail | 89,783 | 78,080 | 3,477 | 1,473 | $0.008075 | $0.008075 | `single-file-memory__i2GJpYE` |
| superconducting-heat | luna | high | 1 | fail | 594,638 | 551,168 | 11,523 | 7,456 | $0.033545 | $0.033545 | `superconducting-heat__4Rtevrv` |
| superconducting-heat | luna | high | 2 | fail | 773,525 | 724,736 | 12,433 | 8,039 | $0.039172 | $0.039172 | `superconducting-heat__VgvFiLg` |
| superconducting-heat | luna | high | 3 | fail | 902,873 | 845,568 | 21,594 | 14,612 | $0.054285 | $0.054285 | `superconducting-heat__zEMw2fB` |
| superconducting-heat | luna | low | 1 | fail | 141,757 | 120,064 | 1,321 | 140 | $0.008325 | $0.008325 | `superconducting-heat__QirJhPK` |
| superconducting-heat | luna | low | 2 | fail | 243,311 | 216,320 | 2,984 | 1,153 | $0.013305 | $0.013305 | `superconducting-heat__ayeYidt` |
| superconducting-heat | luna | low | 3 | fail | 116,685 | 98,816 | 1,328 | 149 | $0.007144 | $0.007144 | `superconducting-heat__J3ExJZr` |
| superconducting-heat | luna | max | 1 | fail | 1,759,035 | 1,676,800 | 31,449 | 24,639 | $0.087722 | $0.087722 | `superconducting-heat__hAmZJT9` |
| superconducting-heat | luna | max | 2 | fail | 4,517,835 | 4,360,448 | 68,366 | 33,372 | $0.200726 | $0.200726 | `superconducting-heat__NSdCmQa` |
| superconducting-heat | luna | max | 3 | fail | 1,302,851 | 1,229,056 | 28,608 | 23,010 | $0.073670 | $0.073670 | `superconducting-heat__XDu9oRP` |
| superconducting-heat | luna | medium | 1 | fail | 453,360 | 391,936 | 5,408 | 2,054 | $0.026613 | $0.026613 | `superconducting-heat__DVJoQum` |
| superconducting-heat | luna | medium | 2 | fail | 385,898 | 353,536 | 4,338 | 1,559 | $0.018749 | $0.018749 | `superconducting-heat__MQmzsfF` |
| superconducting-heat | luna | medium | 3 | fail | 307,474 | 277,248 | 3,967 | 1,773 | $0.016351 | $0.016351 | `superconducting-heat__LLrY9B4` |
| superconducting-heat | luna | none | 1 | fail | 123,985 | 97,024 | 936 | 0 | $0.008456 | $0.008456 | `superconducting-heat__iReBsJJ` |
| superconducting-heat | luna | none | 2 | fail | 231,429 | 197,376 | 1,865 | 0 | $0.012996 | $0.012996 | `superconducting-heat__6pLiGBf` |
| superconducting-heat | luna | none | 3 | fail | 285,725 | 252,928 | 1,665 | 0 | $0.013616 | $0.013616 | `superconducting-heat__JTVh2cC` |
| superconducting-heat | luna | xhigh | 1 | fail | 1,214,155 | 1,145,344 | 23,867 | 19,036 | $0.065309 | $0.065309 | `superconducting-heat__9UV7rgh` |
| superconducting-heat | luna | xhigh | 2 | fail | 530,448 | 481,280 | 21,196 | 17,734 | $0.044894 | $0.044894 | `superconducting-heat__sJj36Dw` |
| superconducting-heat | luna | xhigh | 3 | fail | 1,039,509 | 969,472 | 20,483 | 15,974 | $0.057976 | $0.057976 | `superconducting-heat__V59ZC3M` |
| swirling-heat-tails | luna | high | 1 | fail | 159,480 | 142,848 | 4,850 | 2,544 | $0.012003 | $0.012003 | `swirling-heat-tails__CfpTqB9` |
| swirling-heat-tails | luna | high | 2 | fail | 203,836 | 162,304 | 5,766 | 3,147 | $0.018472 | $0.018472 | `swirling-heat-tails__P3nrFaa` |
| swirling-heat-tails | luna | high | 3 | fail | 123,427 | 108,288 | 4,243 | 1,855 | $0.010285 | $0.010285 | `swirling-heat-tails__vhtYurE` |
| swirling-heat-tails | luna | low | 1 | fail | 186,433 | 160,768 | 1,876 | 494 | $0.010600 | $0.010600 | `swirling-heat-tails__tKQNuCc` |
| swirling-heat-tails | luna | low | 2 | fail | 128,812 | 104,960 | 2,122 | 625 | $0.009416 | $0.009416 | `swirling-heat-tails__jup6RMk` |
| swirling-heat-tails | luna | low | 3 | fail | 128,382 | 96,000 | 1,731 | 340 | $0.010474 | $0.010474 | `swirling-heat-tails__EScNM7G` |
| swirling-heat-tails | luna | max | 1 | fail | 148,933 | 115,200 | 7,207 | 4,768 | $0.017699 | $0.017699 | `swirling-heat-tails__6GsfJcR` |
| swirling-heat-tails | luna | max | 2 | fail | 187,723 | 134,400 | 9,712 | 6,325 | $0.025007 | $0.025007 | `swirling-heat-tails__45DesA6` |
| swirling-heat-tails | luna | max | 3 | pass | 684,618 | 616,704 | 31,769 | 26,840 | $0.064040 | $0.064040 | `swirling-heat-tails__85uusrz` |
| swirling-heat-tails | luna | medium | 1 | fail | 106,633 | 89,344 | 2,949 | 1,269 | $0.008783 | $0.008783 | `swirling-heat-tails__hRHoWmh` |
| swirling-heat-tails | luna | medium | 2 | fail | 146,448 | 128,512 | 4,338 | 2,058 | $0.011363 | $0.011363 | `swirling-heat-tails__G9aq6Un` |
| swirling-heat-tails | luna | medium | 3 | fail | 233,904 | 201,472 | 5,361 | 3,427 | $0.016949 | $0.016949 | `swirling-heat-tails__tNYAitr` |
| swirling-heat-tails | luna | none | 1 | fail | 127,030 | 97,024 | 1,148 | 0 | $0.009319 | $0.009319 | `swirling-heat-tails__SYfSkud` |
| swirling-heat-tails | luna | none | 2 | fail | 127,772 | 104,960 | 1,258 | 0 | $0.008171 | $0.008171 | `swirling-heat-tails__rGM38nJ` |
| swirling-heat-tails | luna | none | 3 | fail | 156,690 | 128,256 | 1,475 | 0 | $0.010022 | $0.010022 | `swirling-heat-tails__FAqjVeg` |
| swirling-heat-tails | luna | xhigh | 1 | pass | 228,135 | 193,024 | 16,212 | 13,493 | $0.030337 | $0.030337 | `swirling-heat-tails__VzSKwQ8` |
| swirling-heat-tails | luna | xhigh | 2 | pass | 217,480 | 169,216 | 16,610 | 14,365 | $0.032969 | $0.032969 | `swirling-heat-tails__j89sUXD` |
| swirling-heat-tails | luna | xhigh | 3 | fail | 106,080 | 88,320 | 5,244 | 3,541 | $0.011611 | $0.011611 | `swirling-heat-tails__oL4G7wD` |
| hard-rod-current | luna | high | 1 | fail | 408,245 | 351,232 | 12,442 | 9,930 | $0.033358 | $0.033358 | `hard-rod-current__kELoAcm` |
| hard-rod-current | luna | high | 2 | fail | 233,767 | 198,912 | 6,158 | 4,021 | $0.018339 | $0.018339 | `hard-rod-current__cQHZMVw` |
| hard-rod-current | luna | high | 3 | pass | 337,456 | 290,560 | 18,566 | 14,786 | $0.037470 | $0.037470 | `hard-rod-current__m3wZ32L` |
| hard-rod-current | luna | low | 1 | fail | 143,663 | 123,136 | 2,335 | 641 | $0.009370 | $0.009370 | `hard-rod-current__gwBSkeL` |
| hard-rod-current | luna | low | 2 | fail | 105,656 | 94,208 | 1,893 | 457 | $0.006445 | $0.006445 | `hard-rod-current__ZpcNRWr` |
| hard-rod-current | luna | low | 3 | fail | 126,495 | 99,840 | 1,798 | 580 | $0.009485 | $0.009485 | `hard-rod-current__3DgLxeL` |
| hard-rod-current | luna | max | 1 | pass | 429,724 | 384,000 | 25,686 | 21,808 | $0.047648 | $0.047648 | `hard-rod-current__9p7qTS8` |
| hard-rod-current | luna | max | 2 | fail | 512,872 | 462,080 | 14,765 | 11,860 | $0.037118 | $0.037118 | `hard-rod-current__2mJpyaS` |
| hard-rod-current | luna | max | 3 | fail | 411,784 | 335,360 | 16,222 | 12,611 | $0.041458 | $0.041458 | `hard-rod-current__DcBP77V` |
| hard-rod-current | luna | medium | 1 | fail | 387,117 | 347,904 | 7,653 | 4,150 | $0.023984 | $0.023984 | `hard-rod-current__PiTLigw` |
| hard-rod-current | luna | medium | 2 | fail | 180,165 | 154,880 | 3,688 | 1,391 | $0.012580 | $0.012580 | `hard-rod-current__qRmQns6` |
| hard-rod-current | luna | medium | 3 | fail | 182,074 | 154,624 | 3,619 | 1,885 | $0.012925 | $0.012925 | `hard-rod-current__nQEU43n` |
| hard-rod-current | luna | none | 1 | fail | 212,599 | 185,088 | 2,506 | 0 | $0.012211 | $0.012211 | `hard-rod-current__tems4YD` |
| hard-rod-current | luna | none | 2 | fail | 183,392 | 150,784 | 2,118 | 0 | $0.012079 | $0.012079 | `hard-rod-current__rRnY8NU` |
| hard-rod-current | luna | none | 3 | fail | 155,369 | 131,328 | 1,814 | 0 | $0.009612 | $0.009612 | `hard-rod-current__arBy2tn` |
| hard-rod-current | luna | xhigh | 1 | fail | 261,945 | 225,024 | 8,947 | 6,419 | $0.022621 | $0.022621 | `hard-rod-current__p5XiHNn` |
| hard-rod-current | luna | xhigh | 2 | fail | 489,873 | 439,040 | 17,396 | 11,947 | $0.039823 | $0.039823 | `hard-rod-current__HMakZ5Z` |
| hard-rod-current | luna | xhigh | 3 | pass | 284,063 | 246,528 | 15,282 | 11,909 | $0.030776 | $0.030776 | `hard-rod-current__fXhg3yt` |
| finite-ion-cloud | luna | high | 1 | pass | 759,445 | 707,072 | 17,092 | 10,601 | $0.045126 | $0.045126 | `finite-ion-cloud__ibAFoJA` |
| finite-ion-cloud | luna | high | 2 | pass | 635,038 | 583,680 | 15,424 | 8,435 | $0.040454 | $0.040454 | `finite-ion-cloud__S3eVSwX` |
| finite-ion-cloud | luna | high | 3 | fail | 166,763 | 142,080 | 9,347 | 6,819 | $0.018995 | $0.018995 | `finite-ion-cloud__TzgyHEN` |
| finite-ion-cloud | luna | low | 1 | fail | 151,757 | 121,344 | 1,721 | 225 | $0.010575 | $0.010575 | `finite-ion-cloud__nHqiCmn` |
| finite-ion-cloud | luna | low | 2 | fail | 125,209 | 102,912 | 1,666 | 286 | $0.008517 | $0.008517 | `finite-ion-cloud__pzvShpk` |
| finite-ion-cloud | luna | low | 3 | fail | 155,063 | 130,304 | 1,599 | 247 | $0.009477 | $0.009477 | `finite-ion-cloud__evUnkTF` |
| finite-ion-cloud | luna | max | 1 | fail | 212,041 | 182,528 | 6,966 | 4,054 | $0.017912 | $0.017912 | `finite-ion-cloud__hTtBNwB` |
| finite-ion-cloud | luna | max | 2 | pass | 750,818 | 682,752 | 25,571 | 17,322 | $0.057953 | $0.057953 | `finite-ion-cloud__9jDUZb3` |
| finite-ion-cloud | luna | max | 3 | pass | 1,339,765 | 1,268,736 | 29,723 | 20,296 | $0.075248 | $0.075248 | `finite-ion-cloud__Lah9G5K` |
| finite-ion-cloud | luna | medium | 1 | fail | 200,562 | 156,160 | 3,242 | 1,622 | $0.015894 | $0.015894 | `finite-ion-cloud__JWYujga` |
| finite-ion-cloud | luna | medium | 2 | fail | 220,717 | 191,232 | 2,656 | 919 | $0.012909 | $0.012909 | `finite-ion-cloud__eEHoUk9` |
| finite-ion-cloud | luna | medium | 3 | fail | 156,494 | 127,232 | 3,002 | 1,495 | $0.011999 | $0.011999 | `finite-ion-cloud__FSrBLQJ` |
| finite-ion-cloud | luna | none | 1 | fail | 123,755 | 93,952 | 1,247 | 0 | $0.009336 | $0.009336 | `finite-ion-cloud__XsTmM2G` |
| finite-ion-cloud | luna | none | 2 | fail | 124,347 | 94,976 | 1,303 | 0 | $0.009337 | $0.009337 | `finite-ion-cloud__BaXZBfZ` |
| finite-ion-cloud | luna | none | 3 | fail | 124,561 | 102,912 | 1,192 | 0 | $0.007818 | $0.007818 | `finite-ion-cloud__EbE45L4` |
| finite-ion-cloud | luna | xhigh | 1 | pass | 843,289 | 775,424 | 22,085 | 14,538 | $0.055583 | $0.055583 | `finite-ion-cloud__bcr2wto` |
| finite-ion-cloud | luna | xhigh | 2 | fail | 202,185 | 176,384 | 6,133 | 3,838 | $0.016047 | $0.016047 | `finite-ion-cloud__cYCRp8T` |
| finite-ion-cloud | luna | xhigh | 3 | fail | 244,133 | 210,944 | 6,746 | 4,378 | $0.018952 | $0.018952 | `finite-ion-cloud__AkuzvNb` |

## Comparison and verification

[Combined Sol/Terra/Luna report](../sol-terra-low-medium-20261011-185155/report.md) · [Combined Pareto summary](../sol-terra-low-medium-20261011-185155/pareto-summary.csv) · [Final audit](verification.json) · [Effort compatibility checks](access-check.json)

Interrupted sessions may omit an unfinished request: their recorded tokens and costs are lower bounds, and full cost is unknown. The original ten tasks were selected using earlier Luna High failures. This sweep uses fresh sessions on that selected roster; it is not an independent held-out evaluation. Three runs per task give limited precision.

Six harmless compatibility checks verified the effort settings before the sweep. Their $0.015007 API-equivalent cost is recorded separately in the Luna access-check.json and excluded from the physics totals. Costs are estimates, not subscription charges.
