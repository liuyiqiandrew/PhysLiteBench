# Swirling heat tails r1: independent promotion review

Reviewed 2026-10-11. **Confirm promotion under the requested 0/3 physical-failure criterion.** The predeclared three unhinted gpt-5.6-luna High trials all completed, passed calibration and parameter recovery, and failed the three heat-tail prediction groups. This is one qualifying revision, not evidence of a universal model failure rate. No evaluation was rerun.

I read all three final sources, every emitted trajectory message and tool command, relevant observations, native result/configuration files, verifier output, and the preserved evidence. I verified 68 hash comparisons: the 16 planned files against both current and frozen copies, each evidence artifact against its recorded digest and original job artifact, the three trajectories, and unchanged public README/test/calibration artifacts. The instruction in each trajectory matches the frozen neutral instruction. The job confirms exactly three completed trials, no exceptions, no retries, and no cancellations. Both completed Docker controls match the plan: oracle 7/7, shortcut 4 passes and 3 prediction failures.

| Trial | Observable source/trajectory evidence | Classification |
|---|---|---|
| `SanWJTG` | Step 12 identifies the unrestricted square-root expression as the large-deviation answer; the final source applies it throughout the input domain. Steps 14–16 check finiteness and the zero at the mean and add a roundoff clamp. An earlier duplicated stiffness factor in scratch calibration work is corrected before implementation. | Physical failure: work tails used for heat. |
| `brLJQ9M` | Step 10 calls the same expression exact and implements its algebraically equivalent rationalized form. Final public tests pass. The rationalization changes numerical cancellation, not the counting-field domain. | Physical failure: work tails used for heat. |
| `uCXy9Js` | Step 9 calls the approximation the exact OU heat form. Its source explicitly labels the bulk square-root eigenvalue as the heat generating rate and takes its unrestricted transform. | Physical failure: work tails used for heat. |

The sources omit the stationary initial/final energy integrability constraints; none implements the restricted heat domain. This conclusion concerns their emitted implementation and stated physical identification, not unobservable private reasoning. The initial unfinished-fit test failures in two trajectories are resolved. Non-repository git commands are incidental. No numerical, fitting, interface, timeout, or infrastructure issue explains the final diagnostic pattern.

All three fit mobility `1.0799612406033`, with relative parameter error `3.589e-5` and reduced chi-square `.95008163`. Central-anchor error is `3.589e-5`. Their weak reverse, strong reverse, and forward-tail group errors are respectively about 37.06%, 11.96%, and 24.57%, against the 4% gate. The shared pattern matches the independently completed physical shortcut.

I independently repeated the response-only causal check using the explicit piecewise heat rate, rather than importing the oracle's clipped saddle. Each original model was fitted once. Replacing only its response function, while retaining its exact fitted mobility and prediction path, reduced every group error below `3.589e-5`. No data, parameter, interface, or tolerance was changed. This isolates the omitted heat-domain physics from fitting or coding effects.

The scientific apparatus and independent reference were approved separately in [the source review](expansion-swirling-heat-tails-r1-physics-review.md). Preserve this frozen batch and its full provenance; no unchanged resampling is needed.

Reviewed provenance:

- [Predeclared plan](expansion-swirling-heat-tails-r1-plan.json), SHA-256 `2c814751c4a3c8042f9d95e209264f987ff81c9c3abd6fc04fcb05b2a8f7f1c5`.
- [Results and artifact digests](expansion-swirling-heat-tails-r1-results.json), SHA-256 `3b843c33244e2982a94468a5854b6c9bfc4ed3d8cf941fd904cc32588792b49e`.
- Job `jobs/swirling-heat-tails-expansion-r1-plain-20261011-0815`; evidence under `results/expansion-swirling-heat-tails-r1-evidence/`.
