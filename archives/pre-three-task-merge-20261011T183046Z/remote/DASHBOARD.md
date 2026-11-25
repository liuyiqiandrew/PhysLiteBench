# PhysLiteBench evaluation dashboard

## Fresh Luna confirmation, 2026-10-11

The user requested three new independent unhinted `gpt-5.6-luna` High trials for each expansion task. All nine completed without infrastructure errors or retries, with all 49 task source files matching the original snapshots.

| Task | Fresh passes | Clean physical failures | Mixed failures | Original + fresh passes |
|---|---:|---:|---:|---:|
| Swirling heat tails r1 | 1/3 | 2 | 0 | 1/6 |
| Hard-rod current r2 | 0/3 | 3 | 0 | 0/6 |
| Finite-ion cloud r3 | 0/3 | 2 | 1 | 0/6 |

**The original 0/3 did not repeat for swirling heat tails.** Its passing solution correctly adds the stationary endpoint-energy branches. Seven failures are clean physical-model errors; one ion failure also introduces an incorrect radial exponent. Correcting either error alone in that mixed submission still fails; correcting both passes at the retained coupling. Only the rod batch repeats three clean physical failures. These fixed confirmation outcomes are separate from development selection. [Plan](results/expansion-confirmation-20261011-1709-plan.json), [all trial evidence](results/expansion-confirmation-20261011-1709-results.json) and [independent review](results/expansion-confirmation-20261011-1709-review.md).

**Original qualification: thirteen tasks had 0/3 completed unhinted Luna High passes each.** The [README roster](README.md) and [result manifest](results/qualified-tasks.json) identify all thirteen source snapshots and their trial IDs. Four tasks later had maintenance without new trials, mainly frozen hidden truth in their verifiers; see [post-qualification maintenance](docs/BENCHMARK.md#post-qualification-maintenance) and the manifest's `post_qualification_changes`.

The final hierarchical-glass-overlaps R2 batch contained CyG9A9V, VXrRYjC and sYzT2qQ. All failed for the same reviewed physical correlation assumption, with no infrastructure exceptions. Calibration, parameter recovery and parent-collision checks passed; state and joint-partition groups failed. A same-parameter correction of the descendant-selection law repaired every group. Oracle controls passed 9/9 tests; the complete approximation passed 4 and failed 5.

Scientific validation checked 2,052 domain outputs and 50 labelled-partition normalization cases against positive marked-Poisson integrals. Across 256 noise datasets, the worst correct-model group error was 0.2902%, below the 4% gate. The approximation missed a joint group by at least 48.69%. Full source/history and causal reviews confirmed the qualification.

R1 previously returned 1/3 passes and remains unqualified. Historical native jobs and extended review evidence are retained locally; the original publication did not rerun or resample these tasks.

## Original physics expansion, 2026-10-11

Swirling heat tails r1 passed 0/3 unhinted `gpt-5.6-luna` High trials with no infrastructure errors. All three fitted calibration and mobility correctly but used the unrestricted work rate for heat. The missing endpoint-energy constraints caused the three tail groups to fail by 12–37%; replacing only the response law at the original mobility repaired every group. The oracle passed 7/7 tests and the completed approximation passed four and failed the three intended predictions. [Frozen plan](results/expansion-swirling-heat-tails-r1-plan.json), [scientific review](results/expansion-swirling-heat-tails-r1-physics-review.md), and [per-trial evidence and repairs](results/expansion-swirling-heat-tails-r1-results.json) record provenance.

Hard-rod current r2 also passed 0/3 with no infrastructure exceptions. Every solution fitted speed correctly but asserted that only mean length affects the mixture current law. All three mixture groups failed by 38–52%; correcting the response at the retained speed reduced every error below .004%. Both controls had the same seven-test pattern as the heat task. [Frozen plan](results/expansion-hard-rod-current-r2-plan.json), [scientific review](results/expansion-hard-rod-current-r2-physics-review.md), and [per-trial evidence and repairs](results/expansion-hard-rod-current-r2-results.json) retain the source and trajectory review.

Finite-ion cloud r3 passed 0/3 with no infrastructure exceptions. All three fitted coupling correctly but treated independent pair-angle averages as the exact three-ion Gibbs response. The loop groups failed by 11–30%; restoring the shared-angle constraint at the exact retained coupling reduced every error below .034%. The oracle passed 7/7 and the completed approximation passed four and failed three. [Frozen plan](results/expansion-finite-ion-cloud-r3-plan.json), [scientific review](results/expansion-finite-ion-cloud-r3-physics-review.md), [trial evidence and repairs](results/expansion-finite-ion-cloud-r3-results.json), and [independent promotion review](results/expansion-finite-ion-cloud-r3-promotion-review.md) record the checks.

These complete all 3 requested additions: 0/9 passes with nine reviewed physical failures. The expansion evaluated 66 trials across 22 revisions. Seventeen excluded revisions passed 3/3 each; hard-rod current r1 and counterion cloud r2 each passed 1/3 with two reviewed physical failures. All are [archived](archives/physics-expansion-20261011/README.md); none was resampled unchanged. The unsupported `gpt-6-luna` access attempts are recorded separately from the user-authorized `gpt-5.6-luna` scientific trials.
