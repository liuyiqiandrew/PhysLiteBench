# Hard-rod-current revision 2: independent promotion review

Verdict: approve promotion on the stated criterion. The predeclared three plain trials of `gpt-5.6-luna` at high reasoning effort produced 0/3 passes and three genuine physical failures. This describes this batch, not an estimated universal failure rate. No task source was changed and no model trial was rerun for this review.

The reviewed job is `jobs/hard-rod-current-expansion-r2-plain-20261011-0840`. I read each final submitted model, all emitted trajectory messages and tool commands, native job configuration/results, and verifier output. I independently checked the result classifications and repaired the response using the independent reference. The scientific apparatus and reference were already approved in [the revision-2 source review](expansion-hard-rod-current-r2-physics-review.md).

## Integrity and completion

All 68 hash comparisons passed: the 16 planned files against both current and frozen copies; each trial's four preserved evidence files against their recorded digests and original artifacts; trajectory digests; and public README, test, and calibration artifacts against the frozen public inputs. Each trajectory received the exact frozen neutral instruction. The native job records three completed trials, zero errored trials, zero retries, and no trial exception. It used three concurrent plain trials, with hints and web access disabled, as predeclared. Both completed controls have their expected verifier outcomes: oracle 7/7; shortcut four passing checks and three prediction failures.

Every submitted model passed calibration, parameter recovery, the public check, and the shared prediction anchors. Every model failed only `mixture_fields`, `dense_mixtures`, and `moderate_mixtures`: respective maximum relative errors are about 52.31%, 50.31%, and 38.36%, against a 4% prediction gate. Fitted speed is approximately 1.1199586271, only 0.003694% below the generating speed, and reduced calibration chi-square is about 1.0945.

## Physical cause

All three retain the complete monodisperse current law at the ordinary mean rod length. This already includes current feedback that revision 1's shortcut omitted. The new mistake is treating the composition of the crossed block as typical even inside an exponential current average. A block of order `|j| t` rods can have an atypical fraction of short rods, at an entropy cost of the same order in time. Its composition must therefore be optimized jointly with current. Contrast-independent mean and variance do not imply a contrast-independent finite-field generating function.

| Trial suffix | Evidence in final source and emitted trajectory |
| --- | --- |
| `UrHaHmR` | Leaves the supplied monodisperse response intact. Step 16 interprets the mixture variance calibration as matching dependence on mean length alone; subsequent checks test numerical finiteness and calibration, without adding composition fluctuations. |
| `ZQ6i29B` | Final response docstring explicitly claims mixture contrast is immaterial in the thermodynamic limit. Steps 11 and 15 endorse the mean-length law after inspecting variance data. |
| `yLHP9LP` | Final response docstring and step 15 claim that random ordering and velocity exchange make mean length sufficient. The field root remains monodisperse. |

These are completed, nonzero, numerically stable physical closures. Incidental `git diff` failures outside a Git repository occur after successful public checks and do not affect predictions. In `yLHP9LP`, a scratch discussion miscounts calibration replicas, but the actual fit processes every record and recovers the same speed; that scratch error does not explain the failures.

## Independent response-only repair

I imported each exact preserved model, checked its fit against the original parameter, reproduced the reported prediction errors, and set its speed to the exact original verifier value. I then replaced only its global response function with the frozen reference's independent joint current/composition entropy contraction, evaluated at unit speed and tolerance `1e-14`. I retained the same model instance, prediction path, and recorded speed, without fitting again. This uses the contraction reference rather than copying the oracle's implicit field equation.

All four groups then have maximum relative error below `3.695e-5` in every trial. The largest repaired error is `3.694006013050e-5`, consistent with the unchanged speed estimation error. The response-only intervention establishes that the missing composition fluctuation accounts for the failed predictions. No mathematical, numerical, coding, or infrastructure defect is needed to explain them.

Revision 1's archived 1/3 result remains separate. Revision 2 changes the apparatus and missing physical effect, and this review does not pool the batches or treat revision 2 as an unchanged resample.

## Reviewed evidence hashes

The [result report](expansion-hard-rod-current-r2-results.json) preserves the per-trial model, metrics, reward, stdout, and trajectory digests. I verified all of them against the original job artifacts.

| File | SHA-256 |
| --- | --- |
| `expansion-hard-rod-current-r2-plan.json` | `c7393e7fd45bf4ed2560b767087f0761bed9c15f9ec0682ce269638af3e1f63a` |
| `expansion-hard-rod-current-r2-results.json` | `ba550b00e7b5ea4b6e9d9ce181a2536b4388a4b87f3907b4d5812a87c9954dd2` |
| `expansion-hard-rod-current-r2-physics-review.md` | `0c7ee5127a495fa1cff8a298718cd17c93a6b60d42e6a0356753ec3fa49e97ce` |
| Frozen/current `tests/reference.py` | `76e4605d9b7b09874aac61b2dc659c8558e0f574d854191d48a5fde856e56eac` |
| `UrHaHmR` submitted `model.py` | `85e3c34e5f7c555bb0506f19478c38c171402004289c013c9ea356d107e164cb` |
| `ZQ6i29B` submitted `model.py` | `b741db93cb359caf0d3d6cefbd0c43c907025eb3f143d06b84787709123b1a35` |
| `yLHP9LP` submitted `model.py` | `a28d6f4717ffdf0023a194fabfb0f9e7f3f6807b707ce7c6a191f0e30482aad1` |
