PhysLiteBench: unified study figures

This bundle reproduces the presentation charts from archived result files. It groups
results by exact model version, reasoning effort and prompt condition. The identity
of the person or machine running a batch is not a plotted grouping variable.

Reproduce
1. Use Python 3 with matplotlib and numpy. The tested versions are in requirements.txt.
2. Run: python aggregate_results.py
3. Run: python plot_results.py
The scripts resolve paths relative to themselves. They make no network or model calls.
Times New Roman is used if installed. The script registers the macOS system font;
on another system, install that font or explicitly select an available substitute.
All charts are saved as PNG, vector PDF and SVG in figures/.

Sources and records
source-manifest.json records source paths, pinned GitHub URLs and SHA-256 hashes.
The four hint-matrix-* files were downloaded from GitHub commit
c925ddf30d7729a629125f942bab67d9edc53f1f and matched the earlier archive byte for byte.
Those files contain published aggregates and verification metadata, not raw sessions.
All other experiment inputs were copied from the existing saved result archives.
The presentation's previous chart workbooks are preserved separately alongside the
source deck in the presentation revision archive.

summary.csv contains one row per exact model/effort/condition setting.
summary.json also includes the contributing source components, count rules, cost
bounds and eligible frontier points. deduplicated-trials.csv contains the 1,288
available trial-level records. The remaining 780 observations are represented only
at the published aggregate level; no invented individual records are created.
deduplication-audit.json lists the 38 removed duplicate baseline records by identity.

Primary totals
13 tasks, 7 exact model versions, 48 model/effort/condition settings.
2,068 recorded attempts: 1,054 without hints, 507 general-hint and 507 physics-hint.
899 completed passes, 904 graded failures, 265 execution errors.
Three DeepSeek None pilot attempts are preserved in the source file but excluded
because only one task was attempted. Five pre-agent setup records are retained
outside the published 780-attempt hint matrix according to its original protocol.
No new experiments, regrading, replacement trials or hidden-test changes were made.

Aggregation and interpretation
- A pass requires reward 1 and no execution exception. Timeout checkpoints do not
  count as completed passes. Errors remain in denominators.
- Exact trial identifiers remove 38 known duplicate historical baselines. The single
  fresh Luna High baseline is preserved in addition to all 39 historical trials,
  giving 40 no-hint attempts: four on entropy-anomaly, three on each other task.
- Pool independent repetitions within model/effort/condition by summing passes and
  attempts, not by averaging percentages. Repeated settings have 78 attempts.
- GPT-5.6 and GPT-6 versions remain distinct. No model aliases are conflated.
- The published 780 hinted observations report distinct native sessions, but those
  identities and raw transcripts are not independently available in this bundle.
- Task-content and grading equivalence follow archived hash audits and published
  metadata. Client version, concurrency, CPU quotas and dates differ across batches.
  Baselines may predate hints. Hint lengths differ. These are descriptive pooled
  comparisons and do not isolate a causal hint effect.
- Setup policies differed: five pre-agent setup failures were replaced outside the
  published matrix; the later setup failure and administrative interruptions remain
  in their original outcomes. The source records and denominators remain intact.

Cost and plotting
Costs are frozen Standard API-equivalent estimates, not subscription charges or a
current price quotation. They account for input cache discounts and per-request
long-context factors. Local compute is excluded. Prime accounting bounds are
retained, and conservative upper reserves must not be called confirmed spending.
A setting enters an exact frontier only when every contributing attempt has a
complete estimated cost. Missing cost is not zero. The scripts do not discard a
failed attempt to make a cost estimate complete. All outcomes remain in summary.csv.

01-no-hint-frontier: 19 exact-cost points from all 24 no-hint settings.
02-hint-frontiers: 22 exact-cost points from 36 settings, restricted to the same 12
model/effort combinations tested in all three conditions. Other outcomes remain in
the summary and backup tables. Curves show observed nondominated points only.
03-luna-effort-hints: all six Luna 5.6 efforts, with actual pass/attempt labels.
04-outcomes-by-model: all 2,068 outcomes; these totals mix effort and prompt settings
and are not a controlled model ranking.
05-bead-covariance: exact analytic series extracted from the user-edited slide 16.
This is a formula comparison, not measured bead trajectories.
06-no-hint-efforts: all available no-hint efforts, with model held fixed per line.

Frontier cost axes are logarithmic. Lines connect observed values with straight
segments. There is no spline smoothing, polynomial fit or invented measurement.
The original source presentation and all edited slides retain Times New Roman.

Failure analysis on slide 9
The 186/191 prediction-only failures come from the reviewed 585-attempt no-hint
subset. The 229/315 physics/modeling failures come from the published hinted review
(the other categories are 54 mathematical/numerical and 32 implementation).
These subsets use different diagnostic taxonomies, so their percentages are not
subtracted or presented as a causal hint-effect estimate. An execution error alone
does not establish a physics mistake. Review sources are included in sources/.
