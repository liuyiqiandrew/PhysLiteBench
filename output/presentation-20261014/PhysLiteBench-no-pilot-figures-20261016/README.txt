Dollar and token cost presentation revision
Slide 8 uses one 39-attempt panel per cell; all other plots retain the full study.
The previous version and its pooled figure remain unchanged in the prior bundle.

PhysLiteBench: unified study figures

This bundle reproduces the presentation charts from archived result files. It groups
results by exact model version, reasoning effort and prompt condition. The identity
of the person or machine running a batch is not a plotted grouping variable.

Reproduce
1. Use Python 3 with matplotlib and numpy. The tested versions are in requirements.txt.
2. Run: python aggregate_results.py
3. Run: python plot_results.py
4. Run: python analyze_failures.py
5. Run: python plot_failures.py
6. Run: python aggregate_resources.py
7. Run: python plot_resources.py
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
03-luna-effort-hints: one six-effort Luna 5.6 matrix with 39 attempts per cell.
At High, the labels are 4/39 no hint, 18/39 general, and 30/39 physics. The later
independent High replication is not pooled into this figure. The full study summary
and backup tables retain all original attempts. See luna-effort-panel.json for the
exact selection and baseline identities. This figure uses historical controls.
04-outcomes-by-model: preserved earlier figure of all 2,068 outcomes. It mixes effort
and prompt settings and is no longer used on slide 9 of the updated presentation.
05-bead-covariance: exact analytic series extracted from the user-edited slide 16.
This is a formula comparison, not measured bead trajectories.
06-no-hint-efforts: all available no-hint efforts, with model held fixed per line.

Frontier cost axes are logarithmic. Lines connect observed values with straight
segments. There is no spline smoothing, polynomial fit or invented measurement.
The original source presentation and all edited slides retain Times New Roman.

Updated failure analysis on slide 9
07-reviewed-outcomes shows five mutually exclusive outcomes across the same ten
GPT-5.6 model/effort settings: Luna None, Low, Medium, High, Extra High and Max;
Terra Low and Medium; Sol Low and Medium. Each condition has 390 attempts.
08-physics-failure-matrix shows the strict physics-failure counts by model and
reasoning effort, with 39 attempts per condition/cell. It appears on backup slide 18.
Backup slide 19 provides all five outcome totals and the classification definition.

The outcome-breakdown.md report was pinned at commit
f1164fe2b2bc8d312733bedbfc633b48e3b3cb36. Its associated summary.csv, summary.md and
verification.json matched the previously archived c925ddf files byte for byte.
This is a classification update to 1,170 existing observations, not new trials.
All 390 baseline identities match the deduplicated study records. The 780 hinted
outcomes match the unchanged published aggregate summary. The full study remains
2,068 unique attempts, with zero additional trials. No classification is imputed
to the remaining 898 study attempts. Slide 8 remains unchanged at 39 per cell.

Physics failures require a normally completed reward-0 attempt, successful
calibration and parameter recovery, at least one failed hidden prediction, and a
reviewed primary physical-model error. Public-test failure prevents this label.
Mathematical, numerical or coding primaries remain other graded failures, even
when a secondary physics problem occurs. Exceptions take priority over grades.
A timeout or transport failure does not establish a physics mistake.

The report's 315 hinted graded failures reuse full observable-trajectory and code
reviews. Its 304 historical no-hint graded failures received targeted reviews.
The underlying raw trajectories and unpublished classification ledger were not
independently reviewed for this presentation revision. The ledger's reported
SHA-256 is b9faeffe2d4db9c983eab440f52b0eb19cf9222ead0e142fd0c9dfb2a2e5faca.
The report says 4,686 original artifact hashes were checked. Five replaced pre-agent
setup failures remain outside the original matrix denominator. No grades changed.

Earlier no-hint controls used four workers without newer per-container CPU quotas;
hint runs used up to eight workers capped at 8.8 CPU cores including proxies.
These descriptive comparisons do not isolate a causal effect of hints. They also
do not establish that particular failed attempts turned into passes.

Strict physics failures: 296/390 without hints, 158/390 with a general hint and
66/390 with a physics hint. Luna High: 33, 12 and 2 out of 39. Of the 66 remaining
physics failures under physics hints, 56 occur in Luna None, Low or Medium.
The former 186/191 calibration-fitting and 229/315 primary-cause counts used
different scopes and taxonomies. They are retained in original source archives but
are not pooled with or subtracted from the new strict physics-failure counts.

All-cost update to slides 6 and 7
09-no-hint-all-costs and 10-hint-all-costs contain two panels: recorded dollar cost
and recorded token cost. Both x-axes are logarithmic. Standalone enlarged plots
are 11-no-hint-usd, 11-no-hint-tokens, 12-hint-usd and 12-hint-tokens.
Earlier 01/02 figures are retained as the previous complete-cost-only versions,
but the new presentation uses 09/10. Do not confuse their different inclusion rules.

Every primary setting appears: 24 no-hint settings and all 36 settings in the
matched 12 model/effort hint comparison, covering the union of 48 unique settings.
The 12 reused baseline settings appearing in both plot views are not added again
to the study total of 2,068 unique attempts. The one-task, three-attempt DeepSeek
None pilot is preserved in the data but omitted from the plots, primary
denominator and frontier. Slide 8 remains its original single matrix of
39 attempts per cell. No experiments or regrading were performed.

Five no-hint and fourteen matched-hint settings have incomplete cost/usage.
They now appear at recorded minima, with open markers and rightward uncertainty
arrows. DeepSeek dollar bars terminate at conservative reserve ceilings, not
confirmed spending. Unknown costs were not filled by zero, averages of successful
attempts or a model-based estimate. Means divide by all attempts, including errors.
Dashed Pareto envelopes include all primary points at these displayed coordinates
and are provisional: unrecorded usage can move open points rightward.

Tokens are input plus output across all requests per attempt. Cached input is
already included in input, and reasoning is already included in output. Neither
is added twice. For DeepSeek, archived provider usage supersedes native client
usage, recovering retries and streams that finished after timeout. The sanitized
provider snapshot includes 157 batches and 3,341 unique request IDs (including the
pilot), with hashes and paths for 197 original files. No prompts or credentials
are included. Missing usage remains unknown. One Terra Medium/general source has
only aggregate total tokens; its input/output split remains unavailable. Raw token
count is not latency, FLOPs, uncached tokens or subscription usage.

The resource-summary.csv contains every coordinate, dollar lower/upper bounds,
token totals and completeness flags. resource-plot-audit.json lists the exact
settings drawn in each panel, and resource-summary.json preserves component
evidence. Frozen original prices and provider accounting are used without a new
price lookup. All original observations, figures and source files remain intact.

2026-10-16 display revision: the one-task DeepSeek None pilot marker and label are removed from both no-hint cost panels and standalone figures at the user's request. The underlying pilot records remain in the source data. All 24 primary no-hint settings, 36 matched-hint settings, coordinates, frontiers and denominators are unchanged.
