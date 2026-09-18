# Adding a physics task

A task should expose a specific physical assumption that survives calibration and fails on another allowed preparation. Keep the implementation small enough that a reviewer can understand the apparatus, derive the intended model, and identify why the shortcut fails.

## 1. Agree on the physics before writing the harness

Write a short proposal in the PR or issue:

- **Apparatus:** what is prepared, measured, and held fixed; relevant units and approximations.
- **Unknowns:** what the agent must infer and whether calibration identifies it.
- **Shortcut:** the physically inappropriate assumption and why it fits calibration.
- **Discriminating experiment:** a preparation where the correct model and shortcut differ, within the public apparatus specification.
- **Target evaluation:** model, reasoning effort, number of trials, and the outcome being investigated.

The apparatus description must give a field expert enough information to derive the correct model. Keep the intended diagnosis and solution in author documentation. Hidden tests should examine consequences of the stated apparatus; they should not introduce an undisclosed boundary condition, interaction, or measurement rule.

The completed shortcut must be a valid implementation of its assumed equations. A syntax error, incorrect formula, unstable solver, or unfinished fit does not establish the intended physical failure. See [TASKS.md](../TASKS.md) for worked examples of this distinction.

## 2. Use one self-contained task directory

Choose a descriptive lowercase, hyphenated name, such as `new-system`. Use underscores for Python helper filenames.

```text
tasks/new-system/
  instruction.md             Short request to the solving agent
  task.toml                  Harbor limits and artifact collection
  environment/
    Dockerfile               Agent image, with pinned dependencies
    README.md                Apparatus, units, data schema, and model API
    model.py                 Starter implementation
    test_public.py           Visible interface and calibration checks
    data/                    Public calibration measurements
  tests/
    test.sh                  Runs verification and writes the reward
    test_hidden.py           Private calibration and prediction checks
    reference.py             Independent physical reference
    data/                    Verifier-owned calibration measurements
  solution/
    solve.sh                 Installs the reference solution into /app
    model.py                 Complete correct model
scripts/
  new_system_baseline.py     Completed physically incorrect control
  validate_new_system.py     Data generation and scientific validation
```

Include metadata files where the data format requires them. The three existing tasks are structural examples; the tree above is a convention, not a demand to reuse their physical models or class names.

The [runner](../scripts/run_science.py) accepts any task directory by path. There is no registry or shared base class to update. A task's environment, verifier, and solution must work without importing another task, repository-level scripts, or anything in `scratch/`.

## 3. Keep the agent and author files separate

Use the existing [Dockerfile](../tasks/qubit-control/environment/Dockerfile), [task configuration](../tasks/qubit-control/task.toml), [verification script](../tasks/qubit-control/tests/test.sh), and [solution installer](../tasks/qubit-control/solution/solve.sh) as small examples.

The Docker build context is `environment/`. Copy its public files explicitly into `/app`; keep solutions, hidden cases, true values of fitted parameters, and author explanations outside that image. The solving agent receives the task instruction and public environment. Harbor supplies `/tests` for verification and `/solution` for oracle runs. Preserve the `/app` artifact declaration in `task.toml` so final submissions and source diffs are retained.

Document every public input and output: function signatures, array shapes, units, initial conditions, time conventions, parameter fields, and allowed parameter ranges. For a fitting task, follow the existing `fit(...)` / `predict(...)` pattern, with `fit` returning `self`. Prediction signatures may differ by system.

`tests/test.sh` must write numeric `1` or `0` to `/logs/verifier/reward.txt`, including when pytest fails. Save diagnostic metrics separately under `/logs/verifier/`, so a failed reward can be explained. Private tests must evaluate the submitted `/app/model.py` against trusted references and private calibration copies. Public tests should check the advertised interface and calibration without revealing the hidden diagnostic preparations.

`solution/solve.sh` should install the correct implementation without editing tests. Keeping the implementation in `solution/model.py` also enables the runner's `--solution-model` control: that option replaces **only this file**, then runs the existing solution installer. Adapt this arrangement explicitly if a new task needs multiple solution files.

Keep the current 600-second agent and 60-second verifier limits for comparable evaluations. Explain any required limit or dependency changes in the PR.

## 4. Validate the task before measuring agent difficulty

Build both completed controls and check their expected behavior:

| Control | Interface and calibration | Parameter recovery | Hidden predictions |
|---|---|---|---|
| Physical oracle | Pass | Pass | Pass |
| Completed shortcut | Pass | Pass | Fail for the intended physical reason |
| Unfinished starter | May fail | Not established | Not a physical-failure control |

Use an independent derivation, analytical limit, or numerical reference to check the oracle. Copying the oracle into the verifier can reproduce the same mistake in both. Check relevant conservation laws, limiting cases, and numerical convergence. Choose prediction tolerances from measurement uncertainty and numerical error; document the normalization. The current tasks' thresholds are examples, not universal values for new systems.

For noisy calibration, check that the separation survives repeated noise realizations; the existing validators use 256. Record random seeds. Ordinary validation should read checked-in data; make regeneration an explicit option that updates public and private copies together.

Then check both controls in fresh Docker environments. From the repository root, after creating the example files above:

```bash
python3 scripts/run_science.py tasks/new-system --agent oracle --trials 1
python3 scripts/run_science.py tasks/new-system --agent oracle --trials 1 \
  --solution-model scripts/new_system_baseline.py
```

Inspect verifier metrics and failed test names as well as the reward. A zero from an unfinished starter or broken control is not evidence of a physical modeling error. Run pytest per task; different tasks intentionally reuse module names such as `model` and `reference`.

## 5. Evaluate a frozen revision and hand it over

After the controls behave correctly, run the agreed agent batch:

```bash
python3 scripts/run_science.py tasks/new-system \
  --model gpt-5.6-luna --reasoning-effort high --trials 3 --concurrency 3
```

Independent batches may run concurrently. Use distinct `--job-name` values when launching the same task concurrently or collecting runs from several teammates. Keep prompts, data, starting code, and grading fixed throughout each batch. Review final submissions and trajectories to distinguish physical failures from mathematical, numerical, coding, and infrastructure failures. Report every trial; keep exploratory batches separate from confirmation batches.

Use a branch and PR for each new task. Ask a teammate to review the physical derivation and calibration loophole. Include:

- The task files, completed shortcut, validator, and exact reproduction commands.
- Oracle/shortcut outcomes, threshold justification, seeds, and relevant convergence or noise checks.
- The Git commit and tested task checksum, model, effort, trial count, and a team-accessible location for the full artifacts.
- A short physics explanation in [TASKS.md](../TASKS.md), an entry in the [README](../README.md), and interface/grading details in [BENCHMARK.md](BENCHMARK.md).

Add reviewed evaluation results to [DASHBOARD.md](../DASHBOARD.md) and `results/` together. Keep raw jobs in the shared archive rather than committing them. Results can be pending when the task is proposed; passing scientific controls and demonstrated agent difficulty are separate claims.

Start a new task at revision 1. Increment its documented revision when changing the scientific setup, calibration, starter, prompt, or grading, and evaluate that revision separately. Preserve earlier evidence; the runner's frozen task and checksum identify what was actually tested. Use the Git commit to identify documentation-only or folder-layout changes without renumbering the physics revision.
