"""Copy whitelisted per-trial evidence for qualified tasks out of local Harbor jobs.

Run from the repository root on the machine that holds the raw jobs:

    python3 scripts/collect_trial_evidence.py
    python3 scripts/collect_trial_evidence.py --task entropy-anomaly --force

Trial IDs and revisions come from results/qualified-tasks.json. Each trial must be the only
match for <jobs-root>/*/<trial_id>/. Only the final model.py, the verifier's metrics.json,
reward.txt and test-stdout.txt are copied to <output-root>/<task>-r<revision>-evidence/<trial_id>/,
together with model.diff, the submitted model.py diffed against the job's frozen starter.
Agent sessions, authentication, config.json, result.json, logs and the raw final.diff are never
copied. manifest.json records the raw job as jobs/<job-name>, each trial's reward, the sha256 of
every copied file, the job's frozen-task compared with the qualified source hashes, and the
run-timing model, effort and hint settings.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ("entropy-anomaly", "rotating-reservoir", "superconducting-heat", "hierarchical-glass-overlaps")
WHITELIST = {"model.py": "artifacts/app/model.py", "metrics.json": "verifier/metrics.json",
             "reward.txt": "verifier/reward.txt", "test-stdout.txt": "verifier/test-stdout.txt"}
UNPUBLISHED = ("archives/entropy-anomaly-r6", "archives/rotating-reservoir-r7",
               "results/prototypes/rotating-reservoir-r8-root", "archives/superconducting-heat-r4",
               "results/entropy-anomaly-r7-validation.json", "results/rotating-reservoir-r8-validation.json",
               "results/superconducting-heat-r6-validation.json",
               "results/superconducting-heat-r6-validation-inheritance.json")
TIMING_KEYS = ("agent", "model", "reasoning_effort", "physics_hint")
TRIAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_trial(jobs_root, trial_id):
    matches = sorted(p for p in jobs_root.glob(f"*/{trial_id}") if p.is_dir())
    if not matches:
        raise SystemExit(f"error: trial {trial_id} not found under {jobs_root}/*/")
    if len(matches) > 1:
        raise SystemExit(f"error: trial {trial_id} is ambiguous: " + ", ".join(map(str, matches)))
    return matches[0]


def check_frozen(job, expected):
    frozen = job / "frozen-task"
    if not frozen.is_dir():
        return {"status": "absent"}
    have = {}
    for path in frozen.rglob("*"):
        name = path.relative_to(frozen)
        if path.is_file() and not any(part in ("__pycache__", ".pytest_cache", ".git") for part in name.parts):
            have[str(name)] = sha256(path)
    diff = {"changed": sorted(k for k in expected if k in have and have[k] != expected[k]),
            "missing": sorted(k for k in expected if k not in have),
            "extra": sorted(k for k in have if k not in expected)}
    return {"status": "mismatch" if any(diff.values()) else "match", **diff}


def model_diff(starter, final):
    """Diff of the submitted model.py against the frozen starter.

    Rebuilt here rather than taken from final.diff, which covers every text file the agent
    left in /app and so may hold unrelated or private content.
    """
    before = starter.read_text(errors="replace").splitlines(keepends=True)
    after = final.read_text(errors="replace").splitlines(keepends=True)
    return "".join(difflib.unified_diff(before, after, fromfile="before/model.py", tofile="after/model.py"))


def read_reward(path):
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def main():
    qualified = json.loads((ROOT / "results/qualified-tasks.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", action="append", choices=sorted(qualified["tasks"]), metavar="NAME",
                        help="Qualified task to collect (repeatable); default: " + ", ".join(DEFAULT_TASKS))
    parser.add_argument("--jobs-root", type=Path, default=ROOT / "jobs", help="Harbor jobs directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results", help="Parent of the evidence directories")
    parser.add_argument("--force", action="store_true", help="Delete and rewrite existing evidence directories")
    args = parser.parse_args()
    if not args.jobs_root.is_dir():
        parser.error(f"jobs root does not exist: {args.jobs_root}")
    want = {"agent": "codex", **{k: qualified[k] for k in TIMING_KEYS[1:]}}
    plans = []
    for name in dict.fromkeys(args.task or DEFAULT_TASKS):
        entry = qualified["tasks"][name]
        if not isinstance(entry.get("revision"), int):
            parser.error(f"{name} has no integer revision in results/qualified-tasks.json")
        bad = [t for t in entry["trial_ids"] if not TRIAL_ID.match(t)]
        if bad:
            parser.error(f"{name} has invalid trial ids: {', '.join(bad)}")
        out = args.output_root / f"{name}-r{entry['revision']}-evidence"
        if out.exists() and not args.force:
            parser.error(f"{out} already exists; pass --force to replace it")
        plans.append((name, entry, out, [find_trial(args.jobs_root, t) for t in entry["trial_ids"]]))

    warnings = []
    for name, entry, final, trials in plans:
        out = final.with_name(final.name + ".partial")
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        jobs = sorted({trial.parent for trial in trials})
        raw = [f"jobs/{job.name}" for job in jobs]
        manifest = {"task": name, "revision": entry["revision"], "raw_job": raw[0] if len(raw) == 1 else raw,
                    "frozen_task": {}, "run_timing": {}, "trials": [], "warnings": [], "source_sha256": {}}
        notes = manifest["warnings"]
        print(f"{name} r{entry['revision']} -> {final}")
        for job, label in zip(jobs, raw):
            check = manifest["frozen_task"][label] = check_frozen(job, entry["source_sha256"])
            print(f"  {label}/frozen-task: {check['status']}")
            if check["status"] != "match":
                details = "; ".join(f"{k} {', '.join(v)}" for k, v in check.items() if k != "status" and v)
                notes.append(f"{label}/frozen-task {check['status']}" + (f": {details}" if details else ""))
            timing_file = job / "run-timing.json"
            try:
                timing = json.loads(timing_file.read_text())
                problem = None if isinstance(timing, dict) else "is not a JSON object"
            except FileNotFoundError:
                timing, problem = None, "absent"
            except (OSError, ValueError) as error:
                timing, problem = None, f"unreadable ({type(error).__name__})"
            if problem:
                manifest["run_timing"][label] = None
                notes.append(f"{label}/run-timing.json {problem}")
            else:
                manifest["run_timing"][label] = {k: timing.get(k) for k in TIMING_KEYS}
                notes.extend(f"{label}/run-timing.json {k}={timing.get(k)!r}, expected {v!r}"
                             for k, v in want.items() if timing.get(k) != v)
        for trial in trials:
            copied, absent = [], []
            for dest_name, rel in WHITELIST.items():
                src = trial / rel
                inside = src.resolve().is_relative_to(trial.resolve())
                if src.is_symlink() or (src.exists() and not inside):
                    absent.append(f"{rel} (symlink, not copied)")
                    notes.append(f"{trial.name}/{rel} is or passes through a symlink; not copied")
                elif src.is_file():
                    dest = out / trial.name / dest_name
                    dest.parent.mkdir(exist_ok=True)
                    shutil.copyfile(src, dest)
                    manifest["source_sha256"][f"{trial.name}/{dest_name}"] = sha256(dest)
                    copied.append(dest_name)
                else:
                    absent.append(rel)
            starter = trial.parent / "frozen-task/environment/model.py"
            if "model.py" in copied and starter.is_file() and not starter.is_symlink() \
                    and starter.resolve().is_relative_to(trial.parent.resolve()):
                dest = out / trial.name / "model.diff"
                dest.write_text(model_diff(starter, out / trial.name / "model.py"))
                manifest["source_sha256"][f"{trial.name}/model.diff"] = sha256(dest)
                copied.append("model.diff")
            else:
                absent.append("model.diff (needs model.py and frozen-task/environment/model.py)")
            reward = read_reward(trial / "verifier/reward.txt")
            manifest["trials"].append({"trial": trial.name, "raw_job": f"jobs/{trial.parent.name}",
                                       "reward": reward, "copied": copied, "absent": absent})
            print(f"  {trial.name}: reward {reward}; copied {len(copied)} files"
                  + (f"; absent {', '.join(absent)}" if absent else ""))
            if reward != 0:
                notes.append(f"{trial.name} reward {reward}, expected 0")
        manifest["source_sha256"] = dict(sorted(manifest["source_sha256"].items()))
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if final.exists():
            shutil.rmtree(final)
        out.rename(final)
        warnings.extend(f"{name}: {note}" for note in notes)

    for note in warnings:
        print(f"WARNING: {note}")
    print("Unpublished inputs cited by validators and AUTHOR notes:")
    for rel in UNPUBLISHED:
        print(f"  {'present' if (ROOT / rel).exists() else 'missing'}  {rel}")


if __name__ == "__main__":
    main()
