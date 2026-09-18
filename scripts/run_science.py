"""Run a local science task in Harbor and retain the exact tested revision."""
import argparse
from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]


def save_diffs(job, original):
    for snapshot in job.glob("*/artifacts/app"):
        names = {p.relative_to(snapshot) for p in snapshot.rglob("*") if p.is_file()}
        names.update(p.relative_to(original) for p in original.rglob("*")
                     if p.is_file() and p.name != "Dockerfile")
        patch = []
        for name in sorted(names):
            if any(part in ("__pycache__", ".pytest_cache", ".git") for part in name.parts):
                continue
            before, after = original / name, snapshot / name
            try:
                a = before.read_text().splitlines(keepends=True) if before.exists() else []
                b = after.read_text().splitlines(keepends=True) if after.exists() else []
            except UnicodeDecodeError:
                continue
            patch.extend(difflib.unified_diff(a, b, fromfile=f"before/{name}", tofile=f"after/{name}"))
        (snapshot.parent.parent / "final.diff").write_text("".join(patch))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path, help="Task directory, e.g. tasks/qubit-control")
    parser.add_argument("--agent", choices=("codex", "oracle", "nop"), default="codex")
    parser.add_argument("--solution-model", type=Path, help="Replace the private oracle model for a control run")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.6-luna", help="Codex model ID")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--job-name")
    args = parser.parse_args()
    source = args.task.resolve()
    if args.trials < 1 or args.concurrency < 1:
        parser.error("trials and concurrency must be positive")
    if args.solution_model and args.agent != "oracle":
        parser.error("--solution-model requires --agent oracle")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = args.job_name or f"{source.name}-{args.agent}-{stamp}"
    if Path(name).name != name:
        parser.error("job name must be a directory name")
    job = ROOT / "jobs" / name
    if job.exists():
        parser.error(f"job already exists: {name}; choose a fresh job name")
    with tempfile.TemporaryDirectory(prefix="science-task-") as tmp:
        task = Path(tmp) / source.name
        shutil.copytree(source, task, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".git"))
        if args.solution_model:
            shutil.copyfile(args.solution_model, task / "solution/model.py")
        command = ["uvx", "--python", "3.13", "harbor@0.21.0", "run", "-p", str(task),
                   "-a", args.agent, "--env", "docker", "-n", str(min(args.concurrency, args.trials)),
                   "-k", str(args.trials), "--job-name", name]
        if args.agent == "codex":
            auth = Path(os.environ.get("CODEX_AUTH_JSON_PATH", str(Path.home() / ".codex/auth.json")))
            command += ["-m", args.model, "--ak", "version=0.154.0",
                        "--ak", f"reasoning_effort={args.reasoning_effort}",
                        "--ae", f"CODEX_AUTH_JSON_PATH={auth}"]
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            command = ["caffeinate", "-i", *command]
        started = datetime.now(timezone.utc)
        monotonic_start = time.monotonic()
        try:
            result = subprocess.run(command, cwd=ROOT)
        finally:
            elapsed = time.monotonic()-monotonic_start
            finished = datetime.now(timezone.utc)
            if job.exists():
                shutil.copytree(task, job / "frozen-task")
                shutil.copyfile(task / "instruction.md", job / "instruction.md")
                wall = (finished-started).total_seconds()
                (job / "run-timing.json").write_text(json.dumps({
                    "agent": args.agent, "task": source.name,
                    "model": args.model if args.agent == "codex" else None,
                    "reasoning_effort": args.reasoning_effort if args.agent == "codex" else None,
                    "started_at": started.isoformat(), "finished_at": finished.isoformat(),
                    "wall_seconds": wall, "monotonic_seconds": elapsed,
                    "wall_minus_monotonic_seconds": wall-elapsed,
                }, indent=2)+"\n")
                save_diffs(job, task / "environment")
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
