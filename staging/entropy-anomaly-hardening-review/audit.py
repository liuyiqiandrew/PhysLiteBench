"""Read-only entropy history audit; writes only this review's JSON inventory."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    path = ROOT / relative
    evidence[relative] = sha(path)
    return json.loads(path.read_text())


def functions(path):
    return {n.name: {"start": n.lineno, "end": n.end_lineno,
                     "ast": ast.dump(n, include_attributes=False)}
            for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


APPROVED_INPUTS = [
    "3c1d89a789ebf1e6bcad0e6665c74a735b360cbcc25dbf0efca2f9d71bb8cf56",
    "384aba72a1b1e5f7fb81f944fa54791fc3a15e013328522100a69dbc1d93e0eb",
    "6a5ef18a542d5fd42298ccab151355ecf224974f3b465d9a6299768988739b62",
]
JOBS = {
    3: "jobs/entropy-anomaly-retained-three-r1-plain-20261004-035631",
    4: "jobs/entropy-anomaly-zero-three-entropy-r4-plain-20261004-070639",
}
TASKS = {3: "tasks/entropy-anomaly", 4: "archives/entropy-anomaly-r4/tasks/entropy-anomaly"}
evidence = {}
ledgers = {
    3: read("results/retained-three-brownian-trial-reviews.json"),
    4: read("results/zero-three-entropy-r4-trial-reviews.json"),
}
report = {"scope": "Bounded history and hardening review; no model, Docker or scientific runs.",
          "existing_outcomes_unchanged": True, "trials": {}, "frozen_sources": {}}

for revision, job in JOBS.items():
    read(job + "/config.json")
    read(job + "/result.json")
    frozen = ROOT / job / "frozen-task"
    source_checks = {}
    for path in sorted(frozen.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(frozen))
        current = ROOT / TASKS[revision] / rel
        evidence[str(path.relative_to(ROOT))] = sha(path)
        evidence[str(current.relative_to(ROOT))] = sha(current)
        source_checks[rel] = {"frozen": sha(path), "current_or_archive": sha(current),
                              "equal": sha(path) == sha(current)}
    report["frozen_sources"][str(revision)] = source_checks
    original_functions = functions(frozen / "environment/model.py")
    for trial, old_review in ledgers[revision].items():
        if not trial.startswith("entropy-anomaly__"):
            continue
        base = job + "/" + trial
        result = read(base + "/result.json")
        metrics = read(base + "/verifier/metrics.json")
        trajectory = read(base + "/agent/trajectory.json")
        source = ROOT / base / "artifacts/app/model.py"
        evidence[str(source.relative_to(ROOT))] = sha(source)
        assert sha(source) == old_review["source_sha256"]
        for suffix in ["config.json", "agent/codex.txt", "verifier/test-stdout.txt", "verifier/reward.txt"]:
            path = ROOT / base / suffix
            if path.exists():
                evidence[str(path.relative_to(ROOT))] = sha(path)
        app_checks = {}
        for name in ["README.md", "test_public.py", "data/calibration.json"]:
            path = ROOT / base / "artifacts/app" / name
            if path.exists():
                evidence[str(path.relative_to(ROOT))] = sha(path)
                app_checks[name] = sha(path) == sha(frozen / "environment" / name)
        native_files = sorted((ROOT / base / "agent/sessions").rglob("*.jsonl"))
        assert len(native_files) == 1
        native = native_files[0]
        evidence[str(native.relative_to(ROOT))] = sha(native)
        users, contexts, cli, completions = [], [], set(), 0
        for line in native.read_text().splitlines():
            event = json.loads(line)
            payload = event.get("payload", {})
            if event["type"] == "session_meta":
                cli.add(payload.get("cli_version"))
            elif event["type"] == "turn_context":
                contexts.append((payload.get("model"), payload.get("effort")))
            elif event["type"] == "response_item" and payload.get("role") == "user":
                for item in payload.get("content", []):
                    if item.get("type") == "input_text":
                        users.append(hashlib.sha256(item["text"].encode()).hexdigest())
            elif event["type"] == "event_msg" and payload.get("type") == "task_complete":
                completions += 1
        assert users == APPROVED_INPUTS
        assert set(contexts) == {("gpt-5.6-luna", "high")}
        assert cli == {"0.154.0"}
        source_functions = functions(source)
        public_steps = []
        for step in trajectory["steps"]:
            if step["source"] != "agent":
                continue
            row = {"step": step["step_id"]}
            if step.get("message"):
                row["emitted_message"] = step["message"]
            if step.get("tool_calls"):
                row["emitted_calls"] = [
                    {"function": call["function_name"],
                     "arguments_sha256": hashlib.sha256(json.dumps(call["arguments"], sort_keys=True).encode()).hexdigest()}
                    for call in step["tool_calls"]]
            public_steps.append(row)
        report["trials"][trial] = {
            "revision": revision, "reward": result["verifier_result"]["rewards"]["reward"],
            "exception": result["exception_info"], "classification_preserved": old_review["classification"],
            "reason_preserved": old_review["reason"], "metrics": metrics,
            "source": str(source.relative_to(ROOT)), "source_sha256": sha(source),
            "native": str(native.relative_to(ROOT)), "native_user_block_sha256": users,
            "native_model_effort": sorted(set(contexts)), "cli": sorted(cli),
            "native_completion_events": completions, "unchanged_public_artifacts": app_checks,
            "source_function_lines": {k: {"start": v["start"], "end": v["end"]} for k, v in source_functions.items()},
            "unchanged_forward_function_ast": {k: k in source_functions and source_functions[k]["ast"] == v["ast"]
                                               for k, v in original_functions.items()},
            "public_step_inventory": public_steps,
        }
        assert result["exception_info"] is None
        assert result["verifier_result"]["rewards"]["reward"] == old_review["reward"]

extra = [
    "results/retained-three-r1-native-input-review.json",
    "results/zero-three-entropy-r4-results.json", "results/zero-three-entropy-r4-plan.json",
    "results/zero-three-entropy-r4-causal-diagnostic.json",
    "archives/entropy-anomaly-r1/tasks/entropy-anomaly/AUTHOR.md",
    "archives/entropy-anomaly-r1/results/trial-reviews.json",
    "archives/entropy-anomaly-r2/tasks/entropy-anomaly/AUTHOR.md",
    "archives/entropy-anomaly-r2/results/trial-reviews.json",
    "archives/chemical-route-power-r1/tasks/chemical-route-power/AUTHOR.md",
    "archives/chemical-route-power-r1/tasks/chemical-route-power/environment/model.py",
    "archives/chemical-route-power-r1/tasks/chemical-route-power/solution/model.py",
    "archives/chemical-route-power-r1/results/zero-three-chemical-route-r1-results.json",
    "archives/screened/thermal-bodies-r12/tasks/thermal-bodies/AUTHOR.md",
    "archives/screened/thermal-bodies-r12/trial-reviews.json",
    "archives/thermal-bodies-r11/tasks/thermal-bodies/AUTHOR.md",
    "archives/thermal-bodies-r11/trial-reviews.json",
    "archives/magnetic-bath-transfer-r1/tasks/magnetic-bath-transfer/AUTHOR.md",
    "archives/magnetic-bath-transfer-r1/results/trial-reviews.json",
    "archives/active-bath-work-r1/tasks/active-bath-work/AUTHOR.md",
    "staging/rotating-reservoir-r2/tasks/rotating-reservoir/AUTHOR.md",
]
for name in extra:
    evidence[name] = sha(ROOT / name)
report["evidence_sha256"] = evidence
report["summary"] = {
    "r3": {"completed": 3, "passes": 2, "mixed_failures": 1, "clean_physical_failures": 0, "exceptions": 0},
    "r4": {"completed": 3, "passes": 2, "mixed_failures": 0, "clean_physical_failures": 1, "exceptions": 0},
    "all_native_inputs_match": True, "new_model_or_science_runs": 0,
}
(HERE / "trial-review.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"trials": len(report["trials"]), "evidence_files": len(evidence),
                  "frozen_differences": {r: [p for p, v in values.items() if not v["equal"]]
                                         for r, values in report["frozen_sources"].items()},
                  "review_sha256": sha(HERE / "trial-review.json")}, indent=2))
