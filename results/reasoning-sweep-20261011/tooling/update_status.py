from datetime import datetime, timezone
from pathlib import Path
import json


def main():
    evidence = Path(__file__).resolve().parents[1]
    results = json.loads((evidence / "results.json").read_text())
    path = evidence / "run-status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    prime = [row for row in results["summary"] if row["model"].startswith("deepseek/")]
    recorded = sum(row["attempts"] for row in prime)
    complete = [row["reasoning_effort"] for row in prime if row["complete_panel"]]
    hold_path = evidence / "prime/provider-hold.json"
    hold = json.loads(hold_path.read_text()) if hold_path.exists() else {}
    state = "running"
    if hold.get("state") == "provider_rate_limit_hold":
        state = "waiting_for_provider"
    elif results["needs_review"]:
        state = "reviewing_execution_errors"
    elif recorded == 156 and all(row["complete_panel"] for row in results["summary"]):
        state = "trials_complete_pending_audit"
    note = ("Astra Extra High is complete: 37/39 completed passes, 39 passing checkpoints, two timeouts. "
            f"The fresh DeepSeek cohort has {recorded}/156 recorded attempts; completed effort panels: "
            + (", ".join(complete) if complete else "none yet") + ". ")
    if state == "waiting_for_provider":
        note += "New DeepSeek batches are on hold after provider errors; see the recorded hold reason and probe schedule. "
    elif recorded < 156:
        note += "The remaining finite schedule continues with a lid-open check before each new batch. "
    if (evidence / "timing-flag-reviews.json").exists():
        count = len(json.loads((evidence / "timing-flag-reviews.json").read_text()))
        note += f"{count} started DeepSeek High attempts affected by host sleep remain recorded with their original failures and timing outliers; no replacements. "
    note += "All 48 legacy DeepSeek observations and six unstarted setup failures remain separately preserved."
    status.update(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), note=note,
                  deepseek_recorded_attempts=recorded, deepseek_planned_attempts=156,
                  deepseek_complete_efforts=complete,
                  complete_comparison_points=sum(row["complete_panel"] for row in results["summary"]))
    path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"state": state, "deepseek_recorded_attempts": recorded, "complete_efforts": complete}))


if __name__ == "__main__":
    main()
