from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('run_slug')
parser.add_argument('--accept', action='store_true')
args = parser.parse_args()
repo = Path('/Users/yp9846/Desktop/PhysLiteBench')
evidence = repo / 'results/reasoning-sweep-20261011'
run_path = evidence / 'runs' / args.run_slug / 'run.json'
run = json.loads(run_path.read_text())
assert run['status'] == 'needs_review'
audit = json.loads((evidence / 'evidence-audit.json').read_text())
trials = []
flags = []
for trial in run['trials']:
    result = json.loads((repo / trial['result_path']).read_text())
    exception = result.get('exception_info') or {}
    interval = result.get('agent_execution') or {}
    seconds = None
    if interval.get('started_at') and interval.get('finished_at'):
        seconds = (datetime.fromisoformat(interval['finished_at'].replace('Z', '+00:00')) - datetime.fromisoformat(interval['started_at'].replace('Z', '+00:00'))).total_seconds()
    check = next(item for item in audit if item['trial'] == trial['trial'])
    flags.extend(check['source_flags'] + check['network_flags'])
    assert check['model_effort_verified']
    diagnostic = []
    transcript = repo / trial['result_path']
    transcript = transcript.parent / 'agent/codex.txt'
    if transcript.exists():
        for line in transcript.read_text(errors='replace').splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') == 'error':
                diagnostic.append(event.get('message', ''))
    trials.append(dict(trial=trial['trial'], reward=trial['reward'], exception_type=exception.get('exception_type'), terminal_diagnostics=diagnostic[-3:], agent_seconds=seconds))
record = dict(reviewed_utc=datetime.now(timezone.utc).isoformat(), job=run['job'], task=run['task'], model=run['model'], effort=run['reasoning_effort'], trials=trials, completed_passes=run['passes'], errors=run['errors'], passing_checkpoints=sum(trial['reward'] == 1 for trial in trials), source_or_network_flags=flags, outcome='Preserve every original trial without retry or replacement.', cost_caveat='Interrupted response usage may be incomplete; Prime reserves remain in the ledger.')
if run.get('prime_batch_ledger'):
    ledger = json.loads((repo / run['prime_batch_ledger']).read_text())
    provider_errors = [item for item in ledger['requests'] if item.get('http_status', 0) >= 400]
    record['provider_http_errors'] = [
        {key: item.get(key) for key in ['id', 'http_status', 'error', 'upstream_headers_received_at']}
        for item in provider_errors
    ]
    for trial in record['trials']:
        if trial['exception_type'] != 'NonZeroAgentExitCodeError':
            continue
        source = next(item for item in run['trials'] if item['trial'] == trial['trial'])
        result = json.loads((repo / source['result_path']).read_text())
        ended = datetime.fromisoformat(result['agent_execution']['finished_at'].replace('Z', '+00:00'))
        matches = [item for item in provider_errors if item.get('upstream_headers_received_at')
                   and 0 <= (ended - datetime.fromisoformat(item['upstream_headers_received_at'])).total_seconds() <= 2]
        if len(matches) == 1:
            trial['provider_error_correlation'] = dict(
                http_status=matches[0]['http_status'], request_id=matches[0]['id'],
                basis='Unique provider HTTP error within two seconds before the agent exit; matched to the terminal stream-closure diagnostic.')
print(json.dumps(record, indent=2))
if args.accept:
    assert not flags
    assert len(trials) == 3 and run['returncode'] == 0
    assert all(trial['exception_type'] in (None, 'AgentTimeoutError', 'NonZeroAgentExitCodeError') for trial in trials)
    assert all(any('stream disconnected before completion' in message for message in trial['terminal_diagnostics']) for trial in trials if trial['exception_type'] == 'NonZeroAgentExitCodeError')
    assert all(595 <= trial['agent_seconds'] <= 660 for trial in trials if trial['exception_type'] == 'AgentTimeoutError')
    if run['model'].startswith('deepseek/'):
        assert run['prime_requests_settled'] and not run['prime_budget_blocked']
    record['classification'] = 'Recorded grading outcomes plus reviewed ten-minute agent timeouts or explicitly logged stream disconnects. No physical diagnosis inferred from an execution error.'
    destination = evidence / 'reviews' / (args.run_slug + '-continuation-review.json')
    original = evidence / 'reviews' / (args.run_slug + '-initial-run.json')
    assert not original.exists() and not destination.exists()
    shutil.copy2(run_path, original)
    destination.write_text(json.dumps(record, indent=2) + '\n')
    run['status'] = 'completed_with_reviewed_errors'
    run['review'] = str(destination.relative_to(repo))
    run_path.write_text(json.dumps(run, indent=2) + '\n')
