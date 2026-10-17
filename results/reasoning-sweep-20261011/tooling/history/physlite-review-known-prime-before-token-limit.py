from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import subprocess
import sys

repo = Path('/Users/yp9846/Desktop/PhysLiteBench')
evidence = repo / 'results/reasoning-sweep-20261011'
slug = sys.argv[1]
path = evidence / 'runs' / slug / 'run.json'
run = json.loads(path.read_text())
assert run['status'] == 'needs_review'
assert run.get('comparison_cohort') == 'prime-transport-v2'
assert run['model'] == 'deepseek/deepseek-v4.1-flash'
assert len(run['trials']) == 3 and run['returncode'] == 0
assert run['prime_requests_settled'] and not run['prime_budget_blocked']
command = [sys.executable, '/private/tmp/physlite-review-continuation.py', slug]
review = json.loads(subprocess.check_output(command, cwd=repo, text=True))
assert not review['source_or_network_flags']
for trial in review['trials']:
    exception = trial['exception_type']
    if exception is None:
        continue
    if exception == 'AgentTimeoutError':
        assert 595 <= trial['agent_seconds'] <= 660
    elif exception == 'NonZeroAgentExitCodeError':
        assert trial.get('provider_error_correlation', {}).get('http_status') == 429
        assert any('stream disconnected before completion' in text for text in trial['terminal_diagnostics'])
    else:
        raise AssertionError('Unfamiliar exception requires inspection')

subprocess.check_output(command + ['--accept'], cwd=repo, text=True)
run = json.loads(path.read_text())
review_path = repo / run['review']
review = json.loads(review_path.read_text())
review['review_method'] = 'Automated strict checks of previously inspected failure types: exact native model/effort, zero source/network keyword flags, three terminal results, settled costs, 595–660 second timeouts, or uniquely time-correlated provider HTTP 429 with a terminal stream-closure diagnostic. No physics-cause classification and no retry.'
review_path.write_text(json.dumps(review, indent=2) + '\n')
rate_failures = sum(trial.get('provider_error_correlation', {}).get('http_status') == 429 for trial in review['trials'])
if rate_failures == 3:
    hold_path = evidence / 'prime/provider-hold.json'
    previous = json.loads(hold_path.read_text()) if hold_path.exists() else None
    now = datetime.now(timezone.utc)
    if previous:
        history = evidence / 'prime' / ('provider-hold-history-' + now.strftime('%Y%m%d-%H%M%S') + '.json')
        history.write_text(json.dumps(previous, indent=2) + '\n')
    hold = dict(state='provider_rate_limit_hold', created_utc=now.isoformat(),
                probe_not_before_utc=(now + timedelta(minutes=10)).isoformat(),
                reason='Three of three attempts received provider HTTP 429; hold new DeepSeek batches for cooldown and a bounded availability probe.',
                resume_requires_successful_probe=True, triggering_job=run['job'])
    hold_path.write_text(json.dumps(hold, indent=2) + '\n')
print(json.dumps(dict(task=run['task'], effort=run['reasoning_effort'],
                      completed_passes=run['passes'], errors=run['errors'],
                      rate_limit_failures=rate_failures, hold=rate_failures == 3,
                      review=run['review'])))
