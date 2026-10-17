from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import subprocess
import sys

checkpoint = sys.argv[1:] == ['--checkpoint']
assert checkpoint or not sys.argv[1:], 'Only --checkpoint is supported'

repo = Path('/Users/yp9846/Desktop/PhysLiteBench')
evidence = repo / 'results/reasoning-sweep-20261011'
plan_path = evidence / 'plan.json'
plan = json.loads(plan_path.read_text())
results = json.loads((evidence / 'results.json').read_text())
audit = json.loads((evidence / 'evidence-audit.json').read_text())
trials = results['trials']
new_trials = [trial for trial in trials if not trial['reused']]
assert not results['running'] and not results['needs_review']
assert len({trial['trial'] for trial in trials}) == len(trials)
assert len(audit) == len(new_trials)
assert all(row['model_effort_verified'] for row in audit)
source_reviews = {row['trial']: row for row in json.loads((evidence / 'source-flag-reviews.json').read_text())}
network_reviews = {row['trial']: row for row in json.loads((evidence / 'network-flag-reviews.json').read_text())}
trial_records = {row['trial']: row for row in trials}
reviewed_source_flags = 0
reviewed_network_flags = 0
for row in audit:
    if row['source_flags']:
        review = source_reviews[row['trial']]
        assert review['flags'] == row['source_flags']
        assert review['model_sha256'] == row['file_hashes']['model.py']
        assert review['resolution'] == 'benign_docstring'
        assert review['model'] == row['model'] and review['effort'] == row['effort']
        reviewed_source_flags += len(row['source_flags'])
    if row['network_flags']:
        review = network_reviews[row['trial']]
        assert review['flags'] == row['network_flags']
        assert review['trajectory_sha256'] == trial_records[row['trial']]['trajectory_sha256']
        assert review['resolution'] == 'benign_physics_docstring'
        assert review['model'] == row['model'] and review['effort'] == row['effort']
        reviewed_network_flags += len(row['network_flags'])
timing_reviews_path = evidence / 'timing-flag-reviews.json'
timing_reviews = {row['trial']: row for row in json.loads(timing_reviews_path.read_text())} if timing_reviews_path.exists() else {}
reviewed_timing_outliers = 0
for trial in new_trials:
    if trial.get('timing_outlier'):
        review = timing_reviews[trial['trial']]
        assert review['resolution'] == 'host_sleep_affected'
        assert review['model'] == trial['model'] and review['effort'] == trial['reasoning_effort']
        assert review['agent_execution_seconds'] == trial['agent_execution_seconds']
        assert review['trajectory_sha256'] == trial['trajectory_sha256']
        assert review['source_sha256'] == trial['source_sha256']
        assert hashlib.sha256((repo / review['raw_result_path']).read_bytes()).hexdigest() == review['raw_result_sha256']
        assert hashlib.sha256((repo / review['power_events']).read_bytes()).hexdigest() == review['power_events_sha256']
        assert trial['exception']['exception_type'] == 'NonZeroAgentExitCodeError'
        reviewed_timing_outliers += 1
assert results['plan_sha256'] == hashlib.sha256(plan_path.read_bytes()).hexdigest()

files_checked = 0
for task, specification in plan['tasks'].items():
    for name, expected in specification['qualified_source_sha256'].items():
        assert hashlib.sha256((repo / 'tasks' / task / name).read_bytes()).hexdigest() == expected, (task, name)
        files_checked += 1
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert head == plan['source_commit']

by_trial = {trial['trial']: trial for trial in new_trials}
exported_files_checked = 0
for row in audit:
    destination = repo / by_trial[row['trial']]['evidence']
    for name, expected in row['file_hashes'].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == expected
        exported_files_checked += 1

complete = []
for setting in results['summary']:
    selected = [trial for trial in trials if (trial['model'], trial['reasoning_effort']) == (setting['model'], setting['reasoning_effort'])]
    assert len(selected) == setting['attempts']
    assert sum(trial['reward'] == 1 and trial['exception'] is None for trial in selected) == setting['passes']
    assert sum(trial['reward'] == 1 for trial in selected) == setting['scored_checkpoint_passes']
    assert sum(trial['exception'] is not None for trial in selected) == setting['errors']
    if setting['complete_panel']:
        assert len(selected) == 39
        assert Counter(trial['task'] for trial in selected) == Counter({task: 3 for task in plan['tasks']})
        complete.append((setting['model'], setting['reasoning_effort']))

recovery = json.loads((evidence / 'unstarted-setup-recovery-20261012.json').read_text())
assert recovery['preserved_unstarted_attempts'] == 6
for batch in recovery['batches']:
    old_run = repo / batch['archived_run']
    assert hashlib.sha256(old_run.read_bytes()).hexdigest() == batch['original_run_sha256']
    replacement = json.loads((evidence / 'runs' / batch['slug'] / 'run.json').read_text())
    assert replacement['replaces_unstarted_job'] == batch['original_job']
    assert replacement['job'] != batch['original_job']
    for item in batch['trials']:
        old_result = repo / item['result_path']
        assert hashlib.sha256(old_result.read_bytes()).hexdigest() == item['result_sha256']
        result = json.loads(old_result.read_text())
        assert result.get('agent_execution') is None and result.get('verifier_result') is None
        assert not list((old_result.parent / 'agent').rglob('*.jsonl'))
        assert item['trial'] not in trial_records

if not checkpoint:
    assert len(complete) == 16
    assert len(trials) == 624 and len(new_trials) == 504
else:
    assert 12 <= len(complete) <= 16
    assert len(trials) <= 624 and len(new_trials) <= 504
assert len(trials) - len(new_trials) == 120
prime_plan_path = evidence / 'prime/transport-v2-plan.json'
prime_plan = json.loads(prime_plan_path.read_text())
prime_plan_sha256 = hashlib.sha256(prime_plan_path.read_bytes()).hexdigest()
legacy = json.loads((evidence / 'prime/legacy-transport-results.json').read_text())
assert len(legacy['trials']) == 48
assert not {trial['trial'] for trial in legacy['trials']} & {trial['trial'] for trial in trials}
prime_requests_checked = 0
prime_batches_checked = 0
for path in (evidence / 'runs').glob('prime-v2--*/run.json'):
    run = json.loads(path.read_text())
    assert run['comparison_cohort'] == 'prime-transport-v2'
    assert run['plan_sha256'] == prime_plan_sha256
    assert run['prime_requests_settled'] and not run['prime_budget_blocked']
    assert run['proxy_sha256'] == prime_plan['proxy_sha256']
    assert run['config_sha256'] == prime_plan['config_sha256']
    batch = json.loads((repo / run['prime_batch_ledger']).read_text())
    assert batch['requests']
    for request in batch['requests']:
        assert request['transport_protocol'] == 'prime-transport-v2'
        assert request['provider_reasoning_effort'] == run['reasoning_effort']
        assert request['client_reasoning_effort'] == ('xhigh' if run['reasoning_effort'] == 'max' else run['reasoning_effort'])
        assert request['status'] != 'pending'
        prime_requests_checked += 1
    prime_batches_checked += 1
assert prime_batches_checked * 3 == sum(trial['model'].startswith('deepseek/') for trial in trials)
assert prime_batches_checked <= 52
if not checkpoint:
    assert prime_batches_checked == 52

with (evidence / 'trial-outcomes.csv').open() as stream:
    table = list(csv.DictReader(stream))
assert len(table) == len(trials)
assert {row['trial'] for row in table} == {trial['trial'] for trial in trials}
run_states = Counter(json.loads(path.read_text())['status'] for path in (evidence / 'runs').glob('*/run.json'))
assert set(run_states) <= {'completed', 'completed_with_reviewed_errors'}
checked_utc = datetime.now(timezone.utc).isoformat()
source_record = dict(checked_utc=checked_utc, task_count=len(plan['tasks']), files_checked=files_checked, mismatches=[], source_commit=head, note='All installed task files still match the frozen pre-sweep source hashes.')
(evidence / 'task-source-audit.json').write_text(json.dumps(source_record, indent=2) + '\n')
record = dict(checked_utc=checked_utc, source_commit=head, complete_points=len(complete), total_recorded_attempts=len(trials), new_attempts=len(new_trials), reused_high_attempts=len(trials)-len(new_trials), task_files_checked=files_checked, exported_evidence_files_checked=exported_files_checked, all_model_efforts_verified=True, unreviewed_source_or_network_flags=0, reviewed_benign_source_flags=reviewed_source_flags, reviewed_benign_network_flags=reviewed_network_flags, new_timing_outliers=reviewed_timing_outliers, reviewed_host_sleep_timing_outliers=reviewed_timing_outliers, unreviewed_timing_outliers=0, duplicate_trial_ids=0, run_states=dict(run_states), unstarted_setup_failures_preserved=6, legacy_deepseek_observations_preserved=48, prime_batches_checked=prime_batches_checked, prime_requests_checked=prime_requests_checked, prime_plan_sha256=prime_plan_sha256, result='passed')
record.update(scope='checkpoint' if checkpoint else 'final', expected_complete_points=16, expected_primary_attempts=624, completion_pending=len(trials) != 624)
(evidence / ('checkpoint-audit.json' if checkpoint else 'final-audit.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
