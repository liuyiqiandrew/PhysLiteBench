"""Reconcile late Prime usage without replacing any block 186 outcome."""
from pathlib import Path
from datetime import datetime, timezone
import collections
import hashlib
import json
import shutil
import subprocess

R = Path('/Users/yp9846/Desktop/PhysLiteBench')
B = R / 'results/hint-comparison-20261014'
S = R / 'staging/hint-comparison-20261014'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

p = json.loads((B / 'plan.json').read_text())
plan_hash = sha(B / 'plan.json')
assert plan_hash == '336d3c1f3a65388570b2c35655d3e46dd4cc447f9ee82bd9c52e695f44619d42'
for rel, expected in p['tooling_sha256'].items():
    assert sha(S / rel) == expected
for task, spec in p['tasks'].items():
    for rel, expected in spec['source_sha256'].items():
        assert sha(R / 'tasks' / task / rel) == expected
    for runtime in spec['runtime'].values():
        for rel, expected in runtime['sha256'].items():
            assert sha(R / runtime['path'] / rel) == expected
processes = subprocess.check_output(['ps', '-axo', 'command'], text=True)
assert str(S / 'tooling/controller.py') not in processes
assert str(S / 'tooling/run_block.py') not in processes
assert str(S / 'prime/prime_proxy_v2.py') in processes
assert not subprocess.check_output(['/Users/yp9846/.docker/bin/docker', 'ps', '-q'], text=True).strip()
assert not any(json.loads(f.read_text())['status'] == 'running' for f in (B / 'blocks').glob('*/block.json'))
assert not (B / 'monitoring/STOP_PARALLEL_AFTER_DRAIN').exists()

folder = B / 'blocks/186'
block = json.loads((folder / 'block.json').read_text())
assert block['status'] == 'needs_review'
ledger = json.loads(Path('/private/tmp/physlite-hints-prime-session/ledger.json').read_text())
assert not any(x['status'] == 'pending' for x in ledger['requests'])
requests = ledger['requests'][block['prime_request_start']:]
assert len(requests) == 38
assert all(x['status'] == 'completed' and x['model'] == block['model'] and x['provider_reasoning_effort'] == 'high' for x in requests)
assert {x['trial_tag'] for x in requests} == {f'block-186-{c}' for c in p['conditions']}
late = next(x for x in requests if x['id'] == 'c8f043242ab442a9a425b1ca8315f7e2')
assert late['finished_at'] > block['finished_utc']
records = []
runs = {}
snapshots = {}
for condition in p['conditions']:
    run = json.loads((folder / (condition + '.json')).read_text())
    assert len(run['outcomes']) == 1
    outcome = run['outcomes'][0]
    trial = (R / outcome['result_path']).parent
    assert sha(trial / 'result.json') == outcome['result_sha256']
    assert outcome['exception_type'] == 'AgentTimeoutError'
    assert outcome['exception_message'] == 'Agent execution timed out after 600.0 seconds'
    assert outcome['model_effort'] == [[block['model'], 'high']]
    contexts = set()
    for f in (trial / 'agent/sessions').rglob('*.jsonl'):
        for line in f.read_text().splitlines():
            event = json.loads(line)
            if event.get('type') == 'turn_context':
                contexts.add((event['payload'].get('model'), event['payload'].get('effort')))
    assert contexts == {('deepseek-v4.1-flash', 'high')}
    events = [json.loads(line) for line in (trial / 'agent/codex.txt').read_text().splitlines() if line.startswith('{')]
    selected = [x for x in requests if x['trial_tag'] == run['prime_trial_tag']]
    old = json.loads((folder / (condition + '-prime-ledger.json')).read_text())
    assert {x['id'] for x in old['requests']} == {x['id'] for x in selected}
    bounds = [sum(max(x.get('estimated_token_cost_usd') or 0, x.get('reported_cost_usd') or 0) for x in selected), sum(x['accounted_usd'] for x in selected)]
    assert abs(bounds[1] - bounds[0]) < 1e-12
    if condition != 'physics':
        assert all(abs(a-b)<1e-12 for a,b in zip(bounds, run['prime_cost_range_usd']))
    records.append({'condition': condition, 'trial': outcome['trial'], 'result_sha256': outcome['result_sha256'],
                    'exception_type': outcome['exception_type'], 'retained_reward': outcome['reward'],
                    'observed_model_effort': outcome['model_effort'],
                    'native_event_types': dict(collections.Counter(x.get('type') for x in events)),
                    'provider_request_statuses': dict(collections.Counter(x['status'] for x in selected)),
                    'previous_cost_range_usd': run['prime_cost_range_usd'], 'settled_cost_range_usd': bounds})
    runs[condition] = run
    snapshots[condition] = {'job': run['job'], 'requests': selected}

history = B / 'history/block-186-before-review-20261015'
assert not history.exists()
history.mkdir(parents=True)
shutil.copytree(folder, history / 'block-records')
for name in ['plan.json', 'results.json', 'status.json', 'summary.csv', 'REPORT.txt']:
    shutil.copy2(B / name, history / name)
now = datetime.now(timezone.utc).isoformat()
review_path = B / 'reviews/block-186-review-20261015.json'
review = {'reviewed_utc': now, 'block': 186,
          'cause_of_hold': 'The physics-condition provider request remained pending beyond the 190-second accounting wait after the native 600-second agent timeout.',
          'resolution': 'The original provider request subsequently completed at 19:34:42 UTC and reported usage. All 38 requests are now terminal with exact accounted cost.',
          'resumption_basis': 'No active controller, trials, containers or pending provider requests. Existing relay remains alive. Resume only unstarted blocks with the current parallel controller.',
          'records': records, 'late_request_id': late['id'], 'late_request_finished_at': late['finished_at'],
          'original_outcomes_preserved': True, 'retries': 0, 'replacements': 0, 'physics_failure_attribution': False,
          'tasks_prompts_grading_timeouts_resource_policy_unchanged': True, 'plan_sha256': plan_hash,
          'resume_allowed': True, 'next_unstarted_blocks': [194],
          'accounting': 'The late request now has reported usage, replacing its pending reserve in current accounting. The original pending snapshot is archived. Seven earlier observations retain unknown actual charges and conservative bounds.',
          'history': str(history.relative_to(R))}
write(review_path, review)
for record in records:
    c = record['condition']
    run = runs[c]
    run.update(status='completed_with_errors', prime_requests_settled=True, prime_usage_complete=True,
               prime_cost_range_usd=record['settled_cost_range_usd'], operational_review=str(review_path.relative_to(R)))
    write(folder / (c + '.json'), run)
    write(folder / (c + '-prime-ledger.json'), snapshots[c])
block.update(status='completed', reviewed_utc=now, operational_review=str(review_path.relative_to(R)))
write(folder / 'block.json', block)
assert sha(B / 'plan.json') == plan_hash
shutil.copy2(__file__, B / 'tooling/review-block-186-20261015.py')
print(json.dumps({'review': str(review_path), 'outcomes_preserved': 3, 'plan_unchanged': True, 'records': records}))
