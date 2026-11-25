"""Preserve block 99 outcomes and review its terminal transport failure."""
from pathlib import Path
from datetime import datetime, timezone
import collections
import hashlib
import json
import shutil
import subprocess

ROOT = Path('/Users/yp9846/Desktop/PhysLiteBench')
RESULTS = ROOT / 'results/hint-comparison-20261014'
STAGING = ROOT / 'staging/hint-comparison-20261014'
PRIVATE = Path('/private/tmp/physlite-hints-prime-session')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)

plan = json.loads((RESULTS / 'plan.json').read_text())
plan_hash = sha(RESULTS / 'plan.json')
for relative, expected in plan['tooling_sha256'].items():
    assert sha(STAGING / relative) == expected
for task, spec in plan['tasks'].items():
    for relative, expected in spec['source_sha256'].items():
        assert sha(ROOT / 'tasks' / task / relative) == expected
    for runtime in spec['runtime'].values():
        for relative, expected in runtime['sha256'].items():
            assert sha(ROOT / runtime['path'] / relative) == expected
processes = subprocess.check_output(['ps', '-axo', 'command'], text=True)
assert str(STAGING / 'tooling/controller.py') not in processes
assert str(STAGING / 'tooling/run_block.py') not in processes
assert str(STAGING / 'prime/prime_proxy_v2.py') in processes
assert not subprocess.check_output(['/Users/yp9846/.docker/bin/docker', 'ps', '-q'], text=True).strip()
assert not any(json.loads(f.read_text())['status'] == 'running'
               for f in (RESULTS / 'blocks').glob('*/block.json'))

folder = RESULTS / 'blocks/099'
block = json.loads((folder / 'block.json').read_text())
assert block['status'] == 'needs_review'
ledger = json.loads((PRIVATE / 'ledger.json').read_text())
assert not any(x['status'] == 'pending' for x in ledger['requests'])
requests = ledger['requests'][block['prime_request_start']:]
assert len(requests) == 37
assert all(x['provider_reasoning_effort'] == 'high' and
           x['model'] == block['model'] for x in requests)
errors = [x for x in requests if x['status'] != 'completed']
assert len(errors) == 1 and errors[0]['status'] == 'transport_error'
assert errors[0]['error'] == '[Errno 54] Connection reset by peer'
later = [x for x in requests if x['status'] == 'completed' and
         x['finished_at'] > errors[0]['finished_at']]
assert len(later) >= 1

records = []
for condition in plan['conditions']:
    run = json.loads((folder / (condition + '.json')).read_text())
    assert len(run['outcomes']) == 1
    outcome = run['outcomes'][0]
    trial = (ROOT / outcome['result_path']).parent
    assert sha(trial / 'result.json') == outcome['result_sha256']
    assert outcome['model_effort'] == [[block['model'], 'high']]
    expected_error = 'NonZeroAgentExitCodeError' if condition == 'physics' else 'AgentTimeoutError'
    assert outcome['exception_type'] == expected_error
    contexts = set()
    for f in (trial / 'agent/sessions').rglob('*.jsonl'):
        for line in f.read_text().splitlines():
            event = json.loads(line)
            if event.get('type') == 'turn_context':
                payload = event['payload']
                contexts.add((payload.get('model'), payload.get('effort')))
    assert contexts == {('deepseek-v4.1-flash', 'high')}
    events = [json.loads(line) for line in (trial / 'agent/codex.txt').read_text().splitlines()
              if line.startswith('{')]
    terminal = [e for e in events if e.get('type') in ['error', 'turn.failed']]
    if condition == 'physics':
        assert any('stream closed before response.completed' in json.dumps(e) for e in terminal)
    selected = [x for x in requests if x['trial_tag'] == run['prime_trial_tag']]
    bounds = [sum(max(x.get('estimated_token_cost_usd') or 0, x.get('reported_cost_usd') or 0)
                  for x in selected), sum(x['accounted_usd'] for x in selected)]
    assert all(abs(a - b) < 1e-12 for a, b in zip(bounds, run['prime_cost_range_usd']))
    records.append({'condition': condition, 'trial': outcome['trial'],
                    'result_sha256': outcome['result_sha256'], 'exception_type': expected_error,
                    'terminal_events': terminal, 'request_statuses': dict(collections.Counter(x['status'] for x in selected)),
                    'cost_range_usd': bounds, 'exact_usage_available': condition != 'physics'})

history = RESULTS / 'history/block-099-before-review-20261015'
assert not history.exists()
history.mkdir(parents=True)
shutil.copytree(folder, history / 'block-records')
for name in ['plan.json', 'results.json', 'status.json', 'summary.csv', 'REPORT.txt']:
    shutil.copy2(RESULTS / name, history / name)
now = datetime.now(timezone.utc).isoformat()
review_path = RESULTS / 'reviews/block-099-review-20261015.json'
review = {'reviewed_utc': now, 'block': 99,
          'cause_of_hold': 'The physics-hint native agent stream terminated after an upstream connection reset. The baseline and general attempts reached the unchanged 600-second agent limit.',
          'provider_root_cause': 'The retained transport error identifies a connection reset, but does not establish its underlying cause.',
          'resumption_basis': 'Relay remains alive, no pending requests or active containers remain, and subsequent provider requests completed after the connection error. Resume the next unstarted fixed block.',
          'records': records, 'successful_provider_requests_after_error': len(later),
          'original_outcomes_preserved': True, 'retries': 0, 'replacements': 0,
          'task_prompt_grading_limits_and_order_unchanged': True,
          'plan_sha256': plan_hash, 'resume_allowed': True, 'next_unstarted_block': 100,
          'accounting': 'The physics-hint request lacks usage. Its conservative reserve remains an upper bound, not confirmed spending. There are now seven cost-uncertain observations.',
          'physics_failure_attribution': False, 'history': str(history.relative_to(ROOT))}
write(review_path, review)
for condition in plan['conditions']:
    path = folder / (condition + '.json')
    run = json.loads(path.read_text())
    run.update(status='completed_with_errors', operational_review=str(review_path.relative_to(ROOT)))
    if condition == 'physics':
        run.update(prime_usage_complete=False,
                   accounting_note='Terminal transport error with unknown actual charge; preserve lower known cost and upper reserve.')
    write(path, run)
block.update(status='completed', reviewed_utc=now, operational_review=str(review_path.relative_to(ROOT)))
write(folder / 'block.json', block)
assert sha(RESULTS / 'plan.json') == plan_hash
shutil.copy2(__file__, RESULTS / 'tooling/review-block-099-20261015.py')
print(json.dumps({'review': str(review_path), 'outcomes_preserved': 3,
                  'next_unstarted_block': 100, 'plan_unchanged': True}))
