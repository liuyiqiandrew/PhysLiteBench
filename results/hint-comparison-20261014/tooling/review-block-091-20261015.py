"""Review a terminal verifier timeout without changing its outcome or limits."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess

ROOT = Path('/Users/yp9846/Desktop/PhysLiteBench')
RESULTS = ROOT / 'results/hint-comparison-20261014'
STAGING = ROOT / 'staging/hint-comparison-20261014'

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
ledger = json.loads(Path('/private/tmp/physlite-hints-prime-session/ledger.json').read_text())
assert not any(x['status'] == 'pending' for x in ledger['requests'])

folder = RESULTS / 'blocks/091'
block = json.loads((folder / 'block.json').read_text())
assert block['status'] == 'needs_review'
records = []
for condition in plan['conditions']:
    run = json.loads((folder / (condition + '.json')).read_text())
    assert len(run['outcomes']) == 1
    outcome = run['outcomes'][0]
    trial = (ROOT / outcome['result_path']).parent
    assert sha(trial / 'result.json') == outcome['result_sha256']
    assert outcome['model_effort'] == [['gpt-5.6-terra', 'high']]
    raw = json.loads((trial / 'result.json').read_text())
    contexts = set()
    for f in (trial / 'agent/sessions').rglob('*.jsonl'):
        for line in f.read_text().splitlines():
            event = json.loads(line)
            if event.get('type') == 'turn_context':
                payload = event['payload']
                contexts.add((payload.get('model'), payload.get('effort')))
    assert contexts == {('gpt-5.6-terra', 'high')}
    events = [json.loads(line) for line in (trial / 'agent/codex.txt').read_text().splitlines()
              if line.startswith('{')]
    assert any(e.get('type') == 'turn.completed' for e in events)
    if condition == 'general':
        assert outcome['exception_type'] == 'VerifierTimeoutError'
        assert outcome['reward'] is None and raw['verifier_result'] is None
        assert not (trial / 'verifier/reward.txt').exists()
        assert json.loads((trial / 'verifier/metrics.json').read_text()) == {'error': 'metrics unavailable'}
        elapsed = (datetime.fromisoformat(raw['verifier']['finished_at'].replace('Z', '+00:00')) -
                   datetime.fromisoformat(raw['verifier']['started_at'].replace('Z', '+00:00'))).total_seconds()
        assert 60 <= elapsed < 61
    else:
        assert outcome['exception_type'] is None
        assert outcome['reward'] == (1 if condition == 'physics' else 0)
    records.append({'condition': condition, 'trial': outcome['trial'],
                    'result_sha256': outcome['result_sha256'], 'exception_type': outcome['exception_type'],
                    'reward': outcome['reward'], 'agent_turn_completed': True,
                    'verifier_file_hashes': {f.name: sha(f) for f in (trial / 'verifier').iterdir() if f.is_file()}})

history = RESULTS / 'history/block-091-before-review-20261015'
assert not history.exists()
history.mkdir(parents=True)
shutil.copytree(folder, history / 'block-records')
for name in ['plan.json', 'results.json', 'status.json', 'summary.csv', 'REPORT.txt']:
    shutil.copy2(RESULTS / name, history / name)
now = datetime.now(timezone.utc).isoformat()
review_path = RESULTS / 'reviews/block-091-review-20261015.json'
review = {'reviewed_utc': now, 'block': 91,
          'cause_of_hold': 'General-hint native agent turn completed, then the verifier exceeded its frozen 60-second limit. No reward was produced.',
          'root_cause_limit': 'Retained verifier output is three progress dots with metrics unavailable. No profiling establishes which computation exhausted the verifier limit.',
          'submission_observation': 'Retained submission includes numerical BDF integration. It was read but not executed or changed during review.',
          'resumption_basis': 'All three native agent turns ended, no containers or model jobs remain active, and the other two verifier runs finished. The terminal timeout is understood and preserved as an execution error.',
          'records': records, 'general_verifier_elapsed_seconds': elapsed,
          'original_outcomes_preserved': True, 'retries': 0, 'replacements': 0,
          'verifier_reruns': 0, 'timeout_changed': False,
          'task_prompt_grading_limits_and_order_unchanged': True,
          'plan_sha256': plan_hash, 'resume_allowed': True, 'next_unstarted_block': 93,
          'physics_failure_attribution': False, 'history': str(history.relative_to(ROOT))}
write(review_path, review)
path = folder / 'general.json'
run = json.loads(path.read_text())
run.update(status='completed_with_errors', operational_review=str(review_path.relative_to(ROOT)))
write(path, run)
block.update(status='completed', reviewed_utc=now, operational_review=str(review_path.relative_to(ROOT)))
write(folder / 'block.json', block)
assert sha(RESULTS / 'plan.json') == plan_hash
shutil.copy2(__file__, RESULTS / 'tooling/review-block-091-20261015.py')
print(json.dumps({'review': str(review_path), 'outcomes_preserved': 3,
                  'next_unstarted_block': 93, 'plan_unchanged': True}))
