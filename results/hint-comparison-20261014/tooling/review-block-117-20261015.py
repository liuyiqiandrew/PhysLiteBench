"""Settle a terminal pre-agent setup error, preserving the fixed schedule."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import socket
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
dns_resolves = bool(socket.getaddrinfo('nodejs.org', 443))
assert dns_resolves

folder = RESULTS / 'blocks/117'
block = json.loads((folder / 'block.json').read_text())
assert block['status'] == 'needs_review'
assert block['condition_order'] == ['physics', 'general']
assert block['reused_conditions'] == {'baseline': 'swirling-heat-tails__P3nrFaa'}
records = []
for condition in block['condition_order']:
    run = json.loads((folder / (condition + '.json')).read_text())
    assert run['model'] == 'gpt-5.6-luna' and run['reasoning_effort'] == 'high'
    assert run['source_sha256'] == plan['tasks'][block['task']]['runtime']['openai/' + condition]['sha256']
    assert len(run['outcomes']) == 1
    outcome = run['outcomes'][0]
    trial = (ROOT / outcome['result_path']).parent
    assert sha(trial / 'result.json') == outcome['result_sha256']
    raw = json.loads((trial / 'result.json').read_text())
    if condition == 'physics':
        assert outcome['exception_type'] == 'NetworkConnectionError'
        assert 'Could not resolve host: nodejs.org' in outcome['exception_message']
        assert outcome['reward'] is None and raw['verifier_result'] is None
        assert raw['agent_execution'] is None and raw['verifier'] is None
        assert outcome['model_effort'] == []
        assert not (trial / 'agent/trajectory.json').exists()
        assert not (trial / 'verifier/reward.txt').exists()
        assert sha(trial / 'artifacts/app/model.py') == plan['tasks'][block['task']]['source_sha256']['environment/model.py']
    else:
        assert outcome['exception_type'] is None and outcome['reward'] == 0
        assert outcome['model_effort'] == [['gpt-5.6-luna', 'high']]
        contexts = set()
        for f in (trial / 'agent/sessions').rglob('*.jsonl'):
            for line in f.read_text().splitlines():
                event = json.loads(line)
                if event.get('type') == 'turn_context':
                    contexts.add((event['payload'].get('model'), event['payload'].get('effort')))
        assert contexts == {('gpt-5.6-luna', 'high')}
        events = [json.loads(line) for line in (trial / 'agent/codex.txt').read_text().splitlines() if line.startswith('{')]
        assert any(e.get('type') == 'turn.completed' for e in events)
    records.append({'condition': condition, 'trial': outcome['trial'],
                    'result_sha256': outcome['result_sha256'], 'exception_type': outcome['exception_type'],
                    'reward': outcome['reward'], 'agent_started': raw['agent_execution'] is not None,
                    'observed_model_effort': outcome['model_effort'],
                    'agent_setup': raw['agent_setup'], 'agent_execution': raw['agent_execution'],
                    'native_file_hashes': {str(f.relative_to(trial)): sha(f) for f in trial.rglob('*') if f.is_file()}})

history = RESULTS / 'history/block-117-before-review-20261015'
assert not history.exists()
history.mkdir(parents=True)
shutil.copytree(folder, history / 'block-records')
for name in ['plan.json', 'results.json', 'status.json', 'summary.csv', 'REPORT.txt']:
    shutil.copy2(RESULTS / name, history / name)
now = datetime.now(timezone.utc).isoformat()
next_block = next(b['index'] for b in plan['blocks'] if not (RESULTS / 'blocks' / f"{b['index']:03d}" / 'block.json').exists())
review_path = RESULTS / 'reviews/block-117-review-20261015.json'
review = {'reviewed_utc': now, 'block': 117,
          'cause_of_hold': 'Luna physics-hint agent setup could not resolve nodejs.org while downloading Node. Harbor recorded NetworkConnectionError before any model execution or verification.',
          'secondary_error': 'The failed installer also reported TMPDIR unbound; the retained log identifies the preceding DNS download failure.',
          'root_cause_limit': 'The underlying cause of the transient DNS failure is not established.',
          'resumption_basis': 'The same-block general-hint setup later completed, followed by a normal Luna High agent turn and verifier. Host DNS resolves nodejs.org now, no benchmark containers or schedulers remain active, and no Prime requests are pending.',
          'host_nodejs_dns_resolves': dns_resolves,
          'records': records, 'original_outcomes_preserved': True,
          'pre_agent_setup_error_observations': 1, 'physics_grade': None,
          'retries': 0, 'replacements': 0, 'verifier_reruns': 0,
          'task_prompt_grading_limits_and_order_unchanged': True,
          'plan_sha256': plan_hash, 'resume_allowed': True, 'next_unstarted_block': next_block,
          'physics_failure_attribution': False, 'history': str(history.relative_to(ROOT))}
write(review_path, review)
path = folder / 'physics.json'
run = json.loads(path.read_text())
run.update(status='completed_with_errors', operational_review=str(review_path.relative_to(ROOT)))
write(path, run)
block.update(status='completed', reviewed_utc=now, operational_review=str(review_path.relative_to(ROOT)))
write(folder / 'block.json', block)
assert sha(RESULTS / 'plan.json') == plan_hash
shutil.copy2(__file__, RESULTS / 'tooling/review-block-117-20261015.py')
print(json.dumps({'review': str(review_path), 'native_outcomes_preserved': 2,
                  'next_unstarted_block': next_block, 'plan_unchanged': True}))
