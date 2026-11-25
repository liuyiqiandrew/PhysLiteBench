"""Run one predeclared three-condition block, retaining every outcome."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path('/Users/yp9846/Desktop/PhysLiteBench')
NAME = 'hint-comparison-20261014'
STAGING = ROOT / 'staging' / NAME
RESULTS = ROOT / 'results' / NAME
PRIVATE = Path('/private/tmp/physlite-hints-prime-session')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and not any(x in p.parts for x in ['__pycache__', '.pytest_cache', '.git'])}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def native_settings(trial):
    values = set()
    for path in (trial / 'agent/sessions').rglob('*.jsonl'):
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') == 'turn_context':
                payload = event.get('payload', {})
                model = payload.get('model')
                if model == 'deepseek-v4.1-flash':
                    model = 'deepseek/deepseek-v4.1-flash'
                values.add((model, payload.get('effort')))
    return sorted(values)


def run_trial(block, condition, plan, folder):
    provider = 'prime' if block['model'].startswith('deepseek/') else 'openai'
    frozen = plan['tasks'][block['task']]['runtime'][provider + '/' + condition]
    source = ROOT / frozen['path']
    assert hashes(source) == frozen['sha256'], 'Frozen task has drifted'
    label = f"block-{block['index']:03d}-{condition}"
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    job = f"hints-{label}-{block['task']}-{stamp}"
    record = {**block, 'condition': condition, 'provider': provider, 'job': job,
              'reasoning_effort': 'high', 'started_utc': now(), 'status': 'running',
              'source_sha256': frozen['sha256'], 'plan_sha256': sha(RESULTS / 'plan.json')}
    record_path = folder / (condition + '.json')
    write(record_path, record)
    runner = STAGING / 'tooling' / f'run_science_{provider}.py'
    command = [sys.executable, str(runner), str(source), '--job-name', job,
               '--model', block['model'], '--reasoning-effort', 'high',
               '--trials', '1', '--concurrency', '1', '--agent-setup-timeout-multiplier', '3',
               '--prompt-condition', condition]
    if provider == 'prime':
        config_path = folder / (condition + '-prime.toml')
        text = (STAGING / 'prime/codex.toml').read_text()
        text += '\nhttp_headers = { "X-Physlite-Trial" = "' + label + '" }\n'
        config_path.write_text(text)
        command += ['--prime-config', str(config_path)]
        record['prime_trial_tag'] = label
        record['prime_config_sha256'] = sha(config_path)
        write(record_path, record)
    env = os.environ.copy()
    env.update(PATH=str(ROOT / '.cache/bin') + ':' + env['PATH'],
               UV_CACHE_DIR='/private/tmp/physlitebench-inspection-uv',
               UV_PYTHON_INSTALL_DIR=str(ROOT / '.cache/python'))
    with (folder / (condition + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    outcomes = []
    for path in sorted((ROOT / 'jobs' / job).glob('*/result.json')):
        raw = json.loads(path.read_text())
        settings = native_settings(path.parent)
        assert not settings or settings == [(block['model'], 'high')], (path.parent.name, settings)
        exception = raw.get('exception_info') or {}
        outcomes.append({'trial': path.parent.name,
                         'result_path': str(path.relative_to(ROOT)),
                         'result_sha256': sha(path),
                         'reward': (raw.get('verifier_result') or {}).get('rewards', {}).get('reward'),
                         'exception_type': exception.get('exception_type'),
                         'exception_message': exception.get('exception_message'),
                         'model_effort': settings,
                         'agent_execution': raw.get('agent_execution')})
    actual = ROOT / 'jobs' / job / 'frozen-task'
    assert actual.exists() and hashes(actual) == frozen['sha256'], 'Runtime snapshot mismatch'
    assert hashes(source) == frozen['sha256'], 'Frozen source changed while running'
    errors = [x for x in outcomes if x['exception_type']]
    known = all(x['exception_type'] == 'AgentTimeoutError' for x in errors)
    record.update(finished_utc=now(), returncode=result.returncode, outcomes=outcomes,
                  passes=sum(x['reward'] == 1 and not x['exception_type'] for x in outcomes),
                  errors=len(errors), status='completed_with_errors' if errors else 'completed')
    if result.returncode or len(outcomes) != 1 or not known or not all(x['model_effort'] for x in outcomes):
        record['status'] = 'needs_review'
    write(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=int)
    args = parser.parse_args()
    plan = json.loads((RESULTS / 'plan.json').read_text())
    block = next(b for b in plan['blocks'] if b['index'] == args.index)
    authorization = json.loads((RESULTS / 'budget-authorization.json').read_text())
    assert authorization['account_cap_used_percent'] is None and authorization['prime_cap_usd'] is None
    for relative, expected in plan['tooling_sha256'].items():
        assert sha(STAGING / relative) == expected, ('Tooling drift', relative)
    readiness = subprocess.check_output(['ioreg', '-r', '-k', 'AppleClamshellState', '-d', '4'], text=True)
    assert '"AppleClamshellState" = No' in readiness, 'Laptop lid is closed; no new trials launched'
    subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'], check=True, stdout=subprocess.DEVNULL)
    folder = RESULTS / 'blocks' / f"{block['index']:03d}"
    with (STAGING / 'launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in (RESULTS / 'blocks').glob('*/block.json'):
            assert json.loads(path.read_text())['status'] != 'running', 'Another block is active'
        folder.mkdir(parents=True)
        summary = {**block, 'status': 'running', 'started_utc': now()}
        write(folder / 'block.json', summary)
    prime = block['model'].startswith('deepseek/')
    if prime:
        ledger = json.loads((PRIVATE / 'ledger.json').read_text())
        assert not any(x['status'] == 'pending' for x in ledger['requests'])
        summary['prime_request_start'] = len(ledger['requests'])
        write(folder / 'block.json', summary)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_trial, block, condition, plan, folder) for condition in block['condition_order']]
        records = [future.result() for future in futures]
    if prime:
        deadline = time.monotonic() + 190
        while True:
            ledger = json.loads((PRIVATE / 'ledger.json').read_text())
            paid = ledger['requests'][summary['prime_request_start']:]
            if not any(x['status'] == 'pending' for x in paid) or time.monotonic() >= deadline:
                break
            time.sleep(2)
        assert all(x.get('provider_reasoning_effort') == 'high' for x in paid), 'Provider effort mismatch'
        tags = {record['prime_trial_tag'] for record in records}
        assert all(x.get('trial_tag') in tags for x in paid), 'Unattributed Prime request'
        for record in records:
            requests = [x for x in paid if x['trial_tag'] == record['prime_trial_tag']]
            record['prime_requests_settled'] = not any(x['status'] == 'pending' for x in requests)
            record['prime_cost_range_usd'] = [sum(max(x.get('estimated_token_cost_usd') or 0, x.get('reported_cost_usd') or 0) for x in requests), sum(x['accounted_usd'] for x in requests)]
            if not requests or not record['prime_requests_settled']:
                record['status'] = 'needs_review'
            write(folder / (record['condition'] + '.json'), record)
            write(folder / (record['condition'] + '-prime-ledger.json'), {'job': record['job'], 'requests': requests})
    summary.update(finished_utc=now(), status='needs_review' if any(x['status'] == 'needs_review' for x in records) else 'completed',
                   passes={x['condition']: x['passes'] for x in records}, errors={x['condition']: x['errors'] for x in records})
    write(folder / 'block.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['condition_order']}), flush=True)


if __name__ == '__main__':
    main()
