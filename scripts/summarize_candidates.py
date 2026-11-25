"""Summarize completed candidate batches; require separate human/agent failure review."""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def digest_tree(folder):
    sha = hashlib.sha256()
    for path in sorted(folder.rglob('*')):
        if path.is_file() and not any(part in ('__pycache__', '.pytest_cache', '.git') for part in path.parts):
            sha.update(str(path.relative_to(folder)).encode()+b'\0'+path.read_bytes()+b'\0')
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/candidates.json')
    args = parser.parse_args()
    review_path = ROOT/'results/candidate-reviews.json'
    reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
    for path in sorted((ROOT/'results').glob('*-trial-reviews.json')):
        reviews.update(json.loads(path.read_text()))
    batches = []
    for index_file in sorted((ROOT/'jobs').glob('matrix-*/index.json')):
        for run in json.loads(index_file.read_text())['runs']:
            job = ROOT/'jobs'/run['job']
            if not (job/'run-timing.json').exists():
                continue
            task = job/'frozen-task'
            config = tomllib.loads((task/'task.toml').read_text())
            batch = {**run, 'index':str(index_file.relative_to(ROOT)),
                     'environment_sha256':digest_tree(task/'environment'),
                     'tests_sha256':digest_tree(task/'tests'),
                     'instruction_sha256':hashlib.sha256((task/'instruction.md').read_bytes()).hexdigest(),
                     'task_config_sha256':hashlib.sha256((task/'task.toml').read_bytes()).hexdigest(),
                     'timeouts':{'agent':config['agent']['timeout_sec'], 'verifier':config['verifier']['timeout_sec']},
                     'trials':[]}
            for trial_file in sorted(job.glob('*/result.json')):
                result = json.loads(trial_file.read_text())
                trial = trial_file.parent
                contexts = set()
                versions = set()
                for session in (trial/'agent/sessions').rglob('*.jsonl'):
                    for line in session.open():
                        event = json.loads(line)
                        if event.get('type') == 'turn_context':
                            payload = event['payload']
                            contexts.add((payload.get('model'), payload.get('effort')))
                        if event.get('type') == 'session_meta':
                            versions.add(event['payload'].get('cli_version'))
                reward = (result.get('verifier_result') or {}).get('rewards', {}).get('reward')
                entry = {'trial':trial.name, 'reward':reward,
                         'exception':result.get('exception_info'),
                         'agent_started':result.get('agent_execution') is not None,
                         'native_model_effort':sorted([list(c) for c in contexts]),
                         'native_cli_versions':sorted(v for v in versions if v),
                         'artifacts':str(trial.relative_to(ROOT))}
                metrics = trial/'verifier/metrics.json'
                if metrics.exists():
                    entry['metrics'] = json.loads(metrics.read_text())
                if run['condition'] in ('plain', 'hint'):
                    entry['review'] = reviews.get(trial.name, {'classification':'pending'})
                batch['trials'].append(entry)
            batches.append(batch)
    comparisons = []
    for plain in batches:
        if plain['condition'] != 'plain':
            continue
        hints = [b for b in batches if b['task'] == plain['task'] and b['condition'] == 'hint'
                 and b['index'] == plain['index']]
        for hint in hints:
            original = ROOT/'jobs'/plain['job']/'frozen-task'
            hinted = ROOT/'jobs'/hint['job']/'frozen-task'
            expected = (original/'instruction.md').read_text().rstrip()+'\n\nPhysics hint:\n'+(original/'hint.md').read_text()
            complete = (len(plain['trials']) == len(hint['trials']) == 3
                        and all(t['reward'] in (0, 1) and t['exception'] is None
                                for t in plain['trials']+hint['trials']))
            comparisons.append({'task':plain['task'], 'plain_job':plain['job'], 'hint_job':hint['job'],
                'complete_three_trial_pair':complete,
                'environment_identical':plain['environment_sha256'] == hint['environment_sha256'],
                'tests_identical':plain['tests_sha256'] == hint['tests_sha256'],
                'task_config_identical':plain['task_config_sha256'] == hint['task_config_sha256'],
                'timeouts_verified':plain['timeouts'] == hint['timeouts'] == {'agent':600., 'verifier':60.},
                'instruction_only_appends_hint':(hinted/'instruction.md').read_text() == expected,
                'native_model_effort_verified':complete and all(t['native_model_effort'] == [['gpt-5.6-luna', 'high']]
                                                   for t in plain['trials']+hint['trials']),
                'native_cli_verified':complete and all(t['native_cli_versions'] == ['0.154.0']
                                          for t in plain['trials']+hint['trials'])})
    args.output.write_text(json.dumps({'batches':batches, 'paired_checks':comparisons}, indent=2)+'\n')
    for b in batches:
        if b['condition'] in ('plain', 'hint'):
            passed = sum(t['reward'] == 1 for t in b['trials'])
            print(b['task'], b['condition'], f'{passed}/{len(b["trials"])}', b['job'])


if __name__ == '__main__':
    main()
