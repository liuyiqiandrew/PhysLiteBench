"""Collect local observations and explicitly authorized historical baseline slots."""
from pathlib import Path
from datetime import datetime, timezone
import collections
import csv
import difflib
import hashlib
import importlib.util
import json
import shutil

ROOT = Path('/Users/yp9846/Desktop/PhysLiteBench')
NAME = 'hint-comparison-20261014'
STAGING = ROOT / 'staging' / NAME
RESULTS = ROOT / 'results' / NAME


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((RESULTS / 'plan.json').read_text())
    prices = json.loads((RESULTS / 'pricing.json').read_text())
    spec = importlib.util.spec_from_file_location('prior_collector', STAGING / 'tooling/prior_collector.py')
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    rows, audit, blocks = [], [], []
    for folder in sorted((RESULTS / 'blocks').glob('*')):
        blockfile = folder / 'block.json'
        if blockfile.exists():
            blocks.append(json.loads(blockfile.read_text()))
        for condition in plan['conditions']:
            path = folder / (condition + '.json')
            if not path.exists():
                continue
            run = json.loads(path.read_text())
            for outcome in run.get('outcomes', []):
                trial = (ROOT / outcome['result_path']).parent
                row = prior.trial_record(run['task'], run['model'], trial, prices)
                row.update(condition=condition, repeat=run['repeat'], block=run['index'],
                           reasoning_effort='high', reused=False, codex_cli=plan['codex_cli'], source_cohort=NAME, observed_model_effort=outcome['model_effort'],
                           run_record=str(path.relative_to(ROOT)), execution_error=outcome['exception_type'])
                if run.get('prime_cost_range_usd'):
                    low, high = run['prime_cost_range_usd']
                    row['cost_range_usd'] = [low, high]
                    # A closed connection does not establish billed usage. An
                    # unconsumed spending reserve is an upper bound, not a cost.
                    complete_usage = bool(run['prime_requests_settled']) and abs(high - low) < 1e-12
                    row['known_recorded_api_usd'] = low
                    row['estimated_standard_api_usd'] = high if complete_usage else None
                    row['cost_reconciled'] = complete_usage
                    row['usage_complete'] = complete_usage
                row['execution_cohort'] = run.get('execution_cohort', 'serial-three-trials')
                row['max_concurrent_trials'] = run.get('max_concurrent_trials', 3)
                row['cpu_set'] = run.get('cpu_set')
                row['completed_pass'] = row['reward'] == 1 and row['exception'] is None
                dst = RESULTS / 'evidence' / f"{run['index']:03d}" / condition / trial.name
                dst.mkdir(parents=True, exist_ok=True)
                for source, name in [('artifacts/app/model.py', 'model.py'), ('verifier/metrics.json', 'metrics.json'), ('verifier/reward.txt', 'reward.txt'), ('verifier/test-stdout.txt', 'test-stdout.txt')]:
                    if (trial / source).exists():
                        shutil.copyfile(trial / source, dst / name)
                if (dst / 'model.py').exists():
                    before = (trial.parent / 'frozen-task/environment/model.py').read_text().splitlines(True)
                    after = (dst / 'model.py').read_text().splitlines(True)
                    (dst / 'model.diff').write_text(''.join(difflib.unified_diff(before, after, fromfile='starter/model.py', tofile='submission/model.py')))
                row['evidence'] = str(dst.relative_to(ROOT))
                audit.append({'trial': trial.name, 'observed_model_effort': outcome['model_effort'], 'expected_model_effort': [run['model'], 'high'], 'file_hashes': {p.name: sha(p) for p in dst.iterdir() if p.is_file()}})
                rows.append(row)
    # Retain started slots whose native runner record was lost. These are explicit
    # administrative execution failures, never reconstructed verifier outcomes.
    if plan.get('administrative_observations'):
        administrative_path = ROOT / plan['administrative_observations']
        assert sha(administrative_path) == plan['administrative_observations_sha256']
        administrative = json.loads(administrative_path.read_text())
        for row in administrative['trials']:
            assert row['administrative_observation'] and row['reward'] is None
            assert row['exception'] and not row['completed_pass'] and not row['verifier_ran']
            assert row['model'] in plan['models'] and row['reasoning_effort'] == 'high'
            rows.append(dict(row))
            audit.append({'trial': row['trial'], 'administrative_observation': True,
                          'recovery_evidence': row['recovery_evidence'], 'native_sources': row['native_sources'],
                          'observed_model_effort': row['observed_model_effort'], 'usage_complete': False})
    reuse = json.loads((ROOT / plan['reuse_manifest']).read_text()) if plan.get('reuse_manifest') else {'trials': [], 'source_file_hashes': {}}
    if plan.get('reuse_manifest'):
        assert sha(ROOT / plan['reuse_manifest']) == plan['reuse_manifest_sha256'], 'Reused baseline manifest drift'
    for path, expected in reuse['source_file_hashes'].items():
        assert sha(ROOT / path) == expected, ('Imported evidence drift', path)
    for row in reuse['trials']:
        rows.append(dict(row))
        audit.append({'trial': row['trial'], 'reused': True, 'verification_basis': row['verification_basis'], 'source_csv': row['source_csv'], 'source_remote_commit': row['source_remote_commit']})
    assert len({row['trial'] for row in rows}) == len(rows), 'Duplicate trial' 
    assert len({(row['model'], row['task'], row['condition'], row['repeat']) for row in rows}) == len(rows), 'Duplicate experimental slot'
    summary = []
    for model in plan['models']:
        for condition in plan['conditions']:
            selected = [r for r in rows if r['model'] == model and r['condition'] == condition]
            counts = collections.Counter(r['task'] for r in selected)
            costs = [r['estimated_standard_api_usd'] for r in selected if r['estimated_standard_api_usd'] is not None]
            summary.append({'model': model, 'condition': condition, 'attempts': len(selected), 'expected': 39,
                            'reused_attempts': sum(r['reused'] for r in selected),
                            'passes': sum(r['completed_pass'] for r in selected),
                            'checkpoint_passes': sum(r['reward'] == 1 for r in selected),
                            'errors': sum(r['exception'] is not None for r in selected),
                            'complete': all(counts[name] == 3 for name in plan['tasks']),
                            'mean_estimated_api_usd': sum(costs) / len(costs) if costs and len(costs) == len(selected) else None,
                            'per_task': {task: {'attempts': counts[task], 'passes': sum(r['completed_pass'] for r in selected if r['task'] == task), 'errors': sum(r['exception'] is not None for r in selected if r['task'] == task)} for task in plan['tasks']}})
    complete = len(rows) == plan['expected_attempts'] and all(r['complete'] for r in summary)
    status = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'state': 'complete' if complete else 'in_progress',
              'recorded_attempts': len(rows), 'expected_attempts': plan['expected_attempts'],
              'fresh_attempts': sum(not r['reused'] for r in rows), 'reused_attempts': sum(r['reused'] for r in rows),
              'planned_fresh_attempts': plan.get('planned_fresh_attempts', plan['expected_attempts']),
              'remaining_fresh_attempts': plan.get('planned_fresh_attempts', plan['expected_attempts']) - sum(not r['reused'] for r in rows),
              'completed_blocks': sum(b['status'] == 'completed' for b in blocks),
              'review_blocks': [b['index'] for b in blocks if b['status'] == 'needs_review'],
              'active_blocks': [b['index'] for b in blocks if b['status'] == 'running'],
              'complete_model_condition_panels': sum(x['complete'] for x in summary)}
    if status['review_blocks']:
        status['state'] = 'needs_review'
    for filename, data in [('results.json', {'status': status, 'summary': summary, 'trials': rows}), ('evidence-audit.json', audit), ('status.json', status)]:
        (RESULTS / filename).write_text(json.dumps(data, indent=2) + '\n')
    fields = ['model', 'condition', 'attempts', 'expected', 'passes', 'checkpoint_passes', 'errors', 'complete', 'mean_estimated_api_usd', 'reused_attempts']
    with (RESULTS / 'summary.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)
    lines = ['Physics hint comparison', '', f"{len(plan['models'])} fixed models, High effort, three observations per condition and task; explicitly labeled historical baseline reuse.",
             'Baseline: unchanged instruction. General: fixed correctness reminder. Physics: fixed task-specific conceptual guidance.',
             'Frozen task and grading hashes match. Luna historical-baseline comparisons also differ in client version, concurrency and date. All execution errors are retained separately.',
             plan.get('reuse_caveat', 'Historical task-selection results are not reused.'),
             plan.get('execution_resource_caveat', ''),
             'Hint length is not matched. Historical and fresh evidence are distinguished.',
             '', f"Progress: {len(rows)}/{plan['expected_attempts']} attempts; {status['complete_model_condition_panels']}/{len(plan['models'])*len(plan['conditions'])} complete panels.", '']
    lines.append(f"Fresh attempts: {status['fresh_attempts']}/{status['planned_fresh_attempts']}; reused baseline observations: {status['reused_attempts']}; remaining fresh attempts: {status['remaining_fresh_attempts']}.")
    for row in summary:
        lines.append(f"{row['model']} | {row['condition']}: {row['passes']}/{row['attempts']} completed passes, {row['errors']} execution errors; planned 39 attempts.")
    (RESULTS / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()
