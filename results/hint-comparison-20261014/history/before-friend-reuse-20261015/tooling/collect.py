"""Collect new hint-comparison observations without reusing historical scores."""
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
                           reasoning_effort='high', reused=False, observed_model_effort=outcome['model_effort'],
                           run_record=str(path.relative_to(ROOT)), execution_error=outcome['exception_type'])
                if run.get('prime_cost_range_usd'):
                    row['cost_range_usd'] = run['prime_cost_range_usd']
                    row['estimated_standard_api_usd'] = run['prime_cost_range_usd'][1]
                    row['cost_reconciled'] = run['prime_requests_settled']
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
    assert len({row['trial'] for row in rows}) == len(rows), 'Duplicate trial'
    assert len({(row['model'], row['task'], row['condition'], row['repeat']) for row in rows}) == len(rows), 'Duplicate experimental slot'
    summary = []
    for model in plan['models']:
        for condition in plan['conditions']:
            selected = [r for r in rows if r['model'] == model and r['condition'] == condition]
            counts = collections.Counter(r['task'] for r in selected)
            costs = [r['estimated_standard_api_usd'] for r in selected if r['estimated_standard_api_usd'] is not None]
            summary.append({'model': model, 'condition': condition, 'attempts': len(selected), 'expected': 39,
                            'passes': sum(r['completed_pass'] for r in selected),
                            'checkpoint_passes': sum(r['reward'] == 1 for r in selected),
                            'errors': sum(r['exception'] is not None for r in selected),
                            'complete': all(counts[name] == 3 for name in plan['tasks']),
                            'mean_estimated_api_usd': sum(costs) / len(costs) if costs and len(costs) == len(selected) else None,
                            'per_task': {task: {'attempts': counts[task], 'passes': sum(r['completed_pass'] for r in selected if r['task'] == task), 'errors': sum(r['exception'] is not None for r in selected if r['task'] == task)} for task in plan['tasks']}})
    complete = len(rows) == plan['expected_attempts'] and all(r['complete'] for r in summary)
    status = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'state': 'complete' if complete else 'in_progress',
              'recorded_attempts': len(rows), 'expected_attempts': plan['expected_attempts'],
              'completed_blocks': sum(b['status'] == 'completed' for b in blocks),
              'review_blocks': [b['index'] for b in blocks if b['status'] == 'needs_review'],
              'active_blocks': [b['index'] for b in blocks if b['status'] == 'running'],
              'complete_model_condition_panels': sum(x['complete'] for x in summary)}
    for filename, data in [('results.json', {'status': status, 'summary': summary, 'trials': rows}), ('evidence-audit.json', audit), ('status.json', status)]:
        (RESULTS / filename).write_text(json.dumps(data, indent=2) + '\n')
    fields = ['model', 'condition', 'attempts', 'expected', 'passes', 'checkpoint_passes', 'errors', 'complete', 'mean_estimated_api_usd']
    with (RESULTS / 'summary.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)
    lines = ['Physics hint comparison', '', f"{len(plan['models'])} fixed models, High effort, three fresh repeats per condition and task.",
             'Baseline: unchanged instruction. General: fixed correctness reminder. Physics: fixed task-specific conceptual guidance.',
             'Only the instruction differs within each model/task comparison. All execution errors are retained separately.',
             'Historical task-selection results are not reused. This experiment measures these hint packages; hint length is not matched.',
             '', f"Progress: {len(rows)}/{plan['expected_attempts']} attempts; {status['complete_model_condition_panels']}/{len(plan['models'])*len(plan['conditions'])} complete panels.", '']
    for row in summary:
        lines.append(f"{row['model']} | {row['condition']}: {row['passes']}/{row['attempts']} completed passes, {row['errors']} execution errors; planned 39 attempts.")
    (RESULTS / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()
