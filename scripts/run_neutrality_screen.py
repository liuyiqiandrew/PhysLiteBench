"""Run one plain trial per task, then two more only after an initial pass."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading

from run_candidate_matrix import summary
from summarize_candidates import digest_tree

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tasks', nargs='+')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--label', default='neutral-v1')
    parser.add_argument('--controls', nargs='*', choices=['oracle', 'shortcut'],
                        help='Check these controls first; --controls alone checks both')
    parser.add_argument('--retry-unstarted-job',
                        help='Replace only a fully unstarted setup-failure batch; preserve its attempts')
    args = parser.parse_args()
    retry_trials = []
    retry_initial = False
    if args.retry_unstarted_job:
        retry_initial = '-initial-plain-' in args.retry_unstarted_job
        assert retry_initial or '-followup-plain-' in args.retry_unstarted_job, 'Only unstarted initial/follow-up batches may be replaced'
        if len(args.tasks) != 1 or args.controls is not None:
            parser.error('Setup retries require exactly one task and no controls')
        failed = ROOT/'jobs'/args.retry_unstarted_job
        frozen = failed/'frozen-task'
        current = ROOT/'tasks'/args.tasks[0]
        assert failed.parent == ROOT/'jobs' and frozen.is_dir(), failed
        for directory in ['environment', 'tests']:
            assert digest_tree(current/directory) == digest_tree(frozen/directory), directory
        for name in ['instruction.md', 'task.toml']:
            assert (current/name).read_bytes() == (frozen/name).read_bytes(), name
        for result_file in sorted(failed.glob('*/result.json')):
            result = json.loads(result_file.read_text())
            assert result.get('agent_execution') is None and result.get('verifier_result') is None
            assert (result.get('exception_info') or {}).get('exception_type') == 'NetworkConnectionError'
            assert not list((result_file.parent/'agent/sessions').rglob('*.jsonl'))
            retry_trials.append(result_file.parent.name)
        assert retry_trials, failed
        assert not retry_initial or len(retry_trials) == 1, 'Initial screen must contain one attempt'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    directory = ROOT/'jobs'/f'matrix-{args.label}-{stamp}'
    directory.mkdir()
    index = dict(label=args.label, started=stamp, model='gpt-5.6-luna', effort='high',
                 policy='One initial trial; exactly two follow-ups if initial reward is 1. No follow-ups after reward 0.',
                 planned_tasks=args.tasks, runs=[])
    if retry_trials:
        index['replaces_unstarted_attempts'] = dict(job=args.retry_unstarted_job, trials=retry_trials)
    lock = threading.Lock()

    def save():
        (directory/'index.json').write_text(json.dumps(index, indent=2)+'\n')

    save()
    print('INDEX', directory/'index.json', flush=True)

    def run(task, stage, trials):
        now = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        condition = stage if stage in ('oracle', 'shortcut') else 'plain'
        phase = stage+'-plain' if condition == 'plain' else condition
        name = f'{task}-{args.label}-{phase}-{now}'
        command = [sys.executable, 'scripts/run_science.py', f'tasks/{task}',
                   '--job-name', name,
                   '--trials', str(trials), '--concurrency', str(1 if retry_trials else trials)]
        if condition in ('oracle', 'shortcut'):
            command += ['--agent', 'oracle']
            if condition == 'shortcut':
                filename = {'reaction-diffusion':'reaction_baseline.py',
                            'thermal-bodies':'thermal_colored_baseline.py'}.get(task, task.replace('-', '_')+'_baseline.py')
                command += ['--solution-model', 'scripts/'+filename]
        else:
            command += ['--model', 'gpt-5.6-luna', '--reasoning-effort', 'high']
        print('START', name, flush=True)
        with (directory/f'{task}-{stage}.log').open('w') as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = dict(task=task, condition=condition, stage=stage, job=name,
                   expected_trials=trials, returncode=process.returncode)
        if (ROOT/'jobs'/name/'result.json').exists():
            row.update(summary(ROOT/'jobs'/name))
        with lock:
            index['runs'].append(row)
            save()
        print(json.dumps(row), flush=True)
        if process.returncode or row.get('errors', 1) or row.get('trials') != trials:
            raise RuntimeError(f'Incomplete/errored batch: {name}')
        if condition in ('oracle', 'shortcut'):
            assert row['passes'] == (1 if condition == 'oracle' else 0), row
            metrics = list((ROOT/'jobs'/name).glob('*/verifier/metrics.json'))
            assert len(metrics) == 1
            m = json.loads(metrics[0].read_text())
            parameter_error = next(m[k] for k in ['parameter_relative_error', 'parameter_relative_error_max',
                                                  'relative_parameter_error', 'relative_diffusivity_error',
                                                  'relative_conductance_error'] if k in m)
            chi2 = m.get('calibration_chi2', m.get('calibration_reduced_chi2'))
            assert chi2 < 1.5 and parameter_error < .03, m
        return row

    def screen(task):
        try:
            if retry_trials:
                first = run(task, 'initial-retry' if retry_initial else 'followup-retry', len(retry_trials))
                if retry_initial and first['passes'] == 1:
                    run(task, 'followup', 2)
                return True
            if args.controls is not None:
                for condition in args.controls or ('oracle', 'shortcut'):
                    run(task, condition, 1)
            first = run(task, 'initial', 1)
            if first['passes'] == 1:
                run(task, 'followup', 2)
            return True
        except Exception as error:
            print(str(error), flush=True)
            return False

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(screen, args.tasks))
    raise SystemExit(0 if all(outcomes) else 1)


if __name__ == '__main__':
    main()
