"""Run candidate controls and matched three-trial plain/hint Harbor batches."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
# Completed shortcuts not named '<task_with_underscores>_baseline.py'.
SHORTCUT_ALIASES = {'reaction-diffusion':'reaction_baseline.py'}
# A failing hidden prediction test; hierarchical-glass-overlaps names them test_coincidences.
PREDICTION_FAILURE = re.compile(r'FAILED \S*\btest_hidden\.py::(test_prediction|test_coincidences|test_fluctuation_prediction)\b')


def summary(job):
    result = json.loads((job/'result.json').read_text())
    rewards = []
    for evaluation in result['stats']['evals'].values():
        for score, trials in evaluation['reward_stats'].get('reward', {}).items():
            rewards.extend([float(score)]*len(trials))
    return {'trials':len(rewards), 'passes':sum(r == 1 for r in rewards),
            'errors':result['stats']['n_errored_trials']}


def shortcut_model(control_root, task):
    path = control_root/f'{task.replace("-", "_")}_baseline.py'
    if (ROOT/path).is_file() or task not in SHORTCUT_ALIASES:
        return path
    return control_root/SHORTCUT_ALIASES[task]


def check_control(job, condition):
    """Oracle passes every check; the shortcut fails only hidden prediction tests."""
    verifiers = list(job.glob('*/verifier'))
    assert len(verifiers) == 1, verifiers
    metrics = json.loads((verifiers[0]/'metrics.json').read_text())
    assert 'error' not in metrics, f'{job.name}: hidden verifier wrote no metrics: {metrics}'
    if condition == 'shortcut':
        lines = (verifiers[0]/'test-stdout.txt').read_text().splitlines()
        failed = [line for line in lines if line.startswith('FAILED ')]
        totals = [line for line in lines if re.search(r'\d+ (failed|passed)\b.* in [\d.]+s', line)]
        counts = {word:int(n) for n, word in re.findall(r'(\d+) (\w+)', totals[-1])} if totals else {}
        # Every test ran and none errored or was skipped, so the tests not listed as failed passed.
        complete = failed and counts.get('failed') == len(failed) and set(counts) <= {'failed', 'passed', 'warning', 'warnings'}
        assert complete and all(PREDICTION_FAILURE.match(line) for line in failed), \
            f'{job.name}: shortcut must fail only hidden prediction tests:\n'+'\n'.join(failed+totals[-1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tasks', nargs='+')
    parser.add_argument('--conditions', nargs='+', choices=['oracle', 'shortcut', 'plain', 'hint'],
                        default=['oracle', 'shortcut', 'plain', 'hint'])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--trial-concurrency', type=int, default=3,
                        help='Concurrent model trials per task; trial count remains three')
    parser.add_argument('--agent-setup-timeout-multiplier', type=float, default=None,
                        help='Multiply Codex setup timeout for model trials only; controls remain unchanged')
    parser.add_argument('--label', default='r1')
    parser.add_argument('--task-root', type=Path, default=Path('tasks'),
                        help='Parent of task directories, including staged candidates')
    parser.add_argument('--control-root', type=Path, default=Path('scripts'),
                        help='Directory containing completed shortcut models')
    args = parser.parse_args()
    if args.trial_concurrency < 1:
        parser.error('--trial-concurrency must be positive')
    if args.agent_setup_timeout_multiplier is not None and (
        not math.isfinite(args.agent_setup_timeout_multiplier) or args.agent_setup_timeout_multiplier <= 0
    ):
        parser.error('--agent-setup-timeout-multiplier must be positive and finite')
    problems = [f'duplicate task: {task}' for task in sorted(set(args.tasks)) if args.tasks.count(task) > 1]
    if not args.label or Path(args.label).name != args.label:
        problems.append(f'label must be a directory name: {args.label!r}')
    for task in dict.fromkeys(args.tasks):
        task_dir = ROOT/args.task_root/task
        if not task or Path(task).name != task:
            problems.append(f'task must be a directory name: {task}')
            continue
        if not task_dir.is_dir():
            problems.append(f'missing task directory: {task_dir}')
            continue
        if 'hint' in args.conditions and not (task_dir/'hint.md').is_file():
            problems.append(f'missing physics hint: {task_dir/"hint.md"}')
        shortcut = ROOT/shortcut_model(args.control_root, task)
        if 'shortcut' in args.conditions and not shortcut.is_file():
            problems.append(f'missing completed shortcut: {shortcut}')
    if problems:
        parser.error('nothing was launched:\n  '+'\n  '.join(problems))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    directory = ROOT/'jobs'/f'matrix-{args.label}-{stamp}'
    directory.mkdir(parents=True)
    lock = threading.Lock()
    index = {'label':args.label, 'started':stamp, 'model':'gpt-5.6-luna', 'effort':'high',
             'task_root':str(args.task_root), 'control_root':str(args.control_root), 'runs':[]}

    def record(row):
        with lock:
            index['runs'].append(row)
            (directory/'index.json').write_text(json.dumps(index, indent=2)+'\n')

    def task_runs(task):
        for condition in args.conditions:
            name = f'{task}-{args.label}-{condition}-{stamp}'
            command = [sys.executable, 'scripts/run_science.py', str(args.task_root/task), '--job-name', name]
            if condition in ['oracle', 'shortcut']:
                command += ['--agent', 'oracle', '--trials', '1', '--concurrency', '1']
                if condition == 'shortcut':
                    command += ['--solution-model', str(shortcut_model(args.control_root, task))]
            else:
                command += ['--model', 'gpt-5.6-luna', '--reasoning-effort', 'high', '--trials', '3',
                            '--concurrency', str(min(args.trial_concurrency, 3))]
                if condition == 'hint':
                    command.append('--hint')
                if args.agent_setup_timeout_multiplier is not None:
                    command += ['--agent-setup-timeout-multiplier', str(args.agent_setup_timeout_multiplier)]
            print(f'START {task} {condition}: {name}', flush=True)
            with (directory/f'{task}-{condition}.log').open('w') as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row = {'task':task, 'condition':condition, 'job':name, 'returncode':process.returncode}
            if (ROOT/'jobs'/name/'result.json').exists():
                row.update(summary(ROOT/'jobs'/name))
            record(row)
            print(json.dumps(row), flush=True)
            expected = 1 if condition in ['oracle', 'shortcut'] else 3
            if process.returncode or row.get('errors', 1) or row.get('trials') != expected:
                raise RuntimeError(f'Incomplete/errored Harbor batch: {name}')
            if condition in ['oracle', 'shortcut']:
                assert row['passes'] == (1 if condition == 'oracle' else 0), row
                check_control(ROOT/'jobs'/name, condition)
        return task

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name in pool.map(task_runs, args.tasks):
            print('COMPLETE', name, flush=True)
    print('INDEX', directory/'index.json', flush=True)


if __name__ == '__main__':
    main()
