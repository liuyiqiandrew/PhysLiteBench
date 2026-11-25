"""Report the instruction screen without conflating 0/1 with historical 0/3."""
import hashlib
import json
from pathlib import Path

from summarize_candidates import digest_tree

ROOT = Path(__file__).resolve().parents[1]
PHYSICS = {'physics_model_failure', 'physical_model_failure', 'memory_state_reset_failure'}


def main():
    ledger = json.loads((ROOT/'results/candidates.json').read_text())
    previous = json.loads((ROOT/'archives/pre-neutral-instructions/manifest.json').read_text())['tasks']
    rows = {}
    for task in sorted((ROOT/'tasks').iterdir()):
        if not task.is_dir():
            continue
        instruction = hashlib.sha256((task/'instruction.md').read_bytes()).hexdigest()
        config = hashlib.sha256((task/'task.toml').read_bytes()).hexdigest()
        batches = [b for b in ledger['batches'] if b['task'] == task.name and b['condition'] == 'plain'
                   and Path(b['index']).parent.name.startswith('matrix-neutral-')
                   and b['instruction_sha256'] == instruction and b['task_config_sha256'] == config
                   and b['environment_sha256'] == digest_tree(task/'environment')
                   and b['tests_sha256'] == digest_tree(task/'tests')]
        batches.sort(key=lambda b: b['job'][-15:])
        trials = [{**t, 'job':b['job'], 'stage':b.get('stage')} for b in batches for t in b['trials']]
        unstarted = [t for t in trials if t.get('agent_started') is False and t['reward'] is None
                     and t['exception'] is not None
                     and t.get('review', {}).get('classification') == 'infrastructure_setup_failure']
        model_trials = [t for t in trials if t not in unstarted]
        completed = [t for t in model_trials if t['exception'] is None and t['reward'] in (0, 1)]
        passes = sum(t['reward'] == 1 for t in completed)
        rewarded = [t for t in trials if t['reward'] in (0, 1)]
        raw_passes = sum(t['reward'] == 1 for t in rewarded)
        exceptions = sum(t['exception'] is not None for t in trials)
        native = bool(model_trials) and all(t['native_model_effort'] == [['gpt-5.6-luna', 'high']]
                                           and t['native_cli_versions'] == ['0.154.0'] for t in model_trials)
        physical = all(t.get('review', {}).get('classification') in PHYSICS
                       for t in completed if t['reward'] == 0)
        reviewed = all(t.get('review', {}).get('classification') not in (None, 'pending') for t in trials)
        state = 'pending'
        if passes > 1:
            state = 'needs_hardening'
            if exceptions:
                state += '_with_exception'
            if not reviewed:
                state += '_awaiting_review'
        elif len(completed) != len(model_trials):
            state = 'errored_trial_requires_review'
        elif len(model_trials) == 1:
            if passes:
                state = 'awaiting_two_followups'
            elif not reviewed:
                state = 'failure_awaiting_review'
            elif not physical:
                state = 'nonphysical_failure_requires_review'
            else:
                state = 'screened_physical_failure_0_of_1'
        elif len(model_trials) == 3:
            state = 'needs_hardening' if passes > 1 else 'within_secondary_cutoff_1_of_3'
            if passes <= 1 and reviewed and not physical:
                state = 'nonphysical_failure_requires_review'
            if not reviewed:
                state += '_awaiting_review'
        elif model_trials:
            state = 'incomplete_or_nonstandard_trial_count'
        old = previous.get(task.name, {}).get('previous_result', {})
        rows[task.name] = dict(state=state, passes=passes, completed_trials=len(completed),
                               total_attempts=len(trials), model_trial_count=len(model_trials),
                               unstarted_infrastructure_attempts=[t['trial'] for t in unstarted],
                               raw_reward_passes=raw_passes, rewarded_trials=len(rewarded), exception_trials=exceptions,
                               native_setup_verified=native, all_terminal_trials_reviewed=reviewed,
                               failed_trials_have_physical_cause=bool(completed) and physical,
                               previous_plain_passes=old.get('plain_passes'), previous_plain_trials=old.get('plain_trials'),
                               instruction_sha256=instruction, batches=[b['job'] for b in batches], trials=trials)
    zeros = [n for n,r in rows.items() if r['state'] == 'screened_physical_failure_0_of_1' and r['native_setup_verified']]
    secondary = [n for n,r in rows.items() if r['state'] == 'within_secondary_cutoff_1_of_3' and r['native_setup_verified']]
    report = dict(protocol='One initial unhinted trial; two more only after an initial pass. Initial results are always retained. These conditional 0/1 and 1/3 outcomes do not establish three-trial failure rates for the current tasks.',
                  model='gpt-5.6-luna', effort='high', cli='0.154.0',
                  screened_physical_zero_tasks=zeros, secondary_cutoff_tasks=secondary,
                  all_active_tasks_screened=len(zeros)+len(secondary)==len(rows),
                  at_least_ten_tasks_failed_initial_screen=len(zeros)>=10,
                  tasks=rows)
    (ROOT/'results/neutrality-status.json').write_text(json.dumps(report, indent=2)+'\n')
    for name,row in rows.items():
        print(name, f"{row['passes']}/{row['completed_trials']}", row['state'])


if __name__ == '__main__':
    main()
