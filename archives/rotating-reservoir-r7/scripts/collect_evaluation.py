"""Collect complete frozen jobs without treating rewards as physics reviews."""
import argparse
import hashlib
import json
from pathlib import Path
from freeze_evaluation import hashes

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('--revision',type=int,required=True)
    parser.add_argument('--job',required=True)
    args=parser.parse_args()
    stem=f'{args.task}-r{args.revision}'
    plan=json.loads((ROOT/'results'/f'{stem}-evaluation-plan.json').read_text())
    job=REPO/'jobs'/args.job
    assert (job/'run-timing.json').exists(), 'Job has not completed'
    assert hashes(job/'frozen-task')==plan['source_sha256'], 'Evaluated task differs from frozen plan'
    rows=[]
    for trial in sorted(job.glob(f'{args.task}__*')):
        result=json.loads((trial/'result.json').read_text())
        assert result['agent_info']['model_info']['name']==plan['model']
        native=[]
        for session in (trial/'agent/sessions').rglob('*.jsonl'):
            for line in session.read_text().splitlines():
                entry=json.loads(line)
                if entry.get('type')=='turn_context':
                    payload=entry['payload']
                    native.append(dict(model=payload.get('model'),effort=payload.get('effort',payload.get('reasoning_effort'))))
        assert native and all(v['model']==plan['model'] and v['effort']=='high' for v in native), native
        m=trial/'verifier/metrics.json'
        exception = result.get('exception_info')
        exception_type = (exception or {}).get('exception_type')
        # An exhausted solving budget is an outcome of the specified trial.
        # Service/setup interruptions cannot substitute for a science trial,
        # even if the verifier assigns the untouched starter a zero reward.
        scientific_outcome = exception_type in (None, 'AgentTimeoutError')
        rows.append(dict(trial=trial.name,reward=(result.get('verifier_result') or {}).get('rewards',{}).get('reward'),
                         exception=exception,scientific_outcome_observed=scientific_outcome,
                         task_checksum=result['task_checksum'],native_contexts=native,
                         metrics=json.loads(m.read_text()) if m.exists() else None,
                         source_sha256=digest(trial/'artifacts/app/model.py') if (trial/'artifacts/app/model.py').exists() else None,
                         result_sha256=digest(trial/'result.json'),agent_execution=result.get('agent_execution'),
                         classification='pending_source_and_trajectory_review'))
    assert len(rows)==3 and len({r['task_checksum'] for r in rows})==1
    report=dict(task=args.task,revision=args.revision,job=args.job,model=plan['model'],effort='high',hint=False,
                source_matches_frozen_plan=True,trials=rows,completed_trials=len(rows),
                completed_scientific_trials=sum(r['scientific_outcome_observed'] for r in rows),
                interrupted_or_unresolved_attempts=sum(not r['scientific_outcome_observed'] for r in rows),
                passes=sum(r['reward']==1 for r in rows),errors=sum(r['exception'] is not None for r in rows),
                qualification='pending_physics_review' if all(r['reward']==0 for r in rows) else 'does_not_meet_zero_of_three_target')
    target=ROOT/'results'/f'{stem}-results.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(target, report['passes'], '/',len(rows), 'passes')

if __name__=='__main__':main()
