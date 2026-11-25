"""Report the latest source-matched batches; never choose a favorable rerun."""
import json
import hashlib
from pathlib import Path
from summarize_candidates import digest_tree

ROOT = Path(__file__).resolve().parents[1]
PHYSICS_FAILURES = {'physics_model_failure', 'physical_model_failure', 'memory_state_reset_failure'}


def main():
    summary = json.loads((ROOT/'results/candidates.json').read_text())
    batches = summary['batches']
    original = json.loads((ROOT/'results/hardening-original-audit.json').read_text())
    original_pairs = json.loads((ROOT/'results/original-hint-paired-audit.json').read_text())['pairs']
    rows = {}
    for name, audit in original.items():
        job = ROOT/audit['job']
        current = ROOT/'tasks'/name
        matching = all(digest_tree(current/p) == digest_tree(job/'frozen-task'/p) for p in ['environment', 'tests'])
        matching = matching and all((current/p).read_bytes()==(job/'frozen-task'/p).read_bytes()
                                    for p in ['instruction.md','task.toml'])
        passes = sum(t['reward'] == 1 for t in audit['trials'])
        pair = original_pairs[name]
        complete = pair['checks']['all_trials_complete_without_exception'] and pair['checks']['three_trials_each']
        native = pair['checks']['native_model_effort'] and pair['checks']['native_cli']
        rows[name] = dict(task=name, plain_job=audit['job'], plain_passes=passes, plain_trials=3,
                          complete=complete, native_setup_verified=native,
                          paired_comparison_verified=pair['all_checks_pass'],
                          current_source_matches=matching, reviewed_physics_zero=matching and complete and native and passes==0,
                          evidence='results/hardening-original-audit.json; results/original-hint-paired-audit.json; archives/pre-curation-docs/DASHBOARD.md')
    names = sorted({b['task'] for b in batches if b['condition']=='plain'})
    for name in names:
        plain = max((b for b in batches if b['task']==name and b['condition']=='plain'),key=lambda b:b['job'][-15:])
        task = ROOT/'tasks'/name
        if not task.exists():
            archive = json.loads((ROOT/'archives/screened/manifest.json').read_text())['tasks'].get(name, {})
            task = ROOT/archive.get('archived_path', 'archives/screened/tasks/'+name)
        matching = task.exists() and all(digest_tree(task/p)==plain[p+'_sha256'] for p in ['environment','tests'])
        matching = matching and all(hashlib.sha256((task/p).read_bytes()).hexdigest()==plain[key]
                                    for p,key in [('instruction.md','instruction_sha256'),('task.toml','task_config_sha256')])
        trials=plain['trials']
        complete=len(trials)==3 and all(t['reward'] in [0,1] and t['exception'] is None for t in trials)
        native=all(t['native_model_effort']==[['gpt-5.6-luna','high']] and t['native_cli_versions']==['0.154.0'] for t in trials)
        passes=sum(t['reward']==1 for t in trials)
        classifications=[t['review']['classification'] for t in trials]
        pairs = [p for p in summary['paired_checks'] if p['plain_job']==plain['job']]
        paired = len(pairs)==1 and all(v for v in pairs[0].values() if isinstance(v,bool))
        rows[name]=dict(task=name,plain_job=plain['job'],plain_passes=passes,plain_trials=len(trials),
                        complete=complete,native_setup_verified=native,current_source_matches=matching,
                        paired_comparison_verified=paired,
                        classifications=classifications,
                        reviewed_physics_zero=bool(complete and native and matching and passes==0 and all(c in PHYSICS_FAILURES for c in classifications)))
    for name,row in rows.items():
        hint_jobs={p['hint_job'] for p in summary['paired_checks'] if p['plain_job']==row['plain_job']}
        if name in original_pairs and row['plain_job']==original_pairs[name]['plain']['job']:
            hint_jobs.add(Path(original_pairs[name]['hint']['job']).name)
        hints=[b for b in batches if b['task']==name and b['condition']=='hint' and b['job'] in hint_jobs]
        if hints:
            hint=max(hints,key=lambda b:b['job'][-15:])
            row.update(hint_job=hint['job'],hint_passes=sum(t['reward']==1 for t in hint['trials']),hint_trials=len(hint['trials']))
        row['active']=(ROOT/'tasks'/name).is_dir()
        row['threshold_candidate']=bool(row['current_source_matches'] and row['complete'] and row['native_setup_verified']
                                        and row['plain_trials']==3 and row['plain_passes']<=1)
    zero=[n for n,r in rows.items() if r['reviewed_physics_zero']]
    retained=[n for n,r in rows.items() if r['threshold_candidate']]
    active=sorted(p.name for p in (ROOT/'tasks').iterdir() if p.is_dir())
    unresolved=sorted(set(active)-set(retained))
    active_zero=sorted(set(active)&set(zero))
    curated=len(active_zero)>=10 and not unresolved and all(rows[n]['paired_comparison_verified'] for n in active)
    report=dict(policy='At least 10 reviewed physics tasks at 0/3; every other retained task at most 1/3. Latest completed batch only; current environment and grading must match.',
                reviewed_zero_tasks=sorted(zero),reviewed_zero_count=len(zero),threshold_candidates=sorted(retained),
                minimum_zero_threshold_met=len(active_zero)>=10,final_curation_complete=curated,
                active_tasks=active,active_reviewed_zero_tasks=active_zero,pending_or_ineligible_active_tasks=unresolved,tasks=rows)
    (ROOT/'results/hardening-status.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Reviewed current-source 0/3:',len(zero),', '.join(sorted(zero)))
    print('Other current-source <=1/3:',', '.join(n for n in retained if n not in zero))
    for n,r in rows.items():
        if r['plain_passes']==0 and not r['reviewed_physics_zero']:
            print('Zero awaiting source/review verification:',n,r.get('classifications'))


if __name__ == '__main__':
    main()
