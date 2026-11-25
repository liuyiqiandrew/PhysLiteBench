from pathlib import Path
from datetime import datetime, timezone
import ast, collections, csv, hashlib, io, json, shutil, subprocess

R=Path('/Users/yp9846/Desktop/PhysLiteBench')
E=R/'results/hint-comparison-20261014';S=R/'staging/hint-comparison-20261014'
IMPORT=E/'imported-friend-results-20261015'
HISTORY=E/'history/before-friend-reuse-20261015'
REF='origin/luna-high-qualified-tasks'
FOLDERS=['luna-all-efforts-20261011-205240','sol-terra-low-medium-20261011-185155']
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):return sha_bytes(p.read_bytes())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def replace_once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)

def main():
    assert not HISTORY.exists(),'Amendment already applied or partially started; inspect history before retry.'
    plan=json.loads((E/'plan.json').read_text());old_hash=sha(E/'plan.json')
    commit=subprocess.check_output(['git','rev-parse',REF],cwd=R,text=True).strip()
    files={};protocols={};remote_rows=[];checks=[];remote_summaries=[]
    for folder in FOLDERS:
        prefix='results/'+folder+'/'
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,prefix],cwd=R,text=True).splitlines()
        for name in names:
            relative=name[len('results/'):]
            files[relative]=subprocess.check_output(['git','show',commit+':'+name],cwd=R)
        p=json.loads(files[folder+'/protocol.json']);protocols[folder]=p
        assert p['physics_hint'] is False and p['general_hint'] is False and p['fresh_session_each_trial'] and not p['stop_after_pass']
        assert p['git_commit']==plan['source_commit'] and p['harbor_version']==plan['harbor']
        assert p['agent_timeout_sec']==plan['agent_seconds'] and p['verifier_timeout_sec']==plan['verifier_seconds']
        assert p['agent_setup_timeout_multiplier']==plan['setup_timeout_multiplier']
        assert set(p['tasks'])==set(plan['tasks'])
        for task in p['tasks']:
            for key,expected in [('source_sha256',plan['tasks'][task]['source_sha256']),('runtime_sha256',plan['tasks'][task]['runtime']['openai/baseline']['sha256'])]:
                assert p[key][task]==expected,(folder,task,key)
                checks.append(dict(batch=folder,task=task,kind=key,files=len(expected),matched=True))
        rows=list(csv.DictReader(io.StringIO(files[folder+'/trials.csv'].decode())))
        assert len(rows)==p['planned_trials']
        assert all(v==3 for v in collections.Counter((r['model'],r['reasoning_effort'],r['task']) for r in rows).values())
        verification=json.loads(files[folder+'/verification.json'])
        assert verification['complete'] and verification['exact_unhinted_prompts']==len(rows)
        for row in rows:
            assert row['hint']=='none'
            assert (row['passed']=='True')==(row['verifier_reward']=='1.0' and not row['exception_type'])
            row['imported_batch']=folder;remote_rows.append(row)
        for summary in csv.DictReader(io.StringIO(files[folder+'/summary.csv'].decode())):
            summary['imported_batch']=folder;remote_summaries.append(summary)
    assert len(remote_rows)==390 and len({r['trial'] for r in remote_rows})==390
    lookup={(r['model'],r['reasoning_effort'],r['task'],int(r['repeat'])):r for r in remote_rows}
    assert len(lookup)==390
    reused=[];cancelled=[];already=[]
    for block in plan['blocks']:
        key=(block['model'],plan['reasoning_effort'],block['task'],block['repeat'])
        if key not in lookup or 'baseline' not in block['condition_order']:continue
        friend=lookup[key];record=E/'blocks'/f"{block['index']:03d}"/'baseline.json'
        if record.exists():
            local=json.loads(record.read_text());assert local['outcomes'] and local['status']!='running'
            already.append(dict(block=block['index'],local_trials=[r['trial'] for r in local['outcomes']],remote_trial=friend['trial'],reason='Preserve the already-started local slot; do not replace or pool its outcome. Remote result remains in the imported full cohort.'))
            continue
        block['condition_order'].remove('baseline')
        block['reused_conditions']={'baseline':friend['trial']}
        folder=friend['imported_batch'];protocol=protocols[folder]
        exc={'exception_type':friend['exception_type'],'exception_message':'Imported report; full native evidence remains on the source machine.'} if friend['exception_type'] else None
        def number(key):return float(friend[key]) if friend.get(key) else None
        row=dict(model=friend['model'],task=friend['task'],condition='baseline',repeat=int(friend['repeat']),block=block['index'],reasoning_effort=friend['reasoning_effort'],trial=friend['trial'],reward=float(friend['verifier_reward']),exception=exc,execution_error=friend['exception_type'] or None,completed_pass=friend['passed']=='True',reused=True,development_selected=False,codex_cli=protocol['codex_version'],concurrency=protocol['concurrency'],source_cohort=folder,source_remote_commit=commit,source_report='https://github.com/liuyiqiandrew/PhysLiteBench/blob/'+commit+'/results/'+folder+'/report.md',source_csv=str((IMPORT/folder/'trials.csv').relative_to(R)),source_protocol=str((IMPORT/folder/'protocol.json').relative_to(R)),verification_basis='Published protocol hashes and verification.json; raw native logs and submitted source are not available locally.',source_reported_model_effort=[[friend['model'],friend['reasoning_effort']]],observed_model_effort=None,estimated_standard_api_usd=number('estimated_api_cost_usd'),known_recorded_api_usd=number('known_request_cost_usd'),usage_complete=friend['usage_complete']=='True',token_usage={k:int(friend[k]) for k in ['input_tokens','cached_input_tokens','cache_write_input_tokens','output_tokens','reasoning_output_tokens']},request_count=int(friend['request_count']),agent_execution_seconds=None,reported_trial_wall_seconds=number('wall_seconds'),source_reported_evidence=friend['evidence'])
        reused.append(row);cancelled.append(dict(block=block['index'],model=block['model'],task=block['task'],repeat=block['repeat'],condition='baseline',replacement_trial=friend['trial']))
    assert len(reused)==38 and len(already)==1
    assert sum(len(b['condition_order']) for b in plan['blocks'])==313
    assert all(b['model']=='gpt-5.6-luna' for b in cancelled)
    selection='For each pending slot, match exact model, effort, task and repeat index to the friend report. Preserve all already-started local slots. Selection does not depend on reward, cost or error status. Import every source result; one overlapping source trial remains supplemental because its local slot already finished.'
    caveat='Luna baseline combines 38 historical Codex CLI 0.154.0/concurrency-4 trials with one existing local CLI 0.160.0/concurrency-3 trial. Hinted trials use CLI 0.160.0/concurrency 3. Task, instruction and grading hashes match, but client version, run date and concurrency differ: this is a descriptive comparison with historical controls, not a fully controlled estimate of hint causality. Other model conditions retain their frozen protocol.'
    reuse_doc=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_remote_commit=commit,selection_rule=selection,comparability_caveat=caveat,trials=reused,already_started_local_slots=already,all_imported_trial_count=390,source_file_hashes={str((IMPORT/rel).relative_to(R)):sha_bytes(data) for rel,data in files.items()})
    manifest=dict(remote_url='https://github.com/liuyiqiandrew/PhysLiteBench.git',remote_commit=commit,imported_utc=datetime.now(timezone.utc).isoformat(),files={rel:sha_bytes(data) for rel,data in files.items()},summary=remote_summaries,trial_count=390,matching_hash_checks=checks,total_file_hash_comparisons=sum(c['files'] for c in checks),scope='Exact snapshots of the two user-linked report directories. Existing benchmark cohorts remain preserved. Reuse by exact model/effort/condition/task hashes; GPT-5.6 and GPT-6 models are distinct.',not_in_current_high_hint_schedule=['gpt-5.6-luna none/low/medium/xhigh/max baseline','gpt-5.6-sol low/medium baseline','gpt-5.6-terra low/medium baseline'])
    # Prepare tooling changes in memory, then validate before writing.
    cp=S/'tooling/collect.py';collector=cp.read_text()
    collector=replace_once(collector,'Collect new hint-comparison observations without reusing historical scores.','Collect local observations and explicitly authorized historical baseline slots.')
    collector=replace_once(collector,"reasoning_effort='high', reused=False, observed_model_effort=outcome['model_effort'],","reasoning_effort='high', reused=False, codex_cli=plan['codex_cli'], source_cohort=NAME, observed_model_effort=outcome['model_effort'],")
    collector=replace_once(collector,"    assert len({row['trial'] for row in rows}) == len(rows), 'Duplicate trial'",'''    reuse = json.loads((ROOT / plan['reuse_manifest']).read_text()) if plan.get('reuse_manifest') else {'trials': [], 'source_file_hashes': {}}
    if plan.get('reuse_manifest'):
        assert sha(ROOT / plan['reuse_manifest']) == plan['reuse_manifest_sha256'], 'Reused baseline manifest drift'
    for path, expected in reuse['source_file_hashes'].items():
        assert sha(ROOT / path) == expected, ('Imported evidence drift', path)
    for row in reuse['trials']:
        rows.append(dict(row))
        audit.append({'trial': row['trial'], 'reused': True, 'verification_basis': row['verification_basis'], 'source_csv': row['source_csv'], 'source_remote_commit': row['source_remote_commit']})
    assert len({row['trial'] for row in rows}) == len(rows), 'Duplicate trial' ''')
    collector=replace_once(collector,"'passes': sum(r['completed_pass'] for r in selected),","'reused_attempts': sum(r['reused'] for r in selected),\n                            'passes': sum(r['completed_pass'] for r in selected),")
    collector=replace_once(collector,"'recorded_attempts': len(rows), 'expected_attempts': plan['expected_attempts'],","'recorded_attempts': len(rows), 'expected_attempts': plan['expected_attempts'],\n              'fresh_attempts': sum(not r['reused'] for r in rows), 'reused_attempts': sum(r['reused'] for r in rows),\n              'planned_fresh_attempts': plan.get('planned_fresh_attempts', plan['expected_attempts']),\n              'remaining_fresh_attempts': plan.get('planned_fresh_attempts', plan['expected_attempts']) - sum(not r['reused'] for r in rows),")
    collector=replace_once(collector,"    for filename, data in [('results.json',", "    if status['review_blocks']:\n        status['state'] = 'needs_review'\n    for filename, data in [('results.json',")
    collector=replace_once(collector,"'passes', 'checkpoint_passes', 'errors', 'complete', 'mean_estimated_api_usd']","'passes', 'checkpoint_passes', 'errors', 'complete', 'mean_estimated_api_usd', 'reused_attempts']")
    collector=replace_once(collector,'High effort, three fresh repeats per condition and task.','High effort, three observations per condition and task; explicitly labeled historical baseline reuse.')
    collector=replace_once(collector,"'Only the instruction differs within each model/task comparison. All execution errors are retained separately.',","'Frozen task and grading hashes match. Luna historical-baseline comparisons also differ in client version, concurrency and date. All execution errors are retained separately.',")
    collector=replace_once(collector,"'Historical task-selection results are not reused. This experiment measures these hint packages; hint length is not matched.',","plan.get('reuse_caveat', 'Historical task-selection results are not reused.'),\n             'Hint length is not matched. Historical and fresh evidence are distinguished.',")
    collector=replace_once(collector,"    for row in summary:\n        lines.append", "    lines.append(f\"Fresh attempts: {status['fresh_attempts']}/{status['planned_fresh_attempts']}; reused baseline observations: {status['reused_attempts']}; remaining fresh attempts: {status['remaining_fresh_attempts']}.\")\n    for row in summary:\n        lines.append")
    pp=S/'tooling/plot.py';plot=pp.read_text()
    plot=replace_once(plot,"fig, ax = plt.subplots(figsize=(12, 6.5))","fig, ax = plt.subplots(figsize=(12, 7.2))")
    plot=replace_once(plot,"label = f\"{row['passes']}/{row['attempts']}\" if row['attempts'] else 'unrun'","label = f\"{row['passes']}/{row['attempts']}\" if row['attempts'] else 'unrun'\n            if row.get('reused_attempts'):\n                label += '*' ")
    plot=replace_once(plot,"Physics benchmark: benefit of hints at High reasoning effort","Physics benchmark: hint comparison at High reasoning effort")
    plot=replace_once(plot,"All baselines are fresh. Failed grades are not automatically classified as physics errors. Task selection used GPT-5.6 Luna.","* Luna baseline: 38 historical observations + 1 fresh; historical client 0.154.0 versus 0.160.0 for hints.\\nDescriptive historical-control comparison; client version, concurrency and date differ. Failed grades are not automatically physics errors.")
    rp=S/'tooling/run_block.py';runner=rp.read_text()
    runner=replace_once(runner,'Run one predeclared three-condition block, retaining every outcome.','Run one predeclared condition block, retaining every outcome and skipping imported baseline slots.')
    runner=replace_once(runner,"    authorization = json.loads((RESULTS / 'budget-authorization.json').read_text())","    assert not set(block.get('reused_conditions', {})) & set(block['condition_order']), 'Duplicate imported baseline launch'\n    authorization = json.loads((RESULTS / 'budget-authorization.json').read_text())")
    for code in [collector,plot,runner]:ast.parse(code)
    HISTORY.mkdir(parents=True)
    for name in ['plan.json','status.json','results.json','summary.csv','REPORT.txt','budget-authorization.json','monitor.json']:
        if (E/name).exists():shutil.copy2(E/name,HISTORY/name)
    shutil.copytree(S/'tooling',HISTORY/'tooling')
    if (R/'output/hint-comparison-20261014').exists():shutil.copytree(R/'output/hint-comparison-20261014',HISTORY/'output')
    for rel,data in files.items():
        path=IMPORT/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    write(IMPORT/'import-manifest.json',manifest)
    write(E/'reused-baselines-20261015.json',reuse_doc)
    cp.write_text(collector);pp.write_text(plot);rp.write_text(runner)
    plan.update(fresh_baselines=False,reuse_historical_trials=True,reused_attempts=38,planned_fresh_attempts=313,openai_attempts=196,prime_attempts=117,reuse_manifest=str((E/'reused-baselines-20261015.json').relative_to(R)),reuse_manifest_sha256=sha(E/'reused-baselines-20261015.json'),reuse_caveat=caveat,previous_plan_sha256=old_hash,scope_amendment='friend-reuse-amendment-20261015.json')
    plan['protocol'].append('User authorized reuse of friend results and cancellation of overlapping pending runs. Use deterministic model/effort/task/repeat matching, preserve all started local outcomes, and label client/concurrency/date differences. No outcome-based selection or resampling.')
    plan['tooling_sha256']={rel:sha(S/rel) for rel in plan['tooling_sha256']}
    write(E/'plan.json',plan)
    amendment=dict(created_utc=datetime.now(timezone.utc).isoformat(),user_request='Use the two friend reports and cancel all overlapping pending attempts.',old_plan_sha256=old_hash,new_plan_sha256=sha(E/'plan.json'),old_plan_path=str((HISTORY/'plan.json').relative_to(R)),import_manifest=str((IMPORT/'import-manifest.json').relative_to(R)),reuse_manifest=plan['reuse_manifest'],cancelled_pending_attempts=38,cancelled_slots=cancelled,already_started_preserved=already,selection_rule=selection,comparability_caveat=caveat,expected_observations=351,planned_fresh_attempts=313,reused_observations=38,at_amendment_fresh_attempts_recorded=9,at_amendment_remaining_fresh_attempts=304,no_active_trials_cancelled=True,task_prompt_grading_hashes_unchanged=True,original_remote_cohorts_fully_preserved=True,raw_remote_native_logs_available_locally=False,original_sweep_results_unchanged=True)
    write(E/'friend-reuse-amendment-20261015.json',amendment)
    shutil.copy2(__file__,E/'tooling/amend_friend_reuse.py')
    print(json.dumps({k:amendment[k] for k in ['cancelled_pending_attempts','expected_observations','planned_fresh_attempts','reused_observations','at_amendment_remaining_fresh_attempts','comparability_caveat']},indent=2))

if __name__=='__main__':main()
