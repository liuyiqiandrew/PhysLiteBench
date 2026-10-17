"""Close interrupted slots without retrying agents or inventing grading results."""
from pathlib import Path
from datetime import datetime, timezone
import collections, hashlib, json, shutil

R=Path('/Users/yp9846/Desktop/PhysLiteBench')
Q=R/'results/hint-comparison-20261014'
S=R/'staging/hint-comparison-20261014'
P=Path('/private/tmp/physlite-hints-prime-session')
H=Q/'history/block-022-before-recovery-20261015'
NOW=datetime.now(timezone.utc).isoformat()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)
def hashes(p):
    return {str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file() and not any(x in f.parts for x in ['__pycache__','.pytest_cache','.git'])}

assert not H.exists(),'Recovery already applied or started; review before repeating'
plan=json.loads((Q/'plan.json').read_text());old_plan=json.loads(json.dumps(plan))
old_plan_sha=sha(Q/'plan.json')
assert old_plan_sha=='9f28d33d2eb0d3202971220c4c95272643158429590afa592e484c22ea430bf5'
for relative,expected in plan['tooling_sha256'].items():assert sha(S/relative)==expected,relative
for task,t in plan['tasks'].items():
    for kind,frozen in t['runtime'].items():assert hashes(R/frozen['path'])==frozen['sha256'],(task,kind)
for relative,expected in json.loads((R/plan['reuse_manifest']).read_text())['source_file_hashes'].items():assert sha(R/relative)==expected
folder=Q/'blocks/022';block=json.loads((folder/'block.json').read_text())
assert block['status']=='running'
assert not any(json.loads(f.read_text())['status']=='running' for f in (Q/'blocks').glob('*/block.json') if f.parent!=folder)
ledger=json.loads((P/'ledger.json').read_text())
pending=[x for x in ledger['requests'] if x['status']=='pending']
assert len(pending)==2 and {x['trial_tag'] for x in pending}=={'block-022-general','block-022-physics'}
paid=ledger['requests'][block['prime_request_start']:]
assert all(x['model']=='deepseek/deepseek-v4.1-flash' and x['provider_reasoning_effort']=='high' for x in paid)
assert {x['trial_tag'] for x in paid}=={'block-022-baseline','block-022-general','block-022-physics'}
for condition in ['general','physics']:
    evidence=json.loads((Q/'reviews/block-022-recovery-evidence'/f'{condition}.json').read_text())
    assert evidence['native_model_effort']==[['deepseek-v4.1-flash','high']]
    assert not evidence['original_trial_result_exists']
    assert evidence['terminal_events'][-1]['type']=='turn.failed'
    recovery=R/evidence['recovery_path']
    for relative,expected in evidence['files'].items():assert sha(recovery/relative)==expected
    assert hashes(recovery/'frozen-task')==plan['tasks']['magnetic-tracer']['runtime']['prime/'+condition]['sha256']
baseline=json.loads((folder/'baseline.json').read_text())
assert len(baseline['outcomes'])==1
assert baseline['outcomes'][0]['exception_type']=='NonZeroAgentExitCodeError'
assert sha(R/baseline['outcomes'][0]['result_path'])==baseline['outcomes'][0]['result_sha256']

H.mkdir(parents=True)
shutil.copytree(folder,H/'block-records')
for f in ['plan.json','results.json','status.json','summary.csv','REPORT.txt','evidence-audit.json']:shutil.copy2(Q/f,H/f)
shutil.copy2(S/'tooling/collect.py',H/'collect.py')
shutil.copy2(P/'ledger.json',P/'ledger-before-block-022-recovery-20261015.json')

# Interrupted requests are no longer in flight locally. Their provider outcome and
# exact charges remain unknown, and their original conservative reserves stay intact.
for row in pending:
    row.update(status='relay_interrupted_usage_unknown',observed_interrupted_utc=NOW,
               recovery_note='Relay process is absent. No completed usage response was retained. Provider outcome unknown; full spending reserve preserved.')
write(P/'ledger.json',ledger)

administrative=[];records=[]
for condition in ['baseline','general','physics']:
    run=json.loads((folder/(condition+'.json')).read_text())
    requests=[x for x in paid if x['trial_tag']==run['prime_trial_tag']]
    settled=all(x['status']=='completed' for x in requests)
    lo=sum(max(x.get('estimated_token_cost_usd') or 0,x.get('reported_cost_usd') or 0) for x in requests)
    hi=sum(x['accounted_usd'] for x in requests)
    write(folder/(condition+'-prime-ledger.json'),{'job':run['job'],'requests':requests,'accounting_note':'Unreported charges retain their upper reserves; no claim of provider settlement.'})
    run.update(status='completed_with_errors',prime_requests_settled=settled,prime_cost_range_usd=[lo,hi],
               operational_review='results/hint-comparison-20261014/reviews/block-022-recovery-20261015.json')
    if condition!='baseline':
        evidence=json.loads((Q/'reviews/block-022-recovery-evidence'/f'{condition}.json').read_text())
        trial=evidence['trial'];recovery=R/evidence['recovery_path']
        dst=Q/'evidence/022'/condition/trial;dst.mkdir(parents=True,exist_ok=True)
        shutil.copy2(recovery/'model.py',dst/'model.py')
        exception={'exception_type':'RunnerInterruptedError','exception_message':'Administrative recovery: the agent recorded a stream-disconnection failure, and the host runner exited before writing a trial result. No verifier result exists.'}
        row={'task':'magnetic-tracer','model':run['model'],'trial':trial,'condition':condition,'repeat':1,'block':22,
             'path':str(recovery.parent.relative_to(R)),'reasoning_effort':'high','reward':None,'exception':exception,
             'execution_error':'RunnerInterruptedError','completed_pass':False,'reused':False,'codex_cli':plan['codex_cli'],
             'source_cohort':plan['experiment'],'observed_model_effort':[[run['model'],'high']],
             'development_selected':False,'administrative_observation':True,'verifier_ran':False,
             'run_record':str((folder/(condition+'.json')).relative_to(R)),
             'evidence':str(dst.relative_to(R)),'recovery_evidence':str((Q/'reviews/block-022-recovery-evidence'/f'{condition}.json').relative_to(R)),
             'source_sha256':sha(recovery/'model.py'),'native_sources':[{'path':str(p.relative_to(R)),'sha256':sha(p)} for p in (recovery/'sessions').rglob('*.jsonl')],
             'token_usage':None,'request_count':len(requests),'estimated_standard_api_usd':None,'known_recorded_api_usd':lo,
             'cost_range_usd':[lo,hi],'cost_reconciled':False,'usage_complete':False,'agent_execution_seconds':None,
             'interpretation':'Started fixed-schedule slot retained as an infrastructure failure. Original job files remain unchanged. No retry, invented grade, or physics-failure attribution.'}
        administrative.append(row)
        run.update(errors=1,passes=0,administrative_observation=trial,observed_interrupted_utc=NOW)
    write(folder/(condition+'.json'),run)
    records.append({'condition':condition,'requests':len(requests),'cost_range_usd':[lo,hi],'usage_complete':settled})

adminpath=Q/'administrative-observations-20261015.json'
write(adminpath,{'created_utc':NOW,'trials':administrative,'review':'reviews/block-022-recovery-20261015.json'})
collector=(S/'tooling/collect.py').read_text()
anchor="    reuse = json.loads((ROOT / plan['reuse_manifest']).read_text()) if plan.get('reuse_manifest') else {'trials': [], 'source_file_hashes': {}}"
assert collector.count(anchor)==1
addition="""    # Retain started slots whose native runner record was lost. These are explicit
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
"""
collector=collector.replace(anchor,addition+anchor)
(S/'tooling/collect.py').write_text(collector)
shutil.copy2(S/'tooling/collect.py',Q/'tooling/collect.py')
plan['tooling_sha256']['tooling/collect.py']=sha(S/'tooling/collect.py')
plan['administrative_observations']=str(adminpath.relative_to(R))
plan['administrative_observations_sha256']=sha(adminpath)
plan['operational_amendment']='reviews/block-022-recovery-20261015.json'
for key in ['tasks','blocks','models','conditions','repeats','reasoning_effort','agent_seconds','verifier_seconds','max_concurrent_trials','reuse_manifest','reuse_manifest_sha256']:
    assert plan[key]==old_plan[key],key
write(Q/'plan.json',plan)
block.update(status='completed',closed_by_operational_review=True,reviewed_utc=NOW,
             passes={x:0 for x in plan['conditions']},errors={x:1 for x in plan['conditions']},
             recovery_review='results/hint-comparison-20261014/reviews/block-022-recovery-20261015.json')
write(folder/'block.json',block)
review={'reviewed_utc':NOW,'block':22,'finding':'Host scheduler and Prime relay are absent. All three Codex streams recorded terminal connection failures. Two native runner results were never written; their idle Docker containers retained recoverable evidence.',
        'root_trigger':'The retained logs do not establish why the host processes terminated. No claim of a model physics failure, laptop sleep, or user action.',
        'safe_recovery_basis':'No model processes remain in the containers. Terminal agent logs, exact frozen input hashes and model/effort records are retained. Dead relay connections cannot resume. Continue only unstarted slots with durable host logs.',
        'original_native_results_preserved':True,'original_two_missing_results_not_fabricated':True,
        'administrative_execution_failures':2,'native_execution_failures':1,'retry_count':0,'replacement_count':0,
        'prior_plan_sha256':old_plan_sha,'new_plan_sha256':sha(Q/'plan.json'),'task_prompt_grading_limits_and_fixed_schedule_unchanged':True,
        'collector_change':'Append two explicitly labeled administrative failures; original job results and all previous rows remain intact.',
        'accounting':records,'pending_request_recovery':'Two rows marked relay_interrupted_usage_unknown. Exact charges unresolved; conservative reserves preserved. They are not claimed to have settled.',
        'next_unstarted_block':25,'history':str(H.relative_to(R))}
write(Q/'reviews/block-022-recovery-20261015.json',review)
print(json.dumps({'review':str(Q/'reviews/block-022-recovery-20261015.json'),'administrative_failures':2,'native_failure':1,'next_block':25,'cost_uncertainty_preserved':True}))
