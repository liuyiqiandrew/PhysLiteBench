"""Review terminal provider failures and keep unknown charges separate from estimates."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil

R=Path('/Users/yp9846/Desktop/PhysLiteBench');Q=R/'results/hint-comparison-20261014';S=R/'staging/hint-comparison-20261014';P=Path('/private/tmp/physlite-hints-prime-session')
now=datetime.now(timezone.utc).isoformat();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)
def hashes(p):
    return {str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file() and not any(x in f.parts for x in ['__pycache__','.pytest_cache','.git'])}

plan=json.loads((Q/'plan.json').read_text());oldplan=json.loads(json.dumps(plan));oldhash=sha(Q/'plan.json')
for name,h in plan['tooling_sha256'].items():assert sha(S/name)==h,name
for task,t in plan['tasks'].items():
    for kind,x in t['runtime'].items():assert hashes(R/x['path'])==x['sha256'],(task,kind)
folder=Q/'blocks/026';block=json.loads((folder/'block.json').read_text());assert block['status']=='needs_review'
assert not any(json.loads(x.read_text())['status']=='running' for x in (Q/'blocks').glob('*/block.json'))
extended_path=P/'compatibility-20261015-0517-extended.json'
extended=json.loads(extended_path.read_text()) if extended_path.exists() else None
resume_allowed=bool(extended and extended['success'])
records=[]
for c in plan['conditions']:
    run=json.loads((folder/(c+'.json')).read_text());assert len(run['outcomes'])==1
    o=run['outcomes'][0];assert o['exception_type']=='NonZeroAgentExitCodeError' and o['model_effort']==[[block['model'],'high']]
    assert sha(R/o['result_path'])==o['result_sha256']
    events=[]
    for line in ((R/o['result_path']).parent/'agent/codex.txt').read_text().splitlines():
        try:e=json.loads(line)
        except ValueError:continue
        if e.get('type') in ['error','turn.failed']:events.append(e)
    assert any('Upstream stream failed' in json.dumps(e) for e in events)
    reqs=json.loads((folder/(c+'-prime-ledger.json')).read_text())['requests']
    assert len(reqs)==3 and sum(x['status']=='missing_usage' for x in reqs)==1
    assert all(x['provider_reasoning_effort']=='high' and x['model']==block['model'] for x in reqs)
    records.append({'condition':c,'trial':o['trial'],'result_sha256':o['result_sha256'],'exception_type':o['exception_type'],'terminal_events':events[-2:],'cost_range_usd':run['prime_cost_range_usd'],'exact_usage_available':False})

history=Q/'history/block-026-before-review-20261015';assert not history.exists();history.mkdir(parents=True)
shutil.copytree(folder,history/'block-records')
for f in ['plan.json','results.json','status.json','summary.csv','REPORT.txt']:shutil.copy2(Q/f,history/f)
shutil.copy2(S/'tooling/collect.py',history/'collect.py')

collector=(S/'tooling/collect.py').read_text()
old="""                    row['cost_range_usd'] = run['prime_cost_range_usd']
                    row['estimated_standard_api_usd'] = run['prime_cost_range_usd'][1]
                    row['cost_reconciled'] = run['prime_requests_settled']"""
new="""                    low, high = run['prime_cost_range_usd']
                    row['cost_range_usd'] = [low, high]
                    # A closed connection does not establish billed usage. An
                    # unconsumed spending reserve is an upper bound, not a cost.
                    complete_usage = bool(run['prime_requests_settled']) and abs(high - low) < 1e-12
                    row['known_recorded_api_usd'] = low
                    row['estimated_standard_api_usd'] = high if complete_usage else None
                    row['cost_reconciled'] = complete_usage
                    row['usage_complete'] = complete_usage"""
assert collector.count(old)==1;collector=collector.replace(old,new)
compile(collector,'collect.py','exec')
(S/'tooling/collect.py').write_text(collector);shutil.copy2(S/'tooling/collect.py',Q/'tooling/collect.py')
plan['tooling_sha256']['tooling/collect.py']=sha(S/'tooling/collect.py')
plan['accounting_amendment']='reviews/block-026-review-20261015.json'
for key in ['tasks','blocks','models','reasoning_effort','conditions','repeats','agent_seconds','verifier_seconds','max_concurrent_trials','reuse_manifest_sha256','administrative_observations_sha256']:assert plan[key]==oldplan[key],key
write(Q/'plan.json',plan)

for c in plan['conditions']:
    run=json.loads((folder/(c+'.json')).read_text())
    run.update(prime_usage_complete=False,accounting_note='prime_requests_settled means no active transport request; one request lacks usage. Preserve the lower known cost and upper spending reserve separately.',operational_review='results/hint-comparison-20261014/reviews/block-026-review-20261015.json')
    if resume_allowed:run['status']='completed_with_errors'
    write(folder/(c+'.json'),run)
if resume_allowed:
    block.update(status='completed',reviewed_utc=now,operational_review='results/hint-comparison-20261014/reviews/block-026-review-20261015.json')
    write(folder/'block.json',block)
probes=[]
for name in ['compatibility-20261015-0517.json','compatibility-20261015-0517-extended.json']:
    if (P/name).exists():
        shutil.copy2(P/name,Q/'reviews'/name);probes.append('reviews/'+name)
review={'reviewed_utc':now,'block':26,'cause_of_hold':'All three agent streams ended with the provider message Upstream stream failed. The local relay remained alive; there were no active containers or pending requests at review.',
        'provider_root_cause':'Not established by retained responses. No attribution to task physics or task implementation.',
        'original_outcomes_preserved':True,'retries':0,'replacements':0,'task_prompt_grading_limits_and_order_unchanged':True,
        'records':records,'prior_plan_sha256':oldhash,'new_plan_sha256':sha(Q/'plan.json'),
        'accounting_fix':'The collector now reports no point cost estimate when a Prime cost range has unequal bounds. Connection settlement alone no longer marks usage reconciled. Known charges and reserves remain separate.',
        'compatibility_probes':probes,'probes_are_benchmark_observations':False,'resume_allowed':resume_allowed,
        'resumption_evidence':'Independent High-effort streaming compatibility check completed successfully.' if resume_allowed else 'No completed compatibility check yet. Keep the scheduler stopped.',
        'next_unstarted_block':27,'history':str(history.relative_to(R))}
write(Q/'reviews/block-026-review-20261015.json',review)
print(json.dumps({'resume_allowed':resume_allowed,'retained_execution_failures':3,'accounting_corrected':True,'next_unstarted_block':27,'review':str(Q/'reviews/block-026-review-20261015.json')}))
