from pathlib import Path
from datetime import datetime,timezone
import ast,collections,csv,difflib,hashlib,importlib.util,json,math,re,shutil
R=Path('/Users/yp9846/Desktop/PhysLiteBench');S=R/'staging/reasoning-sweep-20261011';E=R/'results/reasoning-sweep-20261011'
spec=importlib.util.spec_from_file_location('prior_collector',S/'tooling/prior_collector.py');prior_collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior_collector)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def native_settings(trial):
 settings=set()
 for f in (trial/'agent/sessions').rglob('*.jsonl'):
  for line in f.read_text().splitlines():
   try:d=json.loads(line)
   except ValueError:continue
   if d.get('type')=='turn_context':
    p=d.get('payload',{});settings.add((p.get('model'),p.get('effort')))
 return sorted(settings)
def main():
 plan=json.loads((E/'plan.json').read_text());prices=json.loads((E/'pricing.json').read_text());old=json.loads((R/plan['reuse_source']).read_text())
 prime_prices=json.loads((E/'prime/pricing-v2.json').read_text())
 prices['models']['deepseek/deepseek-v4.1-flash'].update({k:prime_prices[k] for k in ['input','cached_input','cache_write_input','output','apply_openai_long_context_rates']})
 rows=[]
 for r in old['trials']:
  if r['development_selected'] or r['model']=='deepseek/deepseek-v4.1-flash':continue
  x=dict(r);x.update(reasoning_effort='high',reused=True,prior_results=plan['reuse_source']);rows.append(x)
 audits=[];running=[];review=[]
 for runfile in sorted((E/'runs').glob('*/run.json')):
  run=json.loads(runfile.read_text())
  if run['model']=='deepseek/deepseek-v4.1-flash' and run.get('comparison_cohort')!='prime-transport-v2':continue
  task,model,effort=run['task'],run['model'],run['reasoning_effort'];job=R/'jobs'/run['job']
  if run.get('provider_reasoning_effort') is not None and run['status']!='running':
   ledger=json.loads((R/run['prime_batch_ledger']).read_text());assert ledger['requests'] and all(x.get('provider_reasoning_effort')==effort for x in ledger['requests'])
  if run['status']=='running':running.append({k:run[k] for k in ['task','model','reasoning_effort','job']})
  if run['status']=='needs_review':review.append({k:run[k] for k in ['task','model','reasoning_effort','job','errors']})
  frozen=job/'frozen-task'
  if frozen.exists():
   for rel,h in run['source_sha256'].items():assert sha(frozen/rel)==h,(str(runfile),rel)
  for rp in sorted(job.glob('*/result.json')):
   trial=rp.parent
   r=prior_collector.trial_record(task,model,trial,prices)
   r.update(comparison_cohort=run.get('comparison_cohort','openai-effort-sweep'),reasoning_effort=effort,reused=False,run_record=str(runfile.relative_to(R)))
   settings=native_settings(trial);r['observed_model_effort']=[list(x) for x in settings]
   canonical_settings=[('deepseek/deepseek-v4.1-flash' if m=='deepseek-v4.1-flash' and model=='deepseek/deepseek-v4.1-flash' else m,'max' if model=='deepseek/deepseek-v4.1-flash' and effort=='max' and e=='xhigh' and run.get('provider_reasoning_effort')=='max' else e) for m,e in settings]
   if settings:assert canonical_settings==[(model,effort)],(trial.name,settings,model,effort)
   r['canonical_model_effort']=[list(x) for x in canonical_settings]
   if r['model_names']:assert r['model_names']==[model],(trial.name,r['model_names'])
   if r['estimated_standard_api_usd'] is not None:assert abs(sum(e['estimated_standard_api_usd'] for e in r['usage_events'])-r['estimated_standard_api_usd'])<1e-9
   dst=E/'evidence'/model/effort/task/trial.name;dst.mkdir(parents=True,exist_ok=True)
   for src,name in [('artifacts/app/model.py','model.py'),('verifier/metrics.json','metrics.json'),('verifier/reward.txt','reward.txt'),('verifier/test-stdout.txt','test-stdout.txt')]:
    if (trial/src).exists():shutil.copy2(trial/src,dst/name)
   if (dst/'model.py').exists():
    starter=(S/'tasks'/task/'environment/model.py').read_text().splitlines(True)
    (dst/'model.diff').write_text(''.join(difflib.unified_diff(starter,(dst/'model.py').read_text().splitlines(True),fromfile='starter/model.py',tofile='submission/model.py')))
   source_flags=[]
   if (dst/'model.py').exists():
    code=(dst/'model.py').read_text()
    for lineno,line in enumerate(code.splitlines(),1):
     if re.search(r'/tests|test_hidden|/logs/verifier|reward\.txt|subprocess|socket|requests|eval\(|exec\(|__import__|setattr\(',line):source_flags.append({'line':lineno,'text':line[:240]})
   r['source_review_flags']=source_flags
   r['timing_outlier']=r.get('agent_execution_seconds',0)>660
   r['evidence']=str(dst.relative_to(R));audits.append({'trial':trial.name,'model':model,'effort':effort,'model_effort_verified':bool(settings),'file_hashes':{p.name:sha(p) for p in dst.iterdir() if p.is_file()},'network_flags':r['network_command_scan'],'source_flags':source_flags})
   rows.append(r)
 assert len({r['trial'] for r in rows})==len(rows)
 summary=[]
 desired=[(m,e) for m in plan['openai_models'] for e in plan['openai_reasoning_efforts']]+[('deepseek/deepseek-v4.1-flash','high'),('deepseek/deepseek-v4.1-flash','low'),('deepseek/deepseek-v4.1-flash','max'),('deepseek/deepseek-v4.1-flash','none')]
 for model,effort in desired:
  rs=[r for r in rows if r['model']==model and r['reasoning_effort']==effort];counts=collections.Counter(r['task'] for r in rs)
  assert all(n<=3 for n in counts.values()),(model,effort,counts)
  passes=sum(r['reward']==1 and r['exception'] is None for r in rs);n=len(rs);complete=all(counts[t]==3 for t in plan['tasks'])
  known=[r['estimated_standard_api_usd'] for r in rs if r['estimated_standard_api_usd'] is not None]
  p=passes/n if n else None
  if n:
   z=1.95996398454;center=(p+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);ci=[max(0,center-half),min(1,center+half)]
  else:ci=None
  summary.append(dict(model=model,reasoning_effort=effort,attempts=n,expected_attempts=39,passes=passes,pass_rate=p,complete_panel=complete,fully_observed_cost=len(known)==n and n>0,mean_estimated_api_usd=sum(known)/n if n and len(known)==n else None,total_estimated_api_usd=sum(known) if len(known)==n else None,missing_usage_trials=n-len(known),errors=sum(r['exception'] is not None for r in rs),timeouts=sum((r['exception'] or {}).get('exception_type')=='AgentTimeoutError' for r in rs),scored_checkpoint_passes=sum(r['reward']==1 for r in rs),reused_attempts=sum(r['reused'] for r in rs),descriptive_wilson_95=ci,per_task={t:{'attempts':counts[t],'passes':sum(r['task']==t and r['reward']==1 and r['exception'] is None for r in rs)} for t in plan['tasks']}))
 for point in summary:
  if point['model']!='deepseek/deepseek-v4.1-flash':continue
  effort=point['reasoning_effort'];runs=[]
  for f in (E/'runs').glob('*/run.json'):
   r=json.loads(f.read_text())
   if r['model']==point['model'] and r['reasoning_effort']==effort and r.get('comparison_cohort')=='prime-transport-v2':runs.append(r)
  ready=point['complete_panel'] and all(r.get('prime_requests_settled') and not r.get('prime_budget_blocked') and 'prime_cost_range_usd' in r for r in runs)
  if ready:
   base=[0,0]
   interval=[base[i]+sum(r['prime_cost_range_usd'][i] for r in runs) for i in range(2)]
   point.update(prime_cost_reconciled=True,prime_total_cost_range_usd=interval,mean_cost_range_usd=[x/39 for x in interval],mean_estimated_api_usd=interval[1]/39,total_estimated_api_usd=interval[1],fully_observed_cost=abs(interval[1]-interval[0])<1e-12,cost_is_conservative_upper_bound=interval[1]>interval[0])
 eligible=[s for s in summary if s['complete_panel'] and (s['fully_observed_cost'] or s.get('prime_cost_reconciled')) and (s['model']!='deepseek/deepseek-v4.1-flash' or s.get('prime_cost_reconciled'))]
 frontier=[s for s in eligible if not any(o['mean_estimated_api_usd']<=s['mean_estimated_api_usd'] and o['pass_rate']>=s['pass_rate'] and (o['mean_estimated_api_usd']<s['mean_estimated_api_usd'] or o['pass_rate']>s['pass_rate']) for o in eligible)]
 data=dict(updated_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(E/'plan.json'),task_count=13,summary=summary,trials=rows,running=running,needs_review=review,frontier=[{'model':r['model'],'reasoning_effort':r['reasoning_effort']} for r in sorted(frontier,key=lambda x:x['mean_estimated_api_usd'])],cost_note=plan['cost'],interpretation=plan['protocol'],prime_note='Prime cost accounting must include reconciled relay requests and retained reserves before admitting its points to the frontier.',historical_timing_caveat='Four reused Terra High trials were affected by host sleep; see the prior timing audit. No completed outcome was replaced.')
 data['prime_protocol']={'cohort':'prime-transport-v2','plan':'prime/transport-v2-plan.json','plan_sha256':sha(E/'prime/transport-v2-plan.json'),'legacy_results':'prime/legacy-transport-results.json','amendment':'prime/transport-v2-amendment.json','pricing':'prime/pricing-v2.json','pricing_sha256':sha(E/'prime/pricing-v2.json')}
 if (E/'unstarted-setup-recovery-20261012.json').exists():data['unstarted_setup_recovery']={'plan':'unstarted-setup-recovery-20261012.json','preserved_attempts':6,'excluded_from_primary_model_attempts':True}
 (E/'results.json').write_text(json.dumps(data,indent=2)+'\n');(E/'evidence-audit.json').write_text(json.dumps(audits,indent=2)+'\n')
 with (E/'summary.csv').open('w',newline='') as f:
  fields=[k for k in summary[0] if k not in ['per_task','descriptive_wilson_95']];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in summary)
 print(json.dumps({'new_terminal_attempts':sum(not x['reused'] for x in rows),'complete_points':[(x['model'],x['reasoning_effort'],x['passes']) for x in eligible],'running':running,'needs_review':review},indent=2))
if __name__=='__main__':main()
