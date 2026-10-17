from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,collections,re,subprocess
R=Path('/Users/yp9846/Desktop/PhysLiteBench');B=R/'results/hint-comparison-20261014';S=R/'staging/hint-comparison-20261014';W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench/.presentation-local-final-20261015')
p=json.loads((B/'plan.json').read_text());d=json.loads((B/'results.json').read_text());reuse=json.loads((B/'reused-baselines-20261015.json').read_text());audit=json.loads((B/'evidence-audit.json').read_text())
counts=collections.Counter();mismatch=[]
def check(path,expected,category):
 counts[category]+=1
 if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:mismatch.append(str(path.relative_to(R)))
for task,t in p['tasks'].items():
 for rel,h in t['source_sha256'].items():check(R/'tasks'/task/rel,h,'task_source_files')
 for runtime in t['runtime'].values():
  for rel,h in runtime['sha256'].items():check(R/runtime['path']/rel,h,'runtime_files')
for rel,h in p['tooling_sha256'].items():check(S/rel,h,'tooling_files')
check(R/p['reuse_manifest'],p['reuse_manifest_sha256'],'manifests')
check(R/p['administrative_observations'],p['administrative_observations_sha256'],'manifests')
for rel,h in reuse['source_file_hashes'].items():check(R/rel,h,'imported_files')
rows={r['trial']:r for r in d['trials']}
for a in audit:
 if a.get('file_hashes'):
  for rel,h in a['file_hashes'].items():check(R/rows[a['trial']]['evidence']/rel,h,'evidence_files')
expected={(b['model'],b['task'],c,b['repeat']) for b in p['blocks'] for c in list(b['condition_order'])+list(b.get('reused_conditions',{}))}
slots=[(r['model'],r['task'],r['condition'],r['repeat']) for r in d['trials']]
assert len(expected)==351 and len(slots)==len(set(slots)) and set(slots)<=expected
assert len(reuse['trials'])==38 and sum(r['reused'] for r in d['trials'])==38
assert all('baseline' not in b['condition_order'] for b in p['blocks'] if b.get('reused_conditions'))
issues=[]
for r in d['trials']:
 assert r['reasoning_effort']=='high' and r['model'] in p['models']
 if r.get('reused') or r.get('administrative_observation'):continue
 counts['native_outcomes']+=1
 obs=r.get('observed_model_effort',[])
 if not obs:
  counts['pre_agent_setup_errors']+=1
  if r['reward'] is not None or not r['exception']:issues.append(r['trial'])
 elif any(x!=[r['model'],'high'] for x in obs):issues.append(r['trial'])
 uncertain=sum(bool(r.get('cost_range_usd')) and abs(r['cost_range_usd'][1]-r['cost_range_usd'][0])>1e-12 for r in d['trials'])
blocks=[json.loads(x.read_text()) for x in (B/'blocks').glob('*/block.json')]
contract=json.loads((R/'results/presentation-20261014/update-on-completion.json').read_text());deckhash=hashlib.sha256(Path(contract['current_deck']).read_bytes()).hexdigest()
pattern=re.compile(rb'(?:pit_[A-Za-z0-9_-]{24,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})');scanned=0;matches=[]
for f in B.rglob('*'):
 if f.is_file() and f.suffix in {'.json','.csv','.txt','.md','.py'}:
  scanned+=1
  if pattern.search(f.read_bytes()):matches.append(str(f.relative_to(R)))
assert d['status']['state']=='complete' and len(slots)==351 and set(slots)==expected
assert d['status']['fresh_attempts']==313 and d['status']['reused_attempts']==38
assert not d['status']['active_blocks'] and not d['status']['review_blocks']
assert all(x['status']=='completed' for x in blocks)
assert not mismatch and not issues and not matches
native_checks=[]
for r in d['trials']:
 if r.get('reused') or r.get('administrative_observation'):continue
 run=json.loads((R/r['run_record']).read_text());o=next(x for x in run['outcomes'] if x['trial']==r['trial']);native=R/o['result_path']
 check(native,o['result_sha256'],'native_result_files')
 runtime=p['tasks'][r['task']]['runtime'][run['provider']+'/'+r['condition']]
 assert run['source_sha256']==runtime['sha256']
 for rel,h in runtime['sha256'].items():check(native.parent.parent/'frozen-task'/rel,h,'job_frozen_files')
 contexts=set()
 for f in (native.parent/'agent/sessions').rglob('*.jsonl'):
  for line in f.read_text().splitlines():
   try:e=json.loads(line)
   except ValueError:continue
   if e.get('type')=='turn_context':
    model=e['payload'].get('model');model='deepseek/'+model if model=='deepseek-v4.1-flash' else model
    contexts.add((model,e['payload'].get('effort')))
 if contexts:assert contexts=={(r['model'],'high')}
 else:assert r['execution_error']=='NetworkConnectionError' and r['reward'] is None and r['block']==117
 native_checks.append({'trial':r['trial'],'result_sha256':o['result_sha256'],'observed_model_effort':[list(x) for x in sorted(contexts)]})
assert len(native_checks)==311 and not mismatch
admin=[r for r in d['trials'] if r.get('administrative_observation')]
assert len(admin)==2 and all(r['reward'] is None and not r['completed_pass'] for r in admin)
ledger=json.loads(Path('/private/tmp/physlite-hints-prime-session/ledger.json').read_text())
assert not any(x['status']=='pending' for x in ledger['requests'])
primechecks=[]
for block in blocks:
 if not block['model'].startswith('deepseek/'):continue
 folder=B/'blocks'/f"{block['index']:03d}"
 for c in block['condition_order']:
  runpath=folder/(c+'.json')
  if not runpath.exists():continue
  run=json.loads(runpath.read_text())
  if 'prime_cost_range_usd' not in run:continue
  paid=[x for x in ledger['requests'] if x.get('trial_tag')==run.get('prime_trial_tag')]
  if not paid:continue
  bounds=[sum(max(x.get('estimated_token_cost_usd') or 0,x.get('reported_cost_usd') or 0) for x in paid),sum(x['accounted_usd'] for x in paid)]
  assert all(abs(a-b)<1e-10 for a,b in zip(bounds,run['prime_cost_range_usd'])),(block['index'],c,bounds,run['prime_cost_range_usd'])
  assert all(x.get('provider_reasoning_effort')=='high' for x in paid)
  primechecks.append({'block':block['index'],'condition':c,'request_count':len(paid),'cost_range_usd':bounds})
processes=subprocess.check_output(['ps','-axo','command'],text=True)
assert str(S/'tooling/controller.py') not in processes and str(S/'tooling/run_block.py') not in processes
assert not subprocess.check_output(['/Users/yp9846/.docker/bin/docker','ps','-q'],text=True).strip()
record={'audited_utc':datetime.now(timezone.utc).isoformat(),'result':'passed','status':d['status'],'counts':dict(counts),'native_results':native_checks,'prime_accounting':primechecks,'hash_mismatches':mismatch,'duplicate_slots':len(slots)-len(set(slots)),'model_effort_mismatches':issues,'cost_uncertain_observations':uncertain,'credential_pattern_files_scanned':scanned,'credential_pattern_matches':len(matches),'administrative_observations':len(admin),'imported_observations':38,'raw_remote_transcripts_independently_verified':False,'no_active_trials_or_scheduler':True,'no_pending_prime_requests':True,'historical_control_caveat':p['reuse_caveat'],'execution_resource_caveat':p['execution_resource_caveat'],'plan_sha256':hashlib.sha256((B/'plan.json').read_bytes()).hexdigest(),'results_sha256':hashlib.sha256((B/'results.json').read_bytes()).hexdigest(),'presentation_source_sha256':deckhash,'presentation_source_matches_record':deckhash==contract['current_deck_sha256'],'account_usage':{'used_percent':15,'remaining_percent':85,'observation':'Account-wide usage; not the sweep alone.'}}
(W/'final-audit-20261015.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['result','status','counts','hash_mismatches','model_effort_mismatches','cost_uncertain_observations','credential_pattern_matches','presentation_source_matches_record']}))
