from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,os,sys,shutil
R=Path('/Users/yp9846/Desktop/PhysLiteBench');C=R/'staging/overnight-20261011/comparison';P=C/'runs/gpt-5.6-terra--superconducting-heat/run.json'
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def hashes(root):return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and not any(k in p.parts for k in ['__pycache__','.pytest_cache','.git'])}
row=json.loads(P.with_name('before-setup-replacements.json').read_text());plan=json.loads((C/'plan.json').read_text());assert not row.get('additional_jobs')
excluded=[]
for p in sorted((R/'jobs'/row['job']).glob('*/result.json')):
 d=json.loads(p.read_text())
 if not d.get('exception_info'):continue
 assert d.get('agent_execution') is None and d.get('agent_result') is None and d.get('verifier_result') is None
 assert not list((p.parent/'agent/sessions').rglob('*.jsonl'))
 excluded.append(dict(trial=p.parent.name,result_path=str(p.relative_to(R)),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),exception_type=d['exception_info']['exception_type'],reason='Codex CLI installation failed before any model execution, response, tool action or verifier evaluation. Preserve setup failure separately; replace only this unstarted scientific slot.'))
assert len(excluded)==2
assert P.with_name('before-setup-replacements.json').exists()
job='pareto-gpt-5.6-terra-superconducting-heat-setup-replacements-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
row.update(additional_jobs=[job],excluded_pre_model_trials=excluded,status='running',replacement_started_at=now(),replacement_slots_requested=2)
write(P,row)
plan.setdefault('pre_model_setup_failures',[]).extend(excluded);plan.setdefault('protocol_amendments',[]).append(dict(at=now(),reason='Two Terra superconducting-heat attempts failed while installing the pinned Codex CLI; no model execution or score exists.',action='Run exactly two replacement slots. Preserve both setup failures and the completed third scientific attempt; do not rerun its outcome.',replacement_job=job));write(C/'plan.json',plan)
source=C/'tasks/superconducting-heat';assert hashes(source)==row['source_sha256'];assert hashlib.sha256((C/'run_science.py').read_bytes()).hexdigest()==plan['run_science_sha256']
env=dict(os.environ);env['PATH']=str(R/'.cache/bin')+':'+env['PATH'];env['UV_CACHE_DIR']='/private/tmp/physlitebench-inspection-uv';env['UV_PYTHON_INSTALL_DIR']=str(R/'.cache/python')
cmd=[sys.executable,str(C/'run_science.py'),str(source),'--job-name',job,'--model','gpt-5.6-terra','--reasoning-effort','high','--trials','2','--concurrency','2','--agent-setup-timeout-multiplier','3']
print('START',job,flush=True)
with P.with_name('setup-replacements.log').open('w') as out:proc=subprocess.run(cmd,cwd=R,env=env,stdout=out,stderr=subprocess.STDOUT)
records=[];excluded_ids={x['trial'] for x in excluded}
for j in [row['job'],*row['additional_jobs']]:
 for p in sorted((R/'jobs'/j).glob('*/result.json')):
  if p.parent.name in excluded_ids:continue
  d=json.loads(p.read_text());records.append(dict(trial=p.parent.name,reward=(d.get('verifier_result') or {}).get('rewards',{}).get('reward'),exception=d.get('exception_info'),result_path=str(p.relative_to(R))))
row.update(returncode=proc.returncode,finished_at=now(),records=records,passes=sum(t['reward']==1 and t['exception'] is None for t in records),errors=sum(t['exception'] is not None for t in records),trials=len(records),status='completed')
if proc.returncode or len(records)!=3 or row['errors']:row['status']='needs_review'
assert hashes(source)==row['source_sha256'];write(P,row);print('RESULT',json.dumps({k:row[k] for k in ['task','model','status','passes','errors','trials']}),flush=True)
