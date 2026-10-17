from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,hashlib,json,os,subprocess,sys
R=Path('/Users/yp9846/Desktop/PhysLiteBench');S=R/'staging/reasoning-sweep-20261011';E=R/'results/reasoning-sweep-20261011'
def now():return datetime.now(timezone.utc)
def hashes(root):return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
def main():
 p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('model');p.add_argument('effort',choices=['low','high','xhigh']);a=p.parse_args()
 plan=json.loads((E/'plan.json').read_text());assert dict(task=a.task,model=a.model,effort=a.effort) in plan['queue']
 src=S/'tasks'/a.task;expected=plan['tasks'][a.task]['runtime_source_sha256'];assert hashes(src)==expected
 runner=S/'tooling/run_science.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()==plan['runner_sha256']
 with (S/'launch.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  usage=json.loads((S/'current-usage.json').read_text());age=(now()-datetime.fromisoformat(usage['observed_utc'])).total_seconds()
  assert 0<=age<120,'A fresh account-usage observation is required'
  assert usage['used_percent']<plan['budget']['new_batch_guard_percent'],'Account launch guard reached'
  running=[]
  for f in (E/'runs').glob('*/run.json'):
   r=json.loads(f.read_text())
   if r['status']=='running':running.append(r)
   if r['status']=='needs_review' and r['model']==a.model:raise RuntimeError('Review this model’s prior infrastructure error before another launch')
  capacity=3 if usage['used_percent']>=40 else 6
  remaining=sum(max(0,r['trials_requested']-len(list((R/'jobs'/r['job']).glob('*/result.json')))) for r in running)
  assert remaining+3<=capacity,'Global concurrent-trial limit reached'
  folder=E/'runs'/f'{a.model}--{a.effort}--{a.task}';folder.mkdir()
  stamp=now().strftime('%Y%m%d-%H%M%S');job=f'effort-{a.model}-{a.effort}-{a.task}-{stamp}'
  row=dict(task=a.task,model=a.model,reasoning_effort=a.effort,job=job,trials_requested=3,status='running',started_utc=now().isoformat(),usage_at_launch=usage,source_sha256=expected,plan_sha256=hashlib.sha256((E/'plan.json').read_bytes()).hexdigest(),runner_sha256=plan['runner_sha256'])
  def save():
   temp=folder/'run.json.tmp';temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(folder/'run.json')
  save()
 print('START',job,flush=True)
 env=dict(os.environ);env['PATH']=str(R/'.cache/bin')+':'+env['PATH'];env['UV_CACHE_DIR']='/private/tmp/physlitebench-inspection-uv';env['UV_PYTHON_INSTALL_DIR']=str(R/'.cache/python')
 cmd=[sys.executable,str(runner),str(src),'--job-name',job,'--model',a.model,'--reasoning-effort',a.effort,'--trials','3','--concurrency','3','--agent-setup-timeout-multiplier','3']
 with (folder/'runner.log').open('w') as out:proc=subprocess.run(cmd,cwd=R,env=env,stdout=out,stderr=subprocess.STDOUT)
 rows=[]
 for f in sorted((R/'jobs'/job).glob('*/result.json')):
  r=json.loads(f.read_text());exc=r.get('exception_info');reward=(r.get('verifier_result') or {}).get('rewards',{}).get('reward')
  rows.append(dict(trial=f.parent.name,reward=reward,exception_type=(exc or {}).get('exception_type'),result_path=str(f.relative_to(R))))
 row.update(returncode=proc.returncode,finished_utc=now().isoformat(),trials=rows,passes=sum(t['reward']==1 and not t['exception_type'] for t in rows),errors=sum(bool(t['exception_type']) for t in rows),status='completed')
 if proc.returncode or len(rows)!=3 or row['errors']:row['status']='needs_review'
 assert hashes(src)==expected;save();print('RESULT',json.dumps({k:row[k] for k in ['task','model','reasoning_effort','status','passes','errors','job']}),flush=True)
if __name__=='__main__':main()
