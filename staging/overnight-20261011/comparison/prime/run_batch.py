from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys
R=Path('/Users/yp9846/Desktop/PhysLiteBench');S=R/'staging/overnight-20261011';C=S/'comparison'
def hashes(root):return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and not any(k in p.parts for k in ['__pycache__','.pytest_cache','.git'])}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('task');parser.add_argument('model');parser.add_argument('--observed-usage',type=float,required=True);a=parser.parse_args();assert a.observed_usage<47
    plan=json.loads((C/'plan.json').read_text());assert a.task in plan['tasks'] and a.model in plan['models'];source=C/'tasks'/a.task;assert hashes(source)==plan['tasks'][a.task]['comparison_source_sha256']
    assert hashlib.sha256((C/'prime/run_science.py').read_bytes()).hexdigest()==plan['prime_protocol']['runner_sha256']
    folder=C/'runs'/f"{a.model.replace('/','--')}--{a.task}";folder.mkdir()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S');job=f"pareto-{a.model.replace('/','-')}-{a.task}-{stamp}"
    row=dict(task=a.task,model=a.model,job=job,trials_requested=3,reasoning_effort='high',codex_cli_version=plan['codex_cli_version'],runner_sha256=plan['prime_protocol']['runner_sha256'],observed_account_used_percent=a.observed_usage,status='running',started_at=datetime.now(timezone.utc).isoformat(),source_sha256=hashes(source))
    def save():(folder/'run.json').write_text(json.dumps(row,indent=2)+'\n')
    save();print('START',job,flush=True)
    env=dict(os.environ);env['PATH']=str(R/'.cache/bin')+':'+env['PATH'];env['UV_CACHE_DIR']='/private/tmp/physlitebench-inspection-uv';env['UV_PYTHON_INSTALL_DIR']=str(R/'.cache/python')
    cmd=[sys.executable,str(C/'prime/run_science.py'),str(source),'--job-name',job,'--model',a.model,'--reasoning-effort','high','--trials','3','--concurrency','3','--agent-setup-timeout-multiplier','3']
    with (folder/'runner.log').open('w') as stream:proc=subprocess.run(cmd,cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT)
    records=[]
    for p in sorted((R/'jobs'/job).glob('*/result.json')):
        d=json.loads(p.read_text());records.append(dict(trial=p.parent.name,reward=(d.get('verifier_result') or {}).get('rewards',{}).get('reward'),exception=d.get('exception_info'),result_path=str(p.relative_to(R))))
    row.update(returncode=proc.returncode,finished_at=datetime.now(timezone.utc).isoformat(),records=records,passes=sum(t['reward']==1 for t in records),errors=sum(t['exception'] is not None for t in records),trials=len(records),status='completed')
    if proc.returncode or len(records)!=3 or row['errors']:row['status']='needs_review'
    assert hashes(source)==row['source_sha256'];save();print('RESULT',json.dumps({k:row[k] for k in ['task','model','job','status','passes','errors','trials']}),flush=True)
if __name__=='__main__':main()
