from pathlib import Path
from datetime import datetime,timezone
import ast,csv,difflib,hashlib,json,re,shutil

R=Path('/Users/yp9846/Desktop/PhysLiteBench');S=R/'staging/overnight-20261011';C=S/'comparison';O=R/'results/model-comparison-20261011-extended'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and not any(s in f.parts for s in ['__pycache__','.pytest_cache','.git'])}
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
data=read(C/'results.json');plan=read(C/'plan.json');qualification=read(R/'results/qualified-tasks.json')
assert not O.exists(),'Preserve previous finalized evidence; never overwrite it.'
fresh=[r for r in data['trials'] if not r['development_selected']]
assert len(fresh)==30*len(plan['models']) and len({r['trial'] for r in fresh})==len(fresh)
for model in plan['models']:
 row=next(x for x in data['summary'] if x['model']==model)
 assert row['complete_panel'] and row['attempts']==30 and row['missing_usage_trials']==0
for task,row in qualification['tasks'].items():
 assert hashes(R/'tasks'/task)==hashes(S/'tasks'/task)==row['source_sha256']
 assert hashes(C/'tasks'/task)==plan['tasks'][task]['comparison_source_sha256']
O.mkdir()
jobs={}
for p in (C/'runs').glob('*/run.json'):
 r=read(p);assert r['status'] in ['completed','needs_review'] and r['trials']==3
 jobs[(r['model'],r['task'])]=r['job']
 for job in [r['job'],*r.get('additional_jobs',[])]:
  frozen=R/'jobs'/job/'frozen-task';expected=dict(plan['tasks'][r['task']]['comparison_source_sha256'])
  if r['model']=='deepseek/deepseek-v4.1-flash':
   policy=(C/'tasks'/r['task']/'task.toml').read_text().replace('allowed_hosts = ["chatgpt.com", "*.openai.com"]','allowed_hosts = ["host.docker.internal"]')
   expected['task.toml']=hashlib.sha256(policy.encode()).hexdigest()
  assert hashes(frozen)==expected,(job,'frozen task byte mismatch')

audit=[]
for t in fresh:
 trial=R/t['path']
 assert t['model_names']==[t['model']],(t['trial'],t['model_names'])
 assert sha(trial/'artifacts/app/model.py')==t['source_sha256']
 assert sha(trial/'agent/trajectory.json')==t['trajectory_sha256']
 for native in t['native_sources']:assert sha(R/native['path'])==native['sha256']
 assert abs(sum(e['estimated_standard_api_usd'] for e in t['usage_events'])-t['estimated_standard_api_usd'])<1e-10
 target=O/'evidence'/t['model']/t['task']/t['trial'];target.mkdir(parents=True)
 for src,dest in [('artifacts/app/model.py','model.py'),('verifier/metrics.json','metrics.json'),('verifier/reward.txt','reward.txt'),('verifier/test-stdout.txt','test-stdout.txt')]:
  if (trial/src).exists():shutil.copy2(trial/src,target/dest)
 starter=(C/'tasks'/t['task']/'environment/model.py').read_text().splitlines(True);source=(target/'model.py').read_text()
 (target/'model.diff').write_text(''.join(difflib.unified_diff(starter,source.splitlines(True),fromfile='starter/model.py',tofile='submission/model.py')))
 imports=[]
 syntax_error=None
 try:tree=ast.parse(source)
 except SyntaxError as exc:tree=ast.parse("");syntax_error=str(exc)
 for n in ast.walk(tree):
  if isinstance(n,ast.Import):imports.extend(x.name for x in n.names)
  if isinstance(n,ast.ImportFrom):imports.append(n.module)
 flags=[dict(line=i+1,text=line[:220]) for i,line in enumerate(source.splitlines()) if re.search(r'/tests|test_hidden|/logs/verifier|reward\.txt|subprocess|socket|requests|eval\(|exec\(|__import__|setattr\(',line)]
 audit.append(dict(task=t['task'],model=t['model'],trial=t['trial'],reward=t['reward'],exception=t['exception'],files=hashes(target),imports=sorted(set(imports)),syntax_error=syntax_error,review_flags=flags,network_command_flags=t['network_command_scan']))
for failure in plan.get('pre_model_setup_failures',[]):
 source=R/failure['result_path'];assert sha(source)==failure['result_sha256']
 target=O/'setup-failures'/failure['trial'];target.mkdir(parents=True);shutil.copy2(source,target/'result.json')
for name in ['plan.json','qualification-manifest.json','results.json','reviewed-observations.json','scheduling.json','timing-audit.json']:
 shutil.copy2(C/name,O/name)
shutil.copy2(S/'results/comparison-prices.json',O/'pricing.json')
write(O/'evidence-audit.json',dict(created_at=datetime.now(timezone.utc).isoformat(),trials=len(fresh),task_source_hashes_verified=10,native_usage_hashes_verified=True,per_request_cost_sums_verified=True,method='Read-only source and command pattern scan; flags require interpretation and absence is not a proof. No submitted code was executed on the host.',trials_review=audit))
