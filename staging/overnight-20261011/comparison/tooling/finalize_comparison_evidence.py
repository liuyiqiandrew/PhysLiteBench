from pathlib import Path
from datetime import datetime,timezone
import ast,csv,difflib,hashlib,json,re,shutil

R=Path('/Users/yp9846/Desktop/PhysLiteBench');S=R/'staging/overnight-20261011';C=S/'comparison';O=R/'results/model-comparison-20261011'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and not any(s in f.parts for s in ['__pycache__','.pytest_cache','.git'])}
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
data=read(C/'results.json');plan=read(C/'plan.json');qualification=read(R/'results/qualified-tasks.json')
assert not O.exists(),'Preserve previous finalized evidence; never overwrite it.'
fresh=[r for r in data['trials'] if not r['development_selected']]
assert len(fresh)==90 and len({r['trial'] for r in fresh})==90
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
audit=[]
for t in fresh:
 trial=R/'jobs'/jobs[(t['model'],t['task'])]/t['trial']
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
 tree=ast.parse(source);imports=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Import):imports.extend(x.name for x in n.names)
  if isinstance(n,ast.ImportFrom):imports.append(n.module)
 flags=[dict(line=i+1,text=line[:220]) for i,line in enumerate(source.splitlines()) if re.search(r'/tests|test_hidden|/logs/verifier|reward\.txt|subprocess|socket|requests|eval\(|exec\(|__import__|setattr\(',line)]
 audit.append(dict(task=t['task'],model=t['model'],trial=t['trial'],reward=t['reward'],exception=t['exception'],files=hashes(target),imports=sorted(set(imports)),review_flags=flags,network_command_flags=t['network_command_scan']))
for name in ['plan.json','qualification-manifest.json','results.json','reviewed-observations.json','scheduling.json']:
 shutil.copy2(C/name,O/name)
shutil.copy2(S/'results/comparison-prices.json',O/'pricing.json')
write(O/'evidence-audit.json',dict(created_at=datetime.now(timezone.utc).isoformat(),trials=90,task_source_hashes_verified=10,native_usage_hashes_verified=True,per_request_cost_sums_verified=True,method='Read-only source and command pattern scan; flags require interpretation and absence is not a proof. No submitted code was executed on the host.',trials_review=audit))
summary={x['model']:x for x in data['summary']}
models=['gpt-6-luna','gpt-6-sol','gpt-6-astra'];labels={'gpt-6-luna':'GPT-6 Luna','gpt-6-sol':'GPT-6 Sol','gpt-6-astra':'GPT-6 Astra'}
lines=['## Completed model comparison','',f"Completed {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. All three fresh panels used the same ten frozen tasks, High reasoning, three attempts per task, Codex CLI 0.160.0, and 600/60-second solving/verifying limits.",'','| Model | Completed passes | Execution errors | Mean estimated API cost per attempt |','|---|---:|---:|---:|']
for model in models:
 r=summary[model];lines.append(f"| {labels[model]} | {r['passes']}/30 | {r['errors']} | ${r['mean_standard_api_usd']:.5f} |")
lines+=['','![Cost–score frontier](output/overnight-20261011/model-cost-score-pareto.png)','','Costs apply the published Standard API rates to each request’s recorded input, cache and output usage. They exclude local compute and are not the Pro subscription bill. Rates and source links are recorded in [pricing.json](results/model-comparison-20261011/pricing.json). [CSV](output/overnight-20261011/model-cost-score.csv), [PDF](output/overnight-20261011/model-cost-score-pareto.pdf), [SVG](output/overnight-20261011/model-cost-score-pareto.svg), [complete results](results/model-comparison-20261011/results.json), and [submitted-source evidence](results/model-comparison-20261011/evidence) are available.','','The GPT-5.6 Luna point is the selected 0/30 development baseline, using CLI 0.154.0, and is excluded from the fresh-model frontier. These small trial counts are clustered by task; the plotted Wilson intervals are descriptive. Successful timeout checkpoints, if any, remain separate from completed successes in the data.','','The original 0.154.0 runner rejected six GPT-6 requests before any model answer or tool use. Those access failures and their unchanged-starter verifier scores are retained in the protocol record. The client migration was recorded before any fresh model benchmark answer; every fresh comparison panel then used 0.160.0 consistently. No completed model outcome was resampled.','','GPT-6 Terra was unavailable in the model catalog; the requested substitution decision remains unanswered. DeepSeek was not run because its provider/key-file path and spending cap remain unavailable. Neither has an invented score or cost point.','','| Task | GPT-6 Luna | GPT-6 Sol | GPT-6 Astra |','|---|---:|---:|---:|']
for task in plan['tasks']:
 cells=[]
 for model in models:
  t=summary[model]['per_task'][task];cells.append(f"{t['passes']}/3"+(f" ({t['errors']} errors)" if t['errors'] else ''))
 lines.append('| '+task+' | '+' | '.join(cells)+' |')
p=R/'DASHBOARD.md';text=p.read_text().split('\n## Fresh model comparison')[0]
start=text.index('The separate cost–score comparison is in progress.');end=text.index('\n\n',start)
text=text[:start]+'The fresh comparison of GPT-6 Luna, Sol and Astra is complete; the results appear below. GPT-6 Terra and DeepSeek remain pending model/access details.'+text[end:]
p.write_text(text.rstrip()+'\n\n'+'\n'.join(lines)+'\n')
state=read(S/'results/next-work.json');state.update(updated_at=datetime.now(timezone.utc).isoformat(),active_sessions=[],comparison_complete_batches=30,comparison_completed_models=models,next='Review final figure and evidence. Remaining requested models need user input; do not fabricate results or mark the entire goal complete while they remain unrun.');write(S/'results/next-work.json',state)
print(json.dumps(dict(export=str(O),trials=90,flags=sum(len(r['review_flags']) for r in audit),network_flags=sum(len(r['network_command_flags']) for r in audit),summary=[{k:r[k] for k in ['model','passes','attempts','errors','mean_standard_api_usd']} for r in data['summary']])))
