from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
R=Path('/Users/yp9846/Desktop/PhysLiteBench');S=R/'staging/overnight-20261011';C=S/'comparison'
subprocess.run([sys.executable,'/private/tmp/collect_comparison.py'],check=True,stdout=subprocess.DEVNULL)
data=json.loads((C/'results.json').read_text());plan=json.loads((C/'plan.json').read_text());runs=[json.loads(p.read_text()) for p in sorted((C/'runs').glob('*/run.json'))]
summary={r['model']:r for r in data['summary']};models=plan['models']
lines=['## Fresh model comparison','',f"Updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. All fresh panels use Codex CLI 0.160.0, High reasoning, identical frozen apparatus and grading, and 600/60-second solving/verifying budgets.",'','| Task | GPT-6 Luna | GPT-6 Sol | GPT-6 Astra |','|---|---:|---:|---:|']
for task in plan['tasks']:
 cells=[]
 for model in models:
  s=summary.get(model,{}).get('per_task',{}).get(task)
  if s:cells.append(f"{s['passes']}/{s['attempts']}"+(f" ({s['errors']} errors)" if s['errors'] else '')+(' in progress' if s['attempts']<3 else ''))
  else:cells.append('Running' if any(r['task']==task and r['model']==model for r in runs) else 'Pending')
 lines.append('| '+task+' | '+' | '.join(cells)+' |')
lines+=['','Cells count completed passing attempts over attempts with a terminal result. Execution exceptions remain in the denominator; passing timeout checkpoints are retained separately in the raw summary. Incomplete panels do not establish comparable aggregate scores. Six older-client access failures before model output are preserved in `staging/overnight-20261011/comparison/pre-client-upgrade/` and excluded from scientific scores.','']
p=R/'DASHBOARD.md';text=p.read_text();text=text.split('\n## Fresh model comparison')[0].rstrip();p.write_text(text+'\n\n'+'\n'.join(lines))
state=json.loads((S/'results/next-work.json').read_text());state['updated_at']=datetime.now(timezone.utc).isoformat();state['active_sessions']=[dict(model=r['model'],task=r['task'],job=r['job']) for r in runs if r['status']=='running'];state['comparison_complete_batches']=sum(r['status']=='completed' for r in runs);state['comparison_needs_review']=[dict(task=r['task'],model=r['model'],job=r['job'],errors=r.get('errors')) for r in runs if r['status']=='needs_review'];(S/'results/next-work.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(dict(summary=[dict(model=r['model'],passes=r['passes'],trials=r['attempts'],errors=r['errors'],complete=r['complete_panel'],missing_usage=r['missing_usage_trials']) for r in data['summary']],active=[dict(model=r['model'],task=r['task']) for r in state['active_sessions']])))
