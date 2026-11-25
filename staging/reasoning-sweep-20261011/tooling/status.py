from pathlib import Path
import json
R=Path('/Users/yp9846/Desktop/PhysLiteBench');E=R/'results/reasoning-sweep-20261011'
plan=json.loads((E/'plan.json').read_text());runs=[json.loads(p.read_text()) for p in (E/'runs').glob('*/run.json')]
keys={(r['task'],r['model'],r['reasoning_effort']) for r in runs}
active=[{k:r[k] for k in ['task','model','reasoning_effort','job']} for r in runs if r['status']=='running']
review=[{k:r[k] for k in ['task','model','reasoning_effort','job','errors']} for r in runs if r['status']=='needs_review']
ready=[q for q in plan['queue'] if (q['task'],q['model'],q['effort']) not in keys]
schedule=json.loads((E/'scheduling.json').read_text())
priority=schedule.get('queue_priority_amendment',{}).get('panel_priority',[])
rank={tuple(x):i for i,x in enumerate(priority)}
ready.sort(key=lambda q:rank.get((q['model'],q['effort']),len(rank)))
finished=[{k:r[k] for k in ['task','model','reasoning_effort','passes','errors']} for r in runs if r['status']!='running']
terminal=0
for r in runs:
 terminal+=len(list((R/'jobs'/r['job']).glob('*/result.json')))
for a in active:a['terminal_attempts']=len(list((R/'jobs'/a['job']).glob('*/result.json')))
print(json.dumps({'terminal_new_attempts':terminal,'active':active,'needs_review':review,'finished':finished,'started_batches':len(runs),'openai_started_batches':sum(r['model'].startswith('gpt-') for r in runs),'next':ready[:2],'remaining_batches':len(ready)}))
