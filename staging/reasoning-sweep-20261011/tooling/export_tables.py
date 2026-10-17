from pathlib import Path
import collections,csv,json
R=Path(__file__).resolve().parents[3]
E=R/'results/reasoning-sweep-20261011'
d=json.loads((E/'results.json').read_text())
rows=d['trials'];groups=collections.defaultdict(list)
for r in rows:groups[(r['model'],r['reasoning_effort'],r['task'])].append(r)
task_rows=[]
for (model,effort,task),rs in sorted(groups.items()):
 task_rows.append(dict(model=model,effort=effort,task=task,attempts=len(rs),completed_passes=sum(r['reward']==1 and r['exception'] is None for r in rs),passing_checkpoints=sum(r['reward']==1 for r in rs),execution_errors=sum(r['exception'] is not None for r in rs)))
for s in d['summary']:
 selected=[r for r in task_rows if r['model']==s['model'] and r['effort']==s['reasoning_effort']]
 for field,expected in [('attempts','attempts'),('completed_passes','passes'),('passing_checkpoints','scored_checkpoint_passes'),('execution_errors','errors')]:assert sum(r[field] for r in selected)==s[expected]
with (E/'per-task-scores.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(task_rows[0]));w.writeheader();w.writerows(task_rows)
trial_rows=[]
for r in rows:
 trial_rows.append(dict(model=r['model'],effort=r['reasoning_effort'],task=r['task'],trial=r['trial'],completed_pass=int(r['reward']==1 and r['exception'] is None),checkpoint_reward=r['reward'],exception_type=(r['exception'] or {}).get('exception_type'),agent_seconds=r.get('agent_execution_seconds'),reused_high_observation=r['reused'],evidence=r.get('evidence','')))
with (E/'trial-outcomes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(trial_rows[0]));w.writeheader();w.writerows(trial_rows)
inventory=[]
for model in dict.fromkeys(s['model'] for s in d['summary']):
 settings=[dict(effort=s['reasoning_effort'],attempts=s['attempts'],expected_attempts=s['expected_attempts'],complete=s['complete_panel']) for s in d['summary'] if s['model']==model]
 inventory.append(dict(model=model,role='frozen thirteen-task comparator',settings=settings))
inventory.append(dict(model='gpt-5.6-luna',role='historical task-selection and qualification baseline',included_in_new_frontier=False,evidence=['results/qualified-tasks.json','results/expansion-confirmation-20261011-1709-results.json'],note='Original ten-task observations were selected during task development. The three additions have separate remote selection and confirmation records. No fresh complete thirteen-task comparator panel for this model is claimed.'))
(E/'model-inventory.json').write_text(json.dumps({'updated_utc':d['updated_utc'],'models':inventory},indent=2)+'\n')
print(json.dumps({'per_task_rows':len(task_rows),'trial_rows':len(trial_rows),'model_inventory':len(inventory),'aggregate_checks':'passed'}))
