from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, collections, csv
R=Path('/Users/yp9846/Desktop/PhysLiteBench');B=R/'results/hint-comparison-20261014';W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench/.presentation-local-final-20261015')
d=json.loads((B/'results.json').read_text());p=json.loads((B/'plan.json').read_text())
assert d['status']['state']=='complete' and d['status']['recorded_attempts']==351 and not d['status']['review_blocks'] and not d['status']['active_blocks']
rows=d['trials'];models=p['models'];conds=p['conditions'];labels={'baseline':'No hint','general':'General hint','physics':'Physics hint'}
panels=[]
for s in d['summary']:
 a=[x for x in rows if x['model']==s['model'] and x['condition']==s['condition']]
 assert len(a)==39 and all(sum(x['task']==t for x in a)==3 for t in p['tasks'])
 costs=[x['estimated_standard_api_usd'] for x in a if x['estimated_standard_api_usd'] is not None]
 prime_bounds=[x['cost_range_usd'] for x in a if x.get('cost_range_usd')]
 panels.append({**{k:v for k,v in s.items() if k!='per_task'},'cost_covered_attempts':len(costs),
  'known_cost_sum_usd':sum(x.get('known_recorded_api_usd',x.get('estimated_standard_api_usd')) or 0 for x in a),
  'cost_range_sum_usd':[sum(x[0] for x in prime_bounds),sum(x[1] for x in prime_bounds)] if len(prime_bounds)==39 else None,
  'exception_counts':dict(collections.Counter(x['exception'].get('exception_type','UnknownError') if isinstance(x['exception'],dict) else x['exception'] for x in a if x['exception'])),
  'graded_failures_without_execution_error':sum(not x['completed_pass'] and not x['exception'] for x in a),
  'checkpoint_passes_with_execution_error':sum(x['reward']==1 and bool(x['exception']) for x in a)})
frontiers={}
for c in conds:
 eligible=[s for s in panels if s['condition']==c and s['mean_estimated_api_usd'] is not None]
 frontiers[c]=[s for s in eligible if not any(t!=s and t['mean_estimated_api_usd']<=s['mean_estimated_api_usd'] and t['passes']>=s['passes'] and (t['mean_estimated_api_usd']<s['mean_estimated_api_usd'] or t['passes']>s['passes']) for t in eligible)]
data={'created_utc':datetime.now(timezone.utc).isoformat(),'results_sha256':hashlib.sha256((B/'results.json').read_bytes()).hexdigest(),'panels':panels,'frontiers':frontiers,'status':d['status'],'reuse_caveat':p['reuse_caveat'],'resource_caveat':p['execution_resource_caveat']}
(W/'data.json').write_text(json.dumps(data,indent=2)+'\n')
lines=['Physics hint comparison: final results','', '351 observations: 313 local attempts and 38 imported historical Luna High baselines.','13 frozen tasks, 3 models, 3 prompt conditions, 3 repeats. All model reasoning efforts are High.','', 'A completed pass requires verifier reward 1 and no execution exception. Every started slot remains in the denominator. Execution errors are not assigned a physics failure mode.','', 'Model | Condition | Completed passes / attempts | Execution errors | Graded failures without execution error | Mean API-equivalent cost / attempt']
for s in panels:
 cost=f"${s['mean_estimated_api_usd']:.8f}" if s['mean_estimated_api_usd'] is not None else f"withheld ({s['cost_covered_attempts']}/39 costs complete)"
 lines.append(f"{s['model']} | {labels[s['condition']]} | {s['passes']}/39 | {s['errors']} | {s['graded_failures_without_execution_error']} | {cost}")
lines+=['','Interpretation','Physics hints have the highest observed completed-pass counts for Luna and Terra in this cohort. This is a descriptive result. Luna uses historical controls, and runtime resources changed during the local study. Hint length is not matched. DeepSeek execution errors dominate its results, so its low completion rate does not establish a physics reasoning deficit.','',p['reuse_caveat'],p['execution_resource_caveat'],'Imported published aggregates, source metadata and hashes were checked locally. Raw remote transcripts were not independently verified locally.','', 'Cost and frontier rules','Costs are API-equivalent estimates for Codex usage and reconciled provider estimates for Prime, not Codex subscription charges. A setting enters its prompt-condition frontier only when all 39 costs are available. There are four eligible settings; five are withheld. Singleton frontiers are shown as points. These are observed frontiers restricted to complete-cost settings, not proven frontiers over the missing settings. Seven Prime observations retain nonzero uncertainty bounds. Upper reserves are not confirmed spending.','']
for s in panels:
 if s['cost_range_sum_usd']:
  lines.append(f"{s['model']} {labels[s['condition']]} provider cost sum bounds: ${s['cost_range_sum_usd'][0]:.8f} to ${s['cost_range_sum_usd'][1]:.8f}. Upper value includes conservative reserves.")
lines+=['','Execution exceptions']
for s in panels:lines.append(f"{s['model']} / {labels[s['condition']]}: {json.dumps(s['exception_counts'],sort_keys=True)}; reward-1 checkpoints with execution errors: {s['checkpoint_passes_with_execution_error']} (excluded from completed passes).")
lines+=['','Per-task results (passes / 3; execution errors in parentheses)']
for t in p['tasks']:
 lines+=['',t]
 for m in models:
  counts=[]
  for c in conds:
   a=[x for x in rows if x['model']==m and x['task']==t and x['condition']==c]
   counts.append(f"{labels[c]} {sum(x['completed_pass'] for x in a)}/3 ({sum(bool(x['exception']) for x in a)} errors)")
  lines.append(m+': '+', '.join(counts))
lines+=['','Sources',str(B/'results.json'),str(B/'plan.json'),str(B/'reused-baselines-20261015.json'),str(B/'administrative-observations-20261015.json'),'Operational review records remain in '+str(B/'reviews'),'Pricing metadata: '+str(B/'pricing.json'),'Results SHA256: '+data['results_sha256']]
(W/'FINAL-REPORT.txt').write_text('\n'.join(lines)+'\n')
with (W/'final-summary.csv').open('w') as f:
 fields=['model','condition','attempts','passes','errors','graded_failures_without_execution_error','mean_estimated_api_usd','cost_covered_attempts'];writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(panels)
print(json.dumps({'panels':panels,'frontier_points':{c:len(v) for c,v in frontiers.items()}}))
