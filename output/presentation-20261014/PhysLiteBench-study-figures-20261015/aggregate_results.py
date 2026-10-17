"""Rebuild model/effort/condition summaries from archived evidence, without model calls.

Usage: python aggregate_results.py
Imported hint data is published aggregate evidence, not independently inspected raw logs.
Known baseline trial identities are deduplicated before aggregation. Every original
source record remains in sources/. No worker identity is an experimental factor.
"""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,hashlib,math
ROOT=Path(__file__).resolve().parent; S=ROOT/'sources'
load=lambda n:json.loads((S/n).read_text())
for entry in json.loads((ROOT/'source-manifest.json').read_text()):
 assert hashlib.sha256((ROOT/entry['file']).read_bytes()).hexdigest()==entry['sha256'],entry['file']
components=[]; observations=[]; known={}; duplicates=[]
def exc(t):
 x=t.get('execution_error') or t.get('exception');return x.get('exception_type') if isinstance(x,dict) else x
def add_records(ts,source,condition=None,remote=False):
 accepted=[]
 for t in ts:
  model=t['model'];effort=t['reasoning_effort'];cond=condition or t['condition'];key=(model,effort,cond,t['trial'])
  error=t.get('exception_type') or '' if remote else exc(t)
  passed=t.get('passed')=='True' if remote else bool(t.get('reward')==1 and not error)
  if key in known:
   prior=known[key];assert prior['passed']==passed and prior['error']==(error or '')
   duplicates.append({'model':model,'effort':effort,'condition':cond,'trial':t['trial'],'kept_source':prior['source'],'duplicate_source':source});continue
  row={'model':model,'effort':effort,'condition':cond,'trial':t['trial'],'task':t['task'],'passed':passed,'error':error or '', 'source':source}
  if remote:
   row.update(cost=float(t['estimated_api_cost_usd']) if t.get('estimated_api_cost_usd') else None,known_cost=float(t.get('known_request_cost_usd') or 0),cost_complete=t.get('usage_complete')=='True')
  else:
   c=t.get('estimated_standard_api_usd');bounds=t.get('cost_range_usd');complete=c is not None and t.get('usage_complete',True) and (bounds is None or math.isclose(bounds[0],bounds[1],abs_tol=1e-12))
   row.update(cost=c,known_cost=t.get('known_recorded_api_usd',c or 0),cost_complete=complete)
  known[key]=row;accepted.append(row);observations.append(row)
 return accepted

def records_component(rows,source,cost_override=None):
 assert rows
 groups=defaultdict(list)
 for row in rows:groups[(row['model'],row['effort'],row['condition'])].append(row)
 for key,ts in groups.items():
  n=len(ts);passed=sum(t['passed'] for t in ts);errors=sum(bool(t['error']) for t in ts)
  exact=all(t['cost_complete'] and t['cost'] is not None for t in ts)
  lo=sum(t['known_cost'] or 0 for t in ts);hi=sum(t['cost'] for t in ts) if exact else None
  if cost_override is not None:
   s=cost_override[key[:2]];exact=s['fully_observed_cost'];lo,hi=s.get('prime_total_cost_range_usd',[s['total_estimated_api_usd'],s['total_estimated_api_usd'] if exact else None])
   assert n==s['attempts'] and passed==s['passes'] and errors==s['errors']
  components.append(dict(model=key[0],effort=key[1],condition=key[2],attempts=n,passes=passed,errors=errors,graded_failures=n-passed-errors,cost_complete=bool(exact),cost_lower_total=lo,cost_upper_total=hi,cost_total=hi if exact else None,task_count=len({t['task'] for t in ts}),source=source,record_level='trial',trial_ids=[t['trial'] for t in ts]))

# Completed no-hint settings. Preserve, but exclude the three-attempt None pilot.
d=load('no-hint-results.json');summ={(s['model'],s['reasoning_effort']):s for s in d['summary'] if s['complete_panel']}
partial=[t for t in d['trials'] if (t['model'],t['reasoning_effort']) not in summ]
records_component(add_records([t for t in d['trials'] if (t['model'],t['reasoning_effort']) in summ],'no-hint-results.json','baseline'),'no-hint-results.json',summ)
# All 390 baseline records, including one historical Luna trial not reused by the later schedule.
for label in ['luna','sol-terra']:
 name=f'baseline-{label}-trials.csv';rows=list(csv.DictReader((S/name).open()))
 records_component(add_records(rows,name,'baseline',remote=True),name)
# Published 780 hinted attempts. Trial identifiers/raw transcripts are not published.
for r in csv.DictReader((S/'hint-matrix-summary.csv').open()):
 n=int(r['finished_trials']);passed=int(r['passes']);errors=int(r['timeouts'])+int(r['other_errors']);exact=bool(r['total_api_cost_usd'])
 assert n==39 and r['panel_complete']=='True' and passed+int(r['graded_failures'])+errors==n
 components.append(dict(model=r['model'],effort=r['reasoning_effort'],condition={'physical':'physics'}.get(r['hint'],r['hint']),attempts=n,passes=passed,errors=errors,graded_failures=int(r['graded_failures']),cost_complete=exact,cost_lower_total=float(r['known_request_cost_usd']),cost_upper_total=float(r['total_api_cost_usd']) if exact else None,cost_total=float(r['total_api_cost_usd']) if exact else None,task_count=13,source='hint-matrix-summary.csv',record_level='published_aggregate',trial_ids=None))
# Later High trials, including 38 identical historical baselines removed by trial identity.
d=load('high-hint-results.json');fresh=add_records(d['trials'],'high-hint-results.json');assert len(fresh)==313
records_component(fresh,'high-hint-results.json')
# Provider bounds for later DeepSeek components must preserve unconfirmed upper reserves.
for c in components:
 if c['source']=='high-hint-results.json' and c['model'].startswith('deepseek'):
  ts=[t for t in d['trials'] if t['model']==c['model'] and t['condition']==c['condition']]
  assert all(t.get('cost_range_usd') for t in ts)
  c['cost_lower_total']=sum(t['cost_range_usd'][0] for t in ts)
  c['cost_upper_total']=sum(t['cost_range_usd'][1] for t in ts)
# Exact model generations remain distinct. Rates are attempt weighted, not mean-of-means.
groups=defaultdict(list)
for c in components:groups[(c['model'],c['effort'],c['condition'])].append(c)
models=['gpt-5.6-luna','gpt-5.6-terra','gpt-5.6-sol','gpt-6-luna','gpt-6-sol','gpt-6-astra','deepseek/deepseek-v4.1-flash'];efforts=['none','low','medium','high','xhigh','max'];conditions=['baseline','general','physics']
summary=[]
for (model,effort,condition),cs in sorted(groups.items(),key=lambda x:(models.index(x[0][0]),efforts.index(x[0][1]),conditions.index(x[0][2]))):
 n=sum(c['attempts'] for c in cs);exact=all(c['cost_complete'] for c in cs);lower=sum(c['cost_lower_total'] for c in cs);upper=sum(c['cost_upper_total'] for c in cs) if all(c['cost_upper_total'] is not None for c in cs) else None
 summary.append(dict(model=model,effort=effort,condition=condition,tasks=13,attempts=n,passes=sum(c['passes'] for c in cs),errors=sum(c['errors'] for c in cs),graded_failures=sum(c['graded_failures'] for c in cs),pass_rate=sum(c['passes'] for c in cs)/n,cost_complete=exact,mean_cost_usd=sum(c['cost_total'] for c in cs)/n if exact else None,mean_cost_lower_usd=lower/n,mean_cost_upper_usd=upper/n if upper is not None else None,source_count=len(cs),sources=';'.join(c['source'] for c in cs)))
assert len(duplicates)==38 and sum(s['attempts'] for s in summary)==2068 and len(summary)==48
assert len(observations)==1288 and len(partial)==3
assert len({(t['model'],t['effort'],t['condition'],t['trial']) for t in observations})==1288
for s in summary:assert s['passes']+s['errors']+s['graded_failures']==s['attempts']
# Comparable prompt coverage for hint frontier: same reported model/effort groups in all three conditions.
matched={k[:2] for k in groups if all((*k[:2],c) in groups for c in conditions)}
def frontier(rows):
 exact=[r for r in rows if r['cost_complete']]
 return [r for r in sorted(exact,key=lambda t:(t['mean_cost_usd'],-t['pass_rate'])) if not any((q['mean_cost_usd']<=r['mean_cost_usd'] and q['pass_rate']>=r['pass_rate'] and (q['mean_cost_usd']<r['mean_cost_usd'] or q['pass_rate']>r['pass_rate'])) for q in exact)]
fronts={c:frontier([r for r in summary if r['condition']==c and (r['model'],r['effort']) in matched]) for c in conditions}
meta={'attempts':2068,'tasks':13,'model_versions':len(models),'model_effort_condition_settings':len(summary),'known_duplicate_records_removed':len(duplicates),'trial_level_records':len(observations),'published_aggregate_attempts':780,'partial_attempts_retained_outside_primary':len(partial),'preagent_setup_records_retained_outside_published_matrix':5,'condition_attempts':{c:sum(s['attempts'] for s in summary if s['condition']==c) for c in conditions},'matched_hint_model_effort_groups':len(matched),'exact_cost_settings':sum(s['cost_complete'] for s in summary),'all_no_hint_frontier':frontier([r for r in summary if r['condition']=='baseline']),'hint_frontiers':fronts,'scope':'One study, grouped by exact model version, reasoning effort and reported prompt condition. Runtime/batch provenance retained as metadata. Descriptive attempt-weighted pooling, not a randomized causal hint estimate.','limitations':['38 known duplicate baseline records removed using exact trial identities. Raw identities for the 780 published hinted observations are unavailable locally; these remain aggregate-only evidence.','Luna High no-hint has 40 trials: four entropy-anomaly trials and three for each other task. Other pooled settings have three or six trials per task.','Five historical pre-agent setup records were replaced in the published 780 matrix; one later setup failure and two administrative interruptions remain in the High schedule. This protocol difference is retained rather than retroactively resampling.','Client versions, CPU quotas, concurrency and calendar date differ. Hint length is not controlled.','Full cost is unknown whenever any component is incomplete. Conservative upper reserves are not confirmed spending. Exact frontiers omit incomplete-cost settings.','Three DeepSeek None pilot records are preserved but excluded because only one of 13 tasks was attempted.','Initial task selection used GPT-5.6 Luna; selection outcomes differ from subsequent evaluations.']}
(ROOT/'summary.json').write_text(json.dumps({'metadata':meta,'summary':summary,'components':components},indent=2)+'\n')
with (ROOT/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
with (ROOT/'deduplicated-trials.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(observations[0]));w.writeheader();w.writerows(observations)
(ROOT/'deduplication-audit.json').write_text(json.dumps({'removed':duplicates,'excluded_partial_trial_ids':[t['trial'] for t in partial]},indent=2)+'\n')
print(json.dumps({k:v for k,v in meta.items() if k not in ['hint_frontiers','all_no_hint_frontier','limitations','scope']},indent=2))
for s in summary:
 if s['effort']=='high':print(s['model'],s['condition'],f"{s['passes']}/{s['attempts']}",s['errors'],'errors',s['mean_cost_usd'])
