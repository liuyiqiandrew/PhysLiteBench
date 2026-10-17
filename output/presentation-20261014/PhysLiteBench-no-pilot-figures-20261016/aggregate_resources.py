"""Add recorded dollar and token coordinates for every setting; never impute missing usage.
Uses the existing deduplicated 2068-attempt summary and archived request ledgers.
Tokens = input (including cached input) + output (including reasoning), each once.
"""
from pathlib import Path
from collections import defaultdict
import json,csv,hashlib,math
U=Path(__file__).resolve().parent;S=U/'sources'
load=lambda p:json.loads(p.read_text())
D=load(U/'summary.json');lookup={};token_names=['input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens']
for f in ['no-hint-results.json','high-hint-results.json']:
 for t in load(S/f)['trials']:
  lookup[f,t['trial']]=t
for f in ['baseline-luna-trials.csv','baseline-sol-terra-trials.csv']:
 for t in csv.DictReader((S/f).open()):lookup[f,t['trial']]=t
H=list(csv.DictReader((S/'hint-matrix-summary.csv').open()))
P=load(S/'provider-resource-records.json')['batches']
components=[]
for c in D['components']:
 x=dict(c);source=c['source'];totals=dict.fromkeys(token_names,0);missing=0;provider_requests=None;native_total=0;total_override=None
 if source=='hint-matrix-summary.csv':
  h=next(h for h in H if (h['model'],h['reasoning_effort'],{'physical':'physics'}.get(h['hint'],h['hint']))==(c['model'],c['effort'],c['condition']))
  totals={k:int(h[k]) if h[k] else None for k in token_names};total_override=int(h['known_recorded_tokens']);missing=c['attempts']-int(h['trials_with_complete_usage']);complete=missing==0
  if h['total_tokens']:assert totals['input_tokens']+totals['output_tokens']==total_override==int(h['total_tokens'])
  evidence='Published aggregate usage. Incomplete cells contain known usage only.'
 elif c['model'].startswith('deepseek'):
  batches=[b for b in P if b['scope']==source and (b['model'],b['effort'],b['condition'])==(c['model'],c['effort'],c['condition']) and set(b['trials']).issubset(set(c['trial_ids']))]
  assert set(t for b in batches for t in b['trials'])==set(c['trial_ids'])
  reqs=[r for b in batches for r in b['requests']];provider_requests=len(reqs)
  for r in reqs:
   u=r.get('usage') or {}
   if not u:missing+=1;continue
   i=u.get('input_tokens',u.get('prompt_tokens'));o=u.get('output_tokens',u.get('completion_tokens'));assert i is not None and o is not None
   assert u.get('total_tokens',i+o)==i+o
   totals['input_tokens']+=i;totals['output_tokens']+=o
   totals['cached_input_tokens']+=(u.get('input_tokens_details',u.get('prompt_tokens_details')) or {}).get('cached_tokens',0)
   totals['reasoning_output_tokens']+=(u.get('output_tokens_details',u.get('completion_tokens_details')) or {}).get('reasoning_tokens',0)
  native_total=sum(sum((lookup[source,t].get('token_usage') or {}).get(k,0) for k in ['input_tokens','output_tokens']) for t in c['trial_ids'])
  assert totals['input_tokens']+totals['output_tokens']>=native_total,(source,c['model'],c['effort'],c['condition'])
  complete=missing==0 and c['cost_complete'];evidence='Provider ledger usage, including requests that finished after agent timeout and retries. Unknown streams retain missing-token uncertainty.'
 else:
  complete=True
  for tid in c['trial_ids']:
   t=lookup[source,tid]
   if source.endswith('.csv'):
    u={k:int(t.get(k) or 0) for k in token_names};good=t['usage_complete']=='True';assert u['input_tokens']+u['output_tokens']==int(t.get('total_tokens') or 0)
   else:
    u=t.get('token_usage',{});good=t.get('usage_complete',True) and t.get('estimated_standard_api_usd') is not None
   for k in token_names:totals[k]+=u.get(k,0)
   if not good:missing+=1;complete=False
  evidence='Saved native request usage or published baseline trial CSV; same trial identities as the deduplicated study.'
 x.update(recorded_tokens=total_override if total_override is not None else totals['input_tokens']+totals['output_tokens'],recorded_token_components=totals,token_complete=complete,missing_usage_units=missing,missing_unit='request' if provider_requests is not None else 'attempt',provider_request_count=provider_requests,native_recorded_tokens_before_provider_reconciliation=native_total if provider_requests else None,token_evidence=evidence)
 if all(v is not None for v in totals.values()):assert totals['cached_input_tokens']<=totals['input_tokens'] and totals['reasoning_output_tokens']<=totals['output_tokens']
 components.append(x)
def optional_sum(cs,k):
 vals=[c['recorded_token_components'][k] for c in cs]
 return sum(vals) if all(v is not None for v in vals) else None
rows=[]
for r in D['summary']:
 cs=[c for c in components if (c['model'],c['effort'],c['condition'])==(r['model'],r['effort'],r['condition'])];assert sum(c['attempts'] for c in cs)==r['attempts']
 rows.append(dict(r,plot_usd=r['mean_cost_usd'] if r['cost_complete'] else r['mean_cost_lower_usd'],usd_status='complete' if r['cost_complete'] else 'bounded' if r['mean_cost_upper_usd'] is not None else 'lower_bound',recorded_total_tokens=sum(c['recorded_tokens'] for c in cs),plot_tokens=sum(c['recorded_tokens'] for c in cs)/r['attempts'],token_complete=all(c['token_complete'] for c in cs),token_status='complete' if all(c['token_complete'] for c in cs) else 'lower_bound',recorded_input_tokens=optional_sum(cs,'input_tokens'),recorded_cached_input_tokens=optional_sum(cs,'cached_input_tokens'),recorded_output_tokens=optional_sum(cs,'output_tokens'),recorded_reasoning_tokens=optional_sum(cs,'reasoning_output_tokens')))
assert len(rows)==48 and sum(r['attempts'] for r in rows)==2068
matched={(r['model'],r['effort']) for r in rows if all(any((q['model'],q['effort'],q['condition'])==(r['model'],r['effort'],c) for q in rows) for c in ['baseline','general','physics'])}
base=[r for r in rows if r['condition']=='baseline'];hint=[r for r in rows if (r['model'],r['effort']) in matched];assert len(base)==24 and len(hint)==36
# Partial pilot stays outside the 13-task study, but is displayed as a labelled cross.
pilot_batches=[b for b in P if b['scope']=='no-hint-results.json' and b['effort']=='none'];assert len(pilot_batches)==1
reqs=pilot_batches[0]['requests'];assert all(r.get('usage') for r in reqs)
pilot_raw=next(r for r in load(S/'no-hint-results.json')['summary'] if r['model'].startswith('deepseek') and r['reasoning_effort']=='none')
pilot=dict(model=pilot_raw['model'],effort='none',condition='baseline',tasks=1,attempts=3,passes=pilot_raw['passes'],errors=pilot_raw['errors'],pass_rate=pilot_raw['pass_rate'],plot_usd=pilot_raw['mean_estimated_api_usd'],plot_tokens=sum(r['usage']['total_tokens'] for r in reqs)/3,cost_complete=True,token_complete=True,scope='Single-task pilot. Not included in primary totals or frontier calculation.')
assert pilot['attempts']==3 and len(pilot_batches[0]['trials'])==3
out={'metadata':{'study_attempts':2068,'settings':48,'no_hint_settings':24,'matched_hint_settings':36,'unique_primary_settings_in_union':48,'single_task_pilot_attempts_outside_primary':3,'previous_no_hint_settings_omitted':5,'previous_hint_settings_omitted':14,'no_hint_partial_usd':sum(not r['cost_complete'] for r in base),'hint_partial_usd':sum(not r['cost_complete'] for r in hint),'no_hint_partial_tokens':sum(not r['token_complete'] for r in base),'hint_partial_tokens':sum(not r['token_complete'] for r in hint),'token_definition':'Sum of request input plus output tokens per attempt, including cached input and reasoning exactly once. Provider requests supersede native client totals for DeepSeek, including retries and late-finishing streams.','cost_definition':'Frozen API-equivalent rates; known costs divided by all attempts. Incomplete coordinates are recorded minima, not imputed complete costs. Provider upper reserves are conservative ceilings, not confirmed spending.','frontier_definition':'Provisional Pareto envelope evaluated at displayed recorded costs; incomplete costs can move right. No interpolation between model configurations is claimed.','coverage_note':'All 48 primary settings appear in the union of figures. Twelve no-hint settings appear in both comparison views but their records are not added twice. Single-task pilot shown separately outside frontier.'},'summary':rows,'components':components,'pilot':pilot}
(U/'resource-summary.json').write_text(json.dumps(out,indent=2)+'\n')
with (U/'resource-summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(out['metadata'],indent=2))
for r in rows:
 if not r['cost_complete']:print(r['model'],r['effort'],r['condition'],round(r['plot_usd'],4),round(r['plot_tokens']),r['usd_status'],r['token_status'])
