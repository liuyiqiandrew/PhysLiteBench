"""Parse and reconcile the published outcome review without adding benchmark trials."""
from pathlib import Path
from collections import Counter,defaultdict
import json,csv,hashlib
R=Path(__file__).resolve().parent;S=R/'sources'
C=['baseline','general','physics'];cats=['passed','physics_failure','other_failure','timeout','execution_error']
rows=[]
for line in (S/'outcome-breakdown.md').read_text().splitlines():
 if line.startswith('| gpt-'):
  v=[s.strip() for s in line.strip('|').split('|')]
  if len(v)!=9:continue
  row={'model':v[0],'effort':v[1],'condition':{'No hint':'baseline','General':'general','Physics':'physics'}[v[2]],**dict(zip(cats,map(int,v[3:8]))),'attempts':int(v[8])}
  assert sum(row[k] for k in cats)==row['attempts']==39
  rows.append(row)
assert len(rows)==30 and len({(r['model'],r['effort'],r['condition']) for r in rows})==30
baselines=list(csv.DictReader((S/'baseline-luna-trials.csv').open()))+list(csv.DictReader((S/'baseline-sol-terra-trials.csv').open()))
hints=list(csv.DictReader((S/'hint-matrix-summary.csv').open()))
all_unique=list(csv.DictReader((R/'deduplicated-trials.csv').open()));ids={(r['model'],r['effort'],r['condition'],r['trial']) for r in all_unique}
matched_ids=[]
for r in rows:
 if r['condition']=='baseline':
  b=[b for b in baselines if b['model']==r['model'] and b['reasoning_effort']==r['effort']]
  assert len(b)==39 and len({b['trial'] for b in b})==39 and len({b['task'] for b in b})==13
  assert sum(b['passed']=='True' for b in b)==r['passed']
  assert sum(bool(b['exception_type']) for b in b)==r['timeout']+r['execution_error']
  for b in b:
   key=(b['model'],b['reasoning_effort'],'baseline',b['trial']);assert key in ids;matched_ids.append(key)
 else:
  h=[h for h in hints if h['model']==r['model'] and h['reasoning_effort']==r['effort'] and h['hint']=={'general':'general','physics':'physical'}[r['condition']]]
  assert len(h)==1;h=h[0]
  assert int(h['finished_trials'])==39 and int(h['passes'])==r['passed']
  assert int(h['graded_failures'])==r['physics_failure']+r['other_failure']
  assert int(h['timeouts'])==r['timeout'] and int(h['other_errors'])==r['execution_error']
assert len(matched_ids)==len(set(matched_ids))==390
expected={'baseline':[84,296,8,1,1],'general':[183,158,34,14,1],'physics':[257,66,57,10,0]}
totals=[]
for c in C:
 t={'condition':c,**{k:sum(r[k] for r in rows if r['condition']==c) for k in cats},'attempts':390};assert [t[k] for k in cats]==expected[c];totals.append(t)
full=json.loads((R/'summary.json').read_text());assert full['metadata']['attempts']==2068
insights={'luna_high_physics_failure_counts':[next(r['physics_failure'] for r in rows if r['model']=='gpt-5.6-luna' and r['effort']=='high' and r['condition']==c) for c in C], 'physics_hint_remaining_luna_none_low_medium':sum(r['physics_failure'] for r in rows if r['model']=='gpt-5.6-luna' and r['effort'] in ['none','low','medium'] and r['condition']=='physics'),'all_physics_hint_physics_failures':66,'all_reviewed_graded_failures':619,'all_strict_physics_failures':520,'calibration_only_failure_count':604,'reviewed_nonphysics_primaries_with_calibration_success':84,'physics_failure_rate_by_condition':[t['physics_failure']/390 for t in totals]}
assert insights['luna_high_physics_failure_counts']==[33,12,2] and insights['physics_hint_remaining_luna_none_low_medium']==56
manifest=json.loads((R/'outcome-source-manifest.json').read_text())
for e in manifest:
 if e['file']!='sources/outcome-breakdown.md' and not e['file'].startswith('sources/hint-matrix-'):e['file']='sources/hint-matrix-'+Path(e['file']).name
 assert hashlib.sha256((R/e['file']).read_bytes()).hexdigest()==e['sha256']
(R/'outcome-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
report={'source_url':manifest[0]['url'],'scope':'Thirty 39-attempt cells across ten matched GPT-5.6 model/effort settings. Classification update to existing outcomes, not new attempts.','rows':rows,'totals':totals,'insights':insights,'overlap_audit':{'baseline_trials_matched_by_identity':390,'hinted_attempts_reconciled_to_unchanged_published_summary':780,'new_trials_added':0,'full_study_attempts_unchanged':2068,'baseline_identity_keys':matched_ids,'hinted_trial_ids_available':False},'caveats':['Published classification report, not an independent local raw-trajectory review. Underlying classification ledger remains unpublished; reported SHA-256 b9faeffe2d4db9c983eab440f52b0eb19cf9222ead0e142fd0c9dfb2a2e5faca.','Every recorded matrix attempt receives one category. Execution exceptions take precedence over grades. Five replaced pre-agent setup records stay outside the original 1170-attempt matrix denominator.','Physics failure requires calibration and parameter success, hidden-prediction failure, and a reviewed primary physical-model mistake. A calibration fit alone does not establish a physics failure.','Historical controls and runtime/CPU differences limit causal inference. These aggregate changes do not prove individual failures converted to passes.','No classification is imputed to the other 898 study attempts. Existing broader prediction-only or primary-cause categories are not pooled into this stricter outcome definition.']}
(R/'failure-breakdown.json').write_text(json.dumps(report,indent=2)+'\n')
with (R/'failure-breakdown.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'rows':len(rows),'attempts_classified':1170,'full_study_attempts':2068,'new_trials_added':0,'totals':totals,'insights':insights},indent=2))
