from pathlib import Path
from datetime import datetime,timezone
import json
R=Path('/Users/yp9846/Desktop/PhysLiteBench');S=R/'staging/overnight-20261011';C=S/'comparison'
models=['gpt-5.6-terra','deepseek/deepseek-v4.1-flash'];status={m:dict(completed=0,passes=0,errors=0,exception_types={},active_trials=0,batches=0) for m in models};active=[]
for p in (C/'runs').glob('*/run.json'):
 d=json.loads(p.read_text())
 if d['model'] not in models:continue
 s=status[d['model']];s['batches']+=1
 excluded={x['trial'] for x in d.get('excluded_pre_model_trials',[])}
 trials=[json.loads(q.read_text()) for job in [d['job'],*d.get('additional_jobs',[])] for q in (R/'jobs'/job).glob('*/result.json') if q.parent.name not in excluded]
 for t in trials:
  s['completed']+=1;s['passes']+=int(t.get('exception_info') is None and (t.get('verifier_result') or {}).get('rewards',{}).get('reward')==1);s['errors']+=int(t.get('exception_info') is not None)
  if t.get('exception_info'):
   kind=t['exception_info']['exception_type'];s['exception_types'][kind]=s['exception_types'].get(kind,0)+1
 if d['status']=='running':
  pending=max(0,d['trials_requested']-len(trials));s['active_trials']+=pending;active.append(dict(model=d['model'],task=d['task'],job=d['job'],pending=pending))
ledger=json.loads(Path('/private/tmp/physlite-prime-session/ledger.json').read_text())
spend=dict(cap_usd=ledger['cap_usd'],conservative_accounted_usd=sum(x['accounted_usd'] for x in ledger['requests']),completed_token_cost_usd=sum(x.get('estimated_token_cost_usd',0) for x in ledger['requests']),pending_requests=sum(x['status']=='pending' for x in ledger['requests']),request_count=len(ledger['requests']))
print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),models=status,active=active,remaining_active_trials=sum(x['active_trials'] for x in status.values()),prime=spend)))
