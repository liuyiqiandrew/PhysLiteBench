from pathlib import Path
from datetime import datetime,timezone
import json,os,signal,subprocess,time
R=Path('/Users/yp9846/Desktop/PhysLiteBench');C=R/'staging/overnight-20261011/comparison';ledger_path=Path('/private/tmp/physlite-prime-session/ledger.json')
runs=[json.loads(p.read_text()) for p in (C/'runs').glob('deepseek--*/run.json')]
assert len(runs)==10 and all(r['trials']==3 and r['status']!='running' for r in runs),'Wait for all planned trials before closing relay'
expected='/private/tmp/physlitebench-inspection-uv/archive-v0/9GmqprcUj8krKCH4/bin/python /private/tmp/prime_budget_proxy.py'
processes=subprocess.run(['ps','-axo','pid=,command='],capture_output=True,text=True,check=True).stdout
pids=[int(line.strip().split(None,1)[0]) for line in processes.splitlines() if len(line.strip().split(None,1))==2 and line.strip().split(None,1)[1]==expected]
assert len(pids)==1,'Expected exactly one active local relay; inspect before altering ledger'
for pid in pids:
 os.kill(pid,signal.SIGINT)
 for _ in range(20):
  try:os.kill(pid,0)
  except ProcessLookupError:break
  time.sleep(.25)
 else:
  os.kill(pid,signal.SIGTERM);time.sleep(.5)
now=datetime.now(timezone.utc).isoformat();d=json.loads(ledger_path.read_text());pending=[]
for r in d['requests']:
 if r['status']=='pending':
  pending.append(r['id']);r.update(status='closed_without_final_usage',finished_at=now,error='Local relay closed after all scientific trials completed. Final provider usage unavailable; full reservation retained.')
d['shutdown']=dict(at=now,scientific_attempts_completed=30,unresolved_pending_requests=len(pending),note='No further requests can be launched through this relay. Unresolved usage retains the full conservative reservation.')
tmp=ledger_path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(ledger_path)
print(json.dumps(dict(relay_processes_stopped=len(pids),newly_closed_unknown_usage=len(pending),estimated_token_cost_usd=sum(r.get('estimated_token_cost_usd',0) for r in d['requests']),conservative_accounted_usd=sum(r['accounted_usd'] for r in d['requests']))))
