from pathlib import Path
from datetime import datetime,timezone
import json,re,subprocess
R=Path('/Users/yp9846/Desktop/PhysLiteBench');C=R/'staging/overnight-20261011/comparison';p=C/'results.json';d=json.loads(p.read_text());rows=[]
for t in d['trials']:
 if not t['development_selected'] and t.get('agent_execution_seconds',0)>660:
  rows.append(dict(model=t['model'],task=t['task'],trial=t['trial'],elapsed_seconds=t['agent_execution_seconds'],reward=t['reward'],exception=t['exception']))
log=subprocess.run(['pmset','-g','log'],capture_output=True,text=True,check=True).stdout;events=[]
for line in log.splitlines():
 m=re.match(r'^(2026-10-11 10:(?:0[0-9]|[12][0-9]|3[0-9]):[0-9]{2} -0400) (Sleep|Wake|DarkWake)\s+\t',line)
 if m:events.append(dict(local_time=m[1],event=m[2]))
audit=dict(created_at=datetime.now(timezone.utc).isoformat(),configured_agent_timeout_seconds=600,outlier_threshold_seconds=660,outliers=rows,host_power_events=events,interpretation='Four Terra attempts have 2029–2423 seconds of timestamp elapsed time. macOS recorded repeated sleep/dark-wake events during those attempts, ending in full wake at 10:34 local time. Inference: host suspension explains much of the discrepancy, but exact active inference time is not reconstructible here. All four completed with reward zero. Retain them without resampling and disclose the timing limitation; the comparison is not evidence of a uniformly enforced 600-second wall-clock budget.')
for s in d['summary']:s['timing_outliers']=sum(r['model']==s['model'] for r in rows)
d['timing_audit']=audit;p.write_text(json.dumps(d,indent=2)+'\n');(C/'timing-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'outliers':len(rows),'models':sorted({r['model'] for r in rows}),'host_power_events':len(events)}))
