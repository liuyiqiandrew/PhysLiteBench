"""Correct the mislabeled .45 limit and test a genuinely split action band."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import prototype as p
HERE=Path(__file__).resolve().parent
old=json.loads((HERE/'history/initial-domain-run/report.json').read_text())
start=time.perf_counter()
unchanged=[]
for x in old['static_grid']:
    r=p.predictions(x['j0'],x['delta'],x['s_final'])
    unchanged.append(max(abs(r[k]-x[k]) for k in ['physical','source','capture_fraction']))
assert max(unchanged)<1e-10
j0=.14;delta=.2
sf=(2*np.pi*(1.1*j0)/p.geometry(delta)[2].sum())**(2/3)
static=[p.predictions(j0,delta,sf,action_order=n) for n in [32,64,128]]
assert all(abs(x['capture_fraction']-.5)<1e-12 for x in static)
rows=[]
for duration,na,nph,step in [(1024.,32,256,.04),(2048.,32,256,.04),(2048.,64,512,.02)]:
    row=p.trajectory(j0=j0,delta=delta,s_final=sf,duration=duration,action_order=na,phase_order=nph,step=step)
    row['limiting_prediction']=static[-1];row['absolute_mean_energy_error']=abs(row['mean_energy']-static[-1]['physical']);row['relative_mean_energy_error']=abs(row['mean_energy']/static[-1]['physical']-1)
    rows.append(row)
    with (HERE/'partial-runs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(row,allow_nan=False),flush=True)
assert max(abs(static[i]['physical']-static[-1]['physical']) for i in [0,1])<2e-7
assert max(x['absolute_mean_energy_error'] for x in rows)<1e-4
report={'status':'partial_capture_addition_passed','change':'The .45 endpoint in the first validator was mislabeled partial capture but has capture fraction zero. It remains preserved as a noncrossing check. A new endpoint targets precisely half of the initial action band. The oracle now splits its action integral at the capture threshold.','s_final':sf,'static_refinements':static,'trajectory_runs':rows,'maximum_old_fully_captured_prediction_change':max(unchanged),'seconds':time.perf_counter()-start,'qualifications':['This near-barrier mean energy is small and is a limit check, not a proposed discriminating scored signal.','The exact limiting captured fraction is .5. At finite ramp duration the separatrix boundary has a finite-width dynamical correction; total-energy convergence, not exact finite-duration capture fraction, is the primary observable.']}
(HERE/'partial-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
# Final report retains every original trajectory and labels the old endpoint accurately.
final=old.copy();final['near_threshold_noncrossing_action_quadrature']=final.pop('partial_action_quadrature')
final['partial_capture_addendum']={'path':'partial-report.json','sha256':hashlib.sha256((HERE/'partial-report.json').read_bytes()).hexdigest()}
final['development_history']='Original code, complete successful report and logs are under history/initial-domain-run. The .45 partial-capture label was corrected; only action integration for partially captured bands was changed. All 27 previous full-capture predictions remain unchanged to the reported tolerance, and no previous trajectory result was discarded or rerun.'
final['source_sha256']={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['prototype.py','validate.py','partial_capture.py','exploration.jsonl','validation-runs.jsonl','partial-runs.jsonl','partial-report.json']}
(HERE/'report.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
print('SUMMARY',json.dumps({'maximum_old_change':max(unchanged),'partial_absolute_errors':[x['absolute_mean_energy_error'] for x in rows],'partial_relative_errors':[x['relative_mean_energy_error'] for x in rows],'runtime':report['seconds']}),flush=True)
