"""Separate finite-ramp and ensemble-grid errors; keep the first report unchanged."""
import json,time
from pathlib import Path
import numpy as np
import check,screen

HERE=Path(__file__).resolve().parent
start=time.perf_counter();runs=[]
p=dict(j0=.1437,delta_final=.2,slope=.1,s_final=2.2)
for configuration in [dict(duration=256.,step=.04,action_order=32,phase_order=256),
                      dict(duration=1024.,step=.04,action_order=32,phase_order=256),
                      dict(duration=512.,step=.04,action_order=48,phase_order=384)]:
    runs.append(check.save_run(p,configuration,'separate_duration_and_ensemble_refinement'))
for params in [dict(j0=.12,delta_final=.25,slope=.12,s_final=1.8),
               dict(j0=.16,delta_final=.25,slope=.12,s_final=2.2)]:
    runs.append(check.save_run(params,dict(duration=1024.,step=.04,action_order=24,phase_order=256),'edge_slow_refinement'))
zero_limits=[]
for slope in [0.,1e-5,1e-4,.001]:
    value=screen.predict(.1437,.2,slope,2.2)
    zero_limits.append({'slope':slope,**{k:v for k,v in value.items() if k!='details'}})
representative=screen.predict(.1437,.2,.1,2.2)
fractions=[]
for s in [.05,representative['min_crossing'],representative['max_crossing'],2.2]:
    delta=screen.tilt(s,.2,.1,2.2)
    areas=screen.old.geometry(float(delta))[2]
    growth=screen.growth(s,.2,.1,2.2)
    fractions.append({'s':s,'delta':delta,'left_area_fraction':float(areas[0]/areas.sum()),
                      'left_growth_fraction':float(growth[0]/growth.sum()),'lobe_growth':growth.tolist()})
cal=[]
for j in np.linspace(.12,.16,17):
    h=1e-5
    derivative=(screen.predict(j+h,.2,0,1.8)['physical']-screen.predict(j-h,.2,0,1.8)['physical'])/(2*h)
    cal.append({'j0':float(j),'response':screen.predict(j,.2,0,1.8)['physical'],'derivative':derivative})
report={'trajectory_runs':runs,'zero_slope_limits':zero_limits,
        'representative_changing_fractions':fractions,'calibration_derivative_checks':cal,
        'sampled_calibration_derivative_min':min(row['derivative'] for row in cal),
        'seconds':time.perf_counter()-start,'model_evaluations':0}
(HERE/'refinement-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='trajectory_runs'},indent=2))
