"""Independent equilibrium linear-response certificate for the proposed flux model."""
from pathlib import Path
import importlib.util,json,itertools
import numpy as np
s=importlib.util.spec_from_file_location('candidate',Path(__file__).resolve().parents[1]/'rotating-reservoir-r8-brownian/recirculating.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rows=[]
for temperature,lam,gamma,eta in itertools.product([.4,1.2],[.2,3.],[.4,1.1],[.5,1.]):
 e=dict(temperature_a=temperature,temperature_b=temperature,angular_speed=0.,flip_rate=1.2,flux_fraction=eta,relaxation_rate=lam)
 ref=m.hermite(e,gamma,degree=34,variance=2.,noise=True);dt=1e-4
 def current(delta):return m.hermite(dict(e,temperature_a=temperature+delta),gamma,degree=34,variance=2.)['mean']
 derivative=(current(dt)-current(-dt))/(2*dt);noise=ref['noise'];fluctuation=2*temperature**2*derivative;error=abs(noise-fluctuation)/noise
 assert error<2e-6,(e,error)
 rows.append(dict(input=e,gamma=gamma,noise=noise,linear_thermal_conductance=derivative,fdt_noise=fluctuation,relative_defect=error))
Path(__file__).with_suffix('.json').write_text(json.dumps(dict(rows=rows,max_relative_defect=max(r['relative_defect'] for r in rows)),indent=2)+'\n');print('fullflux/mixedflux FDT',len(rows),'cases',max(r['relative_defect'] for r in rows))
