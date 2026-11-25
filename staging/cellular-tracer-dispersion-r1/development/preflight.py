"""Author timing and initial scored-case reference check, retained unchanged."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import importlib.util,json,time
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1];TASK=BASE/'tasks/cellular-tracer-dispersion'
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
O=load('o',TASK/'solution/model.py');S=load('s',BASE/'scripts/cellular_tracer_dispersion_baseline.py');R=load('r',TASK/'tests/reference.py')
out=[];start=time.monotonic()
for name,inputs in R.hidden_inputs().items():
 t=time.monotonic();physical=O.predict_at(inputs,R.TRUE_PARAMETER);source=S.predict_at(inputs,R.TRUE_PARAMETER);ref=R.predict(inputs,R.TRUE_PARAMETER)
 out.append({'name':name,'inputs':inputs,'oracle':physical.tolist(),'source':source.tolist(),'reference':ref.tolist(),'source_nrmse':float(np.linalg.norm(source-ref)/np.linalg.norm(ref)),'max_reference_error':float(abs(ref-physical).max()),'seconds':time.monotonic()-t})
settings=R.calibration_inputs();t=time.monotonic();calref=R.predict(settings,R.TRUE_PARAMETER);cal=O.predict_at(settings,R.TRUE_PARAMETER)
d={'groups':out,'calibration_reference_bias_sigma':float(abs(cal-calref).max()/.003),'calibration_seconds':time.monotonic()-t,'seconds':time.monotonic()-start}
(BASE/'results/development-preflight.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
