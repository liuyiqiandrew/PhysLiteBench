"""Kernel-only posthoc repair, with fitted viscosity fixed and artifacts intact."""
from pathlib import Path
import json,hashlib,types,importlib.util,ast
import numpy as np
R=Path(__file__).resolve().parents[2]
J=R/'jobs/finite-layer-phoresis-zero-three-phoresis-r1-plain-20261004-071333'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
TASK=R/'staging/finite-layer-phoresis-r1/tasks/finite-layer-phoresis'
spec=importlib.util.spec_from_file_location('physical_reference',TASK/'tests/reference.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
records=json.loads((TASK/'tests/data/calibration.json').read_text())
calinputs=[r['input'] for r in records];values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
def module(text,name):
 m=types.ModuleType(name);exec(compile(text,name,'exec'),m.__dict__);return m
needle='weight = 1.5*(r-radius)**2'
replacement='weight = r*r - 1.5*radius*r + 0.5*radius**3/r'
report={'task':'finite-layer-phoresis','revision':1,'diagnostic_only':True,'new_agent_evaluations':0,'original_artifacts_modified':False,
'change':{'old':needle,'new':replacement,'explanation':'Replace the planar near-wall hydrodynamic weight with the exact curved, force-free reciprocal-theorem weight. Leave the full curved solute BVP, potential, quadrature, fit, viscosity and output conventions unchanged.'},
'evidence':{str(p.relative_to(R)):sha(p) for p in [Path(__file__),TASK/'tests/reference.py']},'trials':{}}
for t in sorted(J.glob('finite-layer-phoresis__*')):
 if not (t/'result.json').exists():continue
 result=json.loads((t/'result.json').read_text())
 if result['verifier_result']['rewards']['reward']!=0:continue
 p=t/'artifacts/app/model.py';beforehash=sha(p);text=p.read_text();assert text.count(needle)==1
 old=module(text,'original_'+t.name);new=module(text.replace(needle,replacement),'repaired_'+t.name)
 metrics=json.loads((t/'verifier/metrics.json').read_text());eta=metrics['parameters']['viscosity']
 row={'source':str(p.relative_to(R)),'source_sha256':beforehash,'fixed_parameter':{'viscosity':eta},'calibration_chi2':{},'hidden':{}}
 for name,m in [('original',old),('kernel_repair',new)]:
  pred=m.predict_at(calinputs,eta);row['calibration_chi2'][name]=float(np.sum(((pred-values)/sigma)**2)/(len(records)-1))
 for name,experiments in ref.hidden_inputs().items():
  truth=ref.predict(experiments,ref.TRUE_PARAMETER)
  row['hidden'][name]={label:float(np.sqrt(np.mean((m.predict_at(experiments,eta)-truth)**2)/np.mean(truth**2))) for label,m in [('original',old),('kernel_repair',new)]}
 row['repaired_source_sha256']=hashlib.sha256(text.replace(needle,replacement).encode()).hexdigest()
 row['all_original_hidden_groups_pass_after_repair']=all(x['kernel_repair']<.04 for x in row['hidden'].values())
 assert beforehash==sha(p);report['trials'][t.name]=row
out=R/'results/zero-three-phoresis-r1-causal-diagnostic.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
