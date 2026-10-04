import importlib.util,json,hashlib,tempfile,shutil,subprocess,sys,os
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parents[1];t=r/'tasks/thermal-bodies'
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v=load(r/'scripts/validate_thermal_colored.py','v');o=v.oracle;b=v.shortcut
corners=[]
for g in (.3,1.2):
 for k1,k2 in ((.8,.8),(.8,1.4),(1.4,.8),(1.4,1.4)):
  for c in (.05,1.2):
   for tau in (.15,2):
    for T in ((.5,2),(2,.5)):
     A,S,_=o.stationary_state(g,k1,k2,c,tau,*T);modes=b.stationary_modes(g,k1,k2,c,tau,*T)
     j=np.array([S[2,4],S[3,5]]);jb=np.sum([[C[1,2],C[1,3]] for _,C in modes],axis=0)
     es=[dict(springs=[k1,k2],coupling=c,memory=tau,temperatures=list(T),readout='mode_correlation',mode=mode,lag=lag) for mode in (0,1) for lag in (0,3,6)]
     corners.append(dict(calibration_error=float(np.max(abs(o.predict_at(es,g)-b.predict_at(es,g)))),minimum_covariance=float(min(np.min(np.linalg.eigvalsh(S)),*(np.min(np.linalg.eigvalsh(C)) for _,C in modes))),first_law=float(max(abs(sum(j)),abs(sum(jb)))),entropy=float(min(-sum(j/np.array(T)),-sum(jb/np.array(T))))))
inputs=v.calibration_inputs();recovery=[]
for g in (.3,.41,.7,.93,1.2):
 clean=v.reference.predict(inputs,g);rec=[dict(input=e,value=float(y),sigma=.003) for e,y in zip(inputs,clean)];fit=b.Model().fit(rec).friction;recovery.append(dict(true=g,fitted=fit,error=abs(g-fit)))
controls={}
for name,source in [('oracle',t/'solution/model.py'),('shortcut',r/'scripts/thermal_colored_baseline.py')]:
 with tempfile.TemporaryDirectory(prefix='thermal-r11-'+name+'-') as directory:
  root=Path(directory);app=root/'app';tests=root/'tests';shutil.copytree(t/'environment',app);shutil.copytree(t/'tests',tests);shutil.copy2(source,app/'model.py')
  env=dict(os.environ,PYTHONPATH=str(app),OPENBLAS_NUM_THREADS='1');p=subprocess.run([sys.executable,'-m','pytest','-q',str(app/'test_public.py'),str(tests/'test_hidden.py')],env=env,cwd=app,capture_output=True,text=True);controls[name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
assert controls['oracle']['returncode']==0 and controls['shortcut']['returncode']==1
assert max(x['calibration_error'] for x in corners)<1e-10
assert min(x['minimum_covariance'] for x in corners)>0
assert max(x['first_law'] for x in corners)<1e-10
assert min(x['entropy'] for x in corners)>-1e-10
report={'corners':len(corners),'max_corner_calibration_error':max(x['calibration_error'] for x in corners),'min_corner_covariance_eigenvalue':min(x['minimum_covariance'] for x in corners),'max_corner_first_law_error':max(x['first_law'] for x in corners),'min_corner_entropy_production':min(x['entropy'] for x in corners),'noiseless_parameter_recovery':recovery,'local_pytest':controls}
(r/'results/thermal-r11-local-controls.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
