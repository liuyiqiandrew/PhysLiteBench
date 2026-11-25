"""Validate dipole angular-momentum balance and both completed controls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/optical-torque'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run(generate=False,noise_trials=256):
    started=time.time()
    good=load(TASK/'solution/model.py','oracle')
    bad=load(ROOT/'scripts/optical_torque_baseline.py','shortcut')
    ref=load(TASK/'tests/reference.py','reference')
    inputs=ref.calibration_inputs();true=ref.TRUE_PARAMETER;sigma=5e-5;limit=.04
    exact=ref.predict(inputs,true)
    def records(values):
        return [dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
    if generate:
        data=json.dumps(records(exact+np.random.default_rng(949101).normal(0,sigma,len(inputs))),indent=2)+'\n'
        for name in ['environment/data/calibration.json','tests/data/calibration.json']:(TASK/name).write_text(data)
    data=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert data==json.loads((TASK/'tests/data/calibration.json').read_text())
    groups=ref.hidden_inputs();truth={name:ref.predict(es,true) for name,es in groups.items()}
    def errors(module,strength):
        return {name:float(np.linalg.norm(module.predict_at(es,strength)-truth[name])/np.linalg.norm(truth[name])) for name,es in groups.items()}
    controls={}
    for label,module in [('oracle',good),('shortcut',bad)]:
        model=module.Model().fit(data)
        residual=(model.predict(inputs)-np.array([r['value'] for r in data]))/sigma
        controls[label]=dict(response_strength=model.response_strength,calibration_chi2=float(residual@residual)/(len(inputs)-1),hidden=errors(module,model.response_strength))
        assert abs(model.response_strength/true-1)<.03 and controls[label]['calibration_chi2']<1.5
        assert (max(controls[label]['hidden'].values())<limit if label=='oracle' else min(controls[label]['hidden'].values())>limit)
    checks=dict(calibration_equivalence=float(np.max(abs(good.predict_at(inputs,true)-bad.predict_at(inputs,true)))),calibration_reference=float(np.max(abs(good.predict_at(inputs,true)-exact))),stress_torque=0.,stress_force=0.,self_torque=0.,stress_radius=0.,stress_refinement=0.,tensor_passivity=0.,energy_balance=0.,calibration_torque=0.,rotation_covariance=0.,circular_identity=0.)
    cases=[e for es in groups.values() for e in es]
    for R in [np.eye(3),Rotation.from_euler('xyz',[.3,-.7,1.2]).as_matrix()]:
        for phase in [0.,.8,np.pi/2]:
            for pair in [(0,1),(0,2),(1,2)]:
                i,j=pair;field=.6*R[:,i]+.8*np.exp(1j*phase)*R[:,j];direction=np.cross(R[:,i],R[:,j])
                cases.append(ref.experiment(R,direction,field,direction,'torque'))
    for strength in [.6,1.1,1.6]:
        for e in cases:
            R=np.array(e['orientation']);field=np.array(e['field_real'])+1j*np.array(e['field_imag']);axis=np.array(e['axis']);alpha=good.polarizability(strength,R);p=alpha@field
            force,torque,self_torque=ref.stress_vectors(e,strength)
            f2,t2,_=ref.stress_vectors(e,strength,.9,28);f3,t3,_=ref.stress_vectors(e,strength,.4,40)
            checks['stress_torque']=max(checks['stress_torque'],abs(good.response(e,strength)-axis@torque))
            checks['stress_force']=max(checks['stress_force'],abs(good.response(dict(e,observable='force'),strength)-axis@force))
            checks['self_torque']=max(checks['self_torque'],float(np.max(abs(self_torque+np.imag(np.cross(p.conj(),p))/(12*np.pi)))))
            checks['stress_radius']=max(checks['stress_radius'],float(np.max(abs(torque-t2))),float(np.max(abs(force-f2))))
            checks['stress_refinement']=max(checks['stress_refinement'],float(np.max(abs(torque-t3))),float(np.max(abs(force-f3))))
            checks['tensor_passivity']=max(checks['tensor_passivity'],float(np.max(abs((alpha-alpha.conj().T)/(2j)-alpha.conj().T@alpha/(6*np.pi)))))
            checks['energy_balance']=max(checks['energy_balance'],abs(.5*np.imag(np.vdot(field,p))-np.vdot(p,p).real/(12*np.pi)))
            rotation=Rotation.from_euler('xyz',[.2,.1,-.4]).as_matrix()
            changed=ref.experiment(rotation@R,rotation@np.array(e['direction']),rotation@field,rotation@axis,'torque')
            checks['rotation_covariance']=max(checks['rotation_covariance'],abs(good.response(e,strength)-good.response(changed,strength)),abs(bad.response(e,strength)-bad.response(changed,strength)))
        for e in inputs[:36]:
            z=dict(e,observable='torque')
            checks['calibration_torque']=max(checks['calibration_torque'],abs(good.response(z,strength)),abs(bad.response(z,strength)),abs(ref.predict([z],strength)[0]))
        alpha=np.diag(good.polarizability(strength,np.eye(3)))
        e=ref.experiment(np.eye(3),[0,0,1],np.array([1,1j,0])/np.sqrt(2),[0,0,1],'torque')
        checks['circular_identity']=max(checks['circular_identity'],abs(good.response(e,strength)-abs(alpha[0]-alpha[1])**2/(24*np.pi)))
    assert max(checks.values())<1e-11
    recovery={}
    for strength in [.6,.85,1.1,1.35,1.6]:
        fitted=good.Model().fit(records(ref.predict(inputs,strength))).response_strength
        recovery[str(strength)]=abs(fitted/strength-1)
    assert max(recovery.values())<1e-6
    profile=np.array([np.linalg.norm(good.predict_at(inputs,strength)-exact)**2 for strength in np.linspace(.6,1.6,101)])
    at=int(np.argmin(profile));assert np.all(np.diff(profile[:at+1])<0) and np.all(np.diff(profile[at:])>0)
    rng=np.random.default_rng(949103);noise=[]
    for _ in range(noise_trials):
        sample=records(exact+rng.normal(0,sigma,len(inputs)));model=bad.Model().fit(sample)
        residual=(model.predict(inputs)-np.array([r['value'] for r in sample]))/sigma
        noise.append(dict(response_strength=model.response_strength,chi2=float(residual@residual)/(len(inputs)-1),oracle=errors(good,model.response_strength),shortcut=errors(bad,model.response_strength)))
    summary=dict(calibration_passes=sum(n['chi2']<1.5 for n in noise),parameter_passes=sum(abs(n['response_strength']/true-1)<.03 for n in noise),oracle_passes=sum(max(n['oracle'].values())<limit for n in noise),shortcut_passes=sum(max(n['shortcut'].values())<limit for n in noise),maximum_oracle_error=max(max(n['oracle'].values()) for n in noise),minimum_shortcut_error=min(min(n['shortcut'].values()) for n in noise),maximum_chi2=max(n['chi2'] for n in noise),parameter_range=[min(n['response_strength'] for n in noise),max(n['response_strength'] for n in noise)])
    assert summary['calibration_passes']==noise_trials and summary['parameter_passes']==noise_trials and summary['oracle_passes']==noise_trials and summary['shortcut_passes']==0
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[ROOT/'scripts/optical_torque_baseline.py',Path(__file__).resolve()]
    return dict(revision=1,noise_trials=noise_trials,calibration_seed=949101,noise_seed=949103,measurement_sigma=sigma,controls=controls,physical_checks=checks,noiseless_parameter_recovery=recovery,calibration_profile_unimodal=True,minimum_hidden_absolute_torque=min(float(np.min(abs(y))) for y in truth.values()),noise=summary,noise_realizations=noise,seconds=time.time()-started,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args()
    report=run(args.generate,args.noise_trials);(ROOT/'results/optical-torque-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','noise_realizations']},indent=2))
