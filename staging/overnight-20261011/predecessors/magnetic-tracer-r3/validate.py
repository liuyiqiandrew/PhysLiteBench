"""Validate the retained-contact magnetic tracer without regenerating data."""
from pathlib import Path
import importlib.util
import hashlib
import json
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/magnetic-tracer'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    started=time.monotonic()
    good=load('oracle',TASK/'solution/model.py');bad=load('shortcut',ROOT/'scripts/magnetic_tracer_baseline.py');ref=load('physical_reference',TASK/'tests/reference.py')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    inputs=[r['input'] for r in records];observed=np.array([r['value'] for r in records]);sigmas=np.array([r['sigma'] for r in records]);hidden=ref.hidden_inputs()
    truth={n:ref.predict(es) for n,es in hidden.items()}
    def errors(module,gamma):return {n:float(np.linalg.norm(module.predict_at(es,gamma)-truth[n])/np.linalg.norm(truth[n])) for n,es in hidden.items()}
    controls={}
    for name,module in [('oracle',good),('shortcut',bad)]:
        model=module.Model().fit(records);residual=(model.predict(inputs)-observed)/sigmas
        controls[name]=dict(friction=model.friction,calibration_chi2=float(residual@residual)/(len(records)-1),parameter_error=abs(model.friction/.8-1),groups=errors(module,model.friction))
        assert controls[name]['calibration_chi2']<1.5 and controls[name]['parameter_error']<.03
    assert max(controls['oracle']['groups'].values())<.025
    sensitive=[n for n in hidden if n!='memory_free_anchor']
    assert min(controls['shortcut']['groups'][n] for n in sensitive)>.025
    assert controls['shortcut']['groups']['memory_free_anchor']<.025
    print('Controls',json.dumps(controls),flush=True)
    rng=np.random.default_rng(20261011);independent=[];refined=[];symmetry=[];eigenvalues=[]
    for i in range(64):
        gamma=float(rng.uniform(.5,1.5));angle=rng.uniform(-np.pi,np.pi)
        rotate=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        initial=rotate@np.diag(rng.uniform(.2,2,2))@rotate.T
        e=ref.experiment(rng.uniform(.1,4),field=rng.uniform(-2,2),temperature=rng.uniform(.5,1.2),stiffness=rng.uniform(.7,1.8,2),initial=initial,memory_drag=rng.uniform(.1,1.3),memory_time=rng.uniform(.2,1.6),memory_temperature=rng.uniform(.4,1.4))
        es=[dict(e,readout=r) for r in ['xx','xy','yy','area']]
        a=good.predict_at(es,gamma);b=ref.predict(es,gamma);c=ref.predict(es,gamma,mass_scale=.0005)
        independent.append(float(max(abs(a-b))));refined.append(float(max(abs(a-c))))
        covariance=np.array([[a[0],a[1]],[a[1],a[2]]]);eigenvalues.append(float(np.linalg.eigvalsh(covariance).min()))
        mirror=initial.copy();mirror[0,1]*=-1;mirror[1,0]*=-1
        reversed_experiments=[dict(x,field=-x['field'],initial_covariance=mirror) for x in es]
        symmetry.append(float(max(abs(good.predict_at(reversed_experiments,gamma)-a*[1,-1,1,-1]))))
    assert max(independent)<1e-6 and max(refined)<1e-6
    assert min(eigenvalues)>0 and max(symmetry)<1e-11
    equilibrium=[]
    for gamma in [.5,.8,1.5]:
        for field in [-2.,0.,2.]:
            e=ref.experiment(4.,field=field,temperature=1.1,stiffness=(.7,1.6),initial=np.diag([1.1/.7,1.1/1.6]),memory_drag=1.2,memory_time=.4,memory_temperature=1.1)
            equilibrium.append(abs(float(good.predict_at([e],gamma)[0])))
    assert max(equilibrium)<1e-11
    clean=ref.predict(inputs)
    calibration_equivalence=float(max(abs(good.zero_field_calibration(inputs,.8)-clean)))
    assert calibration_equivalence<1e-8
    draws=np.random.default_rng(905472).normal(size=(256,len(records)))
    rows=[]
    for i,noise in enumerate(draws):
        values=clean+sigmas*noise;data=[dict(r,value=float(v)) for r,v in zip(records,values)]
        fit=good.Model().fit(data);residual=(good.zero_field_calibration(inputs,fit.friction)-values)/sigmas
        oe=errors(good,fit.friction);be=errors(bad,fit.friction)
        row=dict(parameter=fit.friction,calibration_chi2=float(residual@residual)/(len(records)-1),oracle_max=max(oe.values()),shortcut_min=min(be[n] for n in sensitive),anchor=be['memory_free_anchor'])
        assert row['calibration_chi2']<1.5 and abs(fit.friction/.8-1)<.03
        assert row['oracle_max']<.025 and row['shortcut_min']>.025 and row['anchor']<.025
        rows.append(row)
    report=dict(revision=3,controls=controls,physics=dict(random_cases=64,finite_mass_error_max=max(independent),refined_error_max=max(refined),reflection_error_max=max(symmetry),minimum_position_covariance=min(eigenvalues),equilibrium_area_max=max(equilibrium),old_calibration_equivalence=calibration_equivalence),noise=dict(seed=905472,trials=256,oracle_worst=max(r['oracle_max'] for r in rows),shortcut_smallest_miss=min(r['shortcut_min'] for r in rows),max_calibration_chi2=max(r['calibration_chi2'] for r in rows)),seconds=time.monotonic()-started,source_sha256={str(p.relative_to(TASK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    (ROOT/'results/magnetic-tracer-r3-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','controls']},indent=2),flush=True)

if __name__=='__main__':main()
