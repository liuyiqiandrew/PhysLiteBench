"""Scientific controls and noise sensitivity for the emitter lifetime task."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/emitter-lifetime'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def run(generate=False,noise_trials=256):
    start=time.time();g=load(TASK/'solution/model.py','oracle');b=load(ROOT/'scripts/emitter_lifetime_baseline.py','shortcut');r=load(TASK/'tests/reference.py','reference')
    inputs=r.calibration_inputs();true=r.TRUE_PARAMETER;sigma=.002;clean=r.predict(inputs,true)
    def records(values):return [dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,values)]
    if generate:
        data=json.dumps(records(clean+np.random.default_rng(946071).normal(0,sigma,len(inputs))),indent=2)+'\n'
        for name in ['environment/data/calibration.json','tests/data/calibration.json']:(TASK/name).write_text(data)
    data=json.loads((TASK/'environment/data/calibration.json').read_text());assert data==json.loads((TASK/'tests/data/calibration.json').read_text())
    groups=r.hidden_inputs();truth={name:r.predict(es,true) for name,es in groups.items()}
    def errors(module,rate):
        return {name:float(np.linalg.norm(np.array([module.survival(e,rate) for e in es])-truth[name])/np.linalg.norm(truth[name])) for name,es in groups.items()}
    controls={}
    for label,module in [('oracle',g),('shortcut',b)]:
        model=module.Model().fit(data);residual=(model.predict(inputs)-np.array([x['value'] for x in data]))/sigma
        controls[label]=dict(vacuum_rate=model.vacuum_rate,calibration_chi2=float(residual@residual)/(len(data)-1),hidden=errors(module,model.vacuum_rate))
        assert abs(model.vacuum_rate/true-1)<.03 and controls[label]['calibration_chi2']<1.5
        assert (max(controls[label]['hidden'].values())<.03 if label=='oracle' else min(controls[label]['hidden'].values())>.03)
    equivalence=max(abs(g.survival(e,true)-b.survival(e,true)) for e in inputs)
    independent=refinement=0.;minimum_total=minimum_source=minimum_flux=np.inf;minimum_survival=1.;maximum_survival=0.
    cases=[r.experiment(d,real,imag,tilt,.4) for d in [.2,1.5] for real in [-4.,-1.,1.,4.] for imag in [.3,2.] for tilt in [0.,np.pi/2]]
    for e in cases+[e for es in groups.values() for e in es[::4]]:
        expected=r.factor(e);components=g.factors(e['height'],e['epsilon_real'],e['epsilon_imag'])
        target=components@np.array([np.cos(e['tilt'])**2,np.sin(e['tilt'])**2])
        independent=max(independent,abs(target-expected));refinement=max(refinement,abs(r.factor(e,2e-12)-expected))
        minimum_total=min(minimum_total,target);minimum_source=min(minimum_source,float(min(b.factors(e['height'],e['epsilon_real'],e['epsilon_imag']))))
        for vertical in [True,False]:minimum_flux=min(minimum_flux,*r.power_components(e['height'],e['epsilon_real'],e['epsilon_imag'],vertical))
        for rate in [.4,1.2]:
            value=g.survival(e,rate);minimum_survival=min(minimum_survival,value);maximum_survival=max(maximum_survival,value)
    flux_heat=0.
    for eps in [-4+.3j,2+1.2j,4+2j]:
        for u in [.2,.9,1.2,4.]:
            q=np.sqrt(eps-u*u);te=np.array([0.,1.,0.]);tm=np.array([q,0.,u])/np.sqrt(eps)
            for weight in [0.,1.,.5+.7j]:
                electric=te+weight*tm;magnetic=np.cross(np.array([u,0.,-q]),electric)
                incoming=-.5*np.cross(electric,magnetic.conj())[2].real
                heat=eps.imag*np.vdot(electric,electric).real/(4*q.imag)
                flux_heat=max(flux_heat,abs(incoming-heat))
    nearfield=0.
    for eps in [2.5+.8j,-2+1j]:
        d=.001;target=np.array([3/8,3/16])*np.imag((eps-1)/(eps+1))/d**3
        actual=g.factors(d,eps.real,eps.imag)
        nearfield=max(nearfield,float(np.max(abs(actual/target-1))))
    assert equivalence==0 and independent<1e-7 and refinement<1e-7 and flux_heat<1e-12 and nearfield<1e-4
    assert minimum_total>0 and minimum_source>0 and minimum_flux>=0 and minimum_survival>=0 and maximum_survival<=1
    rng=np.random.default_rng(946073);noise=[]
    for _ in range(noise_trials):
        sample=records(clean+rng.normal(0,sigma,len(inputs)));model=b.Model().fit(sample)
        residual=(model.predict(inputs)-np.array([x['value'] for x in sample]))/sigma
        noise.append(dict(vacuum_rate=model.vacuum_rate,chi2=float(residual@residual)/(len(inputs)-1),oracle=errors(g,model.vacuum_rate),shortcut=errors(b,model.vacuum_rate)))
    summary=dict(calibration_passes=sum(n['chi2']<1.5 for n in noise),parameter_passes=sum(abs(n['vacuum_rate']/true-1)<.03 for n in noise),oracle_passes=sum(max(n['oracle'].values())<.03 for n in noise),shortcut_passes=sum(max(n['shortcut'].values())<.03 for n in noise),maximum_oracle_error=max(max(n['oracle'].values()) for n in noise),minimum_shortcut_error=min(min(n['shortcut'].values()) for n in noise),maximum_chi2=max(n['chi2'] for n in noise),rate_range=[min(n['vacuum_rate'] for n in noise),max(n['vacuum_rate'] for n in noise)])
    assert summary['calibration_passes']==noise_trials and summary['parameter_passes']==noise_trials and summary['oracle_passes']==noise_trials and summary['shortcut_passes']==0
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[ROOT/'scripts/emitter_lifetime_baseline.py',Path(__file__).resolve()]
    return dict(revision=1,noise_trials=noise_trials,calibration_seed=946071,noise_seed=946073,measurement_sigma=sigma,controls=controls,calibration_equivalence=equivalence,independent_energy_balance_error=independent,reference_quadrature_refinement=refinement,transmitted_flux_vs_joule_error=flux_heat,nearfield_asymptote_relative_error=nearfield,minimum_rate_factor=minimum_total,minimum_shortcut_rate_factor=minimum_source,minimum_power_channel=minimum_flux,noise=summary,noise_realizations=noise,seconds=time.time()-start,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args()
    report=run(args.generate,args.noise_trials);(ROOT/'results/emitter-lifetime-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['noise_realizations','source_sha256']},indent=2))
