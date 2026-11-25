"""Validate the stationary detector spectrum and local-correlation control."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import warnings
import numpy as np
from scipy.integrate import IntegrationWarning

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/circular-detector'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def run(generate=False,noise_trials=256):
    started=time.time();g=load(TASK/'solution/model.py','oracle');b=load(ROOT/'scripts/circular_detector_baseline.py','shortcut');r=load(TASK/'tests/reference.py','reference')
    inputs=r.calibration_inputs();true=r.TRUE_PARAMETER;sigma=5e-7;clean=r.predict(inputs,true)
    def records(values):return [dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,values)]
    if generate:
        data=json.dumps(records(clean+np.random.default_rng(947081).normal(0,sigma,len(inputs))),indent=2)+'\n'
        for name in ['environment/data/calibration.json','tests/data/calibration.json']:(TASK/name).write_text(data)
    data=json.loads((TASK/'environment/data/calibration.json').read_text());assert data==json.loads((TASK/'tests/data/calibration.json').read_text())
    groups=r.hidden_inputs();truth={name:r.predict(es,true) for name,es in groups.items()}
    def errors(module,coupling):return {name:float(np.linalg.norm(coupling*np.array([module.response(e) for e in es])-truth[name])/np.linalg.norm(truth[name])) for name,es in groups.items()}
    controls={}
    for label,module in [('oracle',g),('shortcut',b)]:
        model=module.Model().fit(data);residual=(model.predict(inputs)-np.array([x['value'] for x in data]))/sigma
        controls[label]=dict(coupling=model.coupling,calibration_chi2=float(residual@residual)/(len(data)-1),hidden=errors(module,model.coupling))
        assert abs(model.coupling/true-1)<.03 and controls[label]['calibration_chi2']<1.5
        assert (max(controls[label]['hidden'].values())<.03 if label=='oracle' else min(controls[label]['hidden'].values())>.03)
    equivalence=max(abs(g.response(e)-b.response(e)) for e in inputs)
    cusp_reference=max(abs(g.response(e)-r.predict([e],1.)[0]) for e in inputs)
    independent=mode_refinement=angular_refinement=commutator=scaling=0.;minimum_rate=np.inf
    cases=[r.experiment('circle',a,gap,readout,speed=v) for a in [.6,1.8] for gap in [.08,2.5] for v in [.3,.55,.85] for readout in ['excitation','deexcitation']]
    cases += [e for es in groups.values() for e in es]
    numerical_warnings=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',IntegrationWarning)
        for e in cases:
            target=r.predict([e],1.)[0];actual=g.response(e);signed=e['gap']*(1 if e['readout']=='excitation' else -1)
            fine=r.mode_rate(e['acceleration'],e['speed'],signed,600,80)
            angular=r.mode_rate(e['acceleration'],e['speed'],signed,400,112)
            independent=max(independent,abs(actual-target));mode_refinement=max(mode_refinement,abs(fine-target));angular_refinement=max(angular_refinement,abs(angular-target))
            minimum_rate=min(minimum_rate,actual,b.response(e),target)
            up=r.mode_rate(e['acceleration'],e['speed'],e['gap']);down=r.mode_rate(e['acceleration'],e['speed'],-e['gap'])
            commutator=max(commutator,abs(down-up-e['gap']/(2*np.pi)))
            altered=dict(e,acceleration=e['acceleration']*.8,gap=e['gap']*.8)
            scaling=max(scaling,abs(g.response(altered)-.8*actual))
        numerical_warnings=[str(x.message) for x in caught]
    # The local closure reproduces the exact circle interval through sixth order.
    interval_order=[]
    for v in [.3,.6,.85]:
        gamma=1/np.sqrt(1-v*v);a=1.;c=a*a/(30*(gamma*v)**2)
        def difference(s):
            x=a*s/(2*gamma*v)
            true_interval=s*s*(1+(gamma*v)**2*(1-np.sinc(x/np.pi)**2))
            local=s*s+a*a*s**4/(12*(1+c*s*s))
            return abs(local-true_interval)
        interval_order.append(difference(.2)/difference(.1))
    ultra_error=abs(g.circular_excitation(1.,.9999,.5)/(1/(8*np.pi*np.sqrt(3))*np.exp(-np.sqrt(3)))-1)
    assert equivalence<1e-16 and cusp_reference<1e-12 and independent<2e-10 and mode_refinement<1e-12 and angular_refinement<1e-12
    assert commutator<1e-12 and scaling<1e-12 and minimum_rate>0 and not numerical_warnings
    assert all(200<ratio<300 for ratio in interval_order) and ultra_error<.001
    rng=np.random.default_rng(947083);noise=[]
    for _ in range(noise_trials):
        sample=records(clean+rng.normal(0,sigma,len(inputs)));model=b.Model().fit(sample)
        residual=(model.predict(inputs)-np.array([x['value'] for x in sample]))/sigma
        noise.append(dict(coupling=model.coupling,chi2=float(residual@residual)/(len(inputs)-1),oracle=errors(g,model.coupling),shortcut=errors(b,model.coupling)))
    summary=dict(calibration_passes=sum(n['chi2']<1.5 for n in noise),parameter_passes=sum(abs(n['coupling']/true-1)<.03 for n in noise),oracle_passes=sum(max(n['oracle'].values())<.03 for n in noise),shortcut_passes=sum(max(n['shortcut'].values())<.03 for n in noise),maximum_oracle_error=max(max(n['oracle'].values()) for n in noise),minimum_shortcut_error=min(min(n['shortcut'].values()) for n in noise),maximum_chi2=max(n['chi2'] for n in noise),coupling_range=[min(n['coupling'] for n in noise),max(n['coupling'] for n in noise)])
    assert summary['calibration_passes']==noise_trials and summary['parameter_passes']==noise_trials and summary['oracle_passes']==noise_trials and summary['shortcut_passes']==0
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[ROOT/'scripts/circular_detector_baseline.py',Path(__file__).resolve()]
    return dict(revision=1,noise_trials=noise_trials,calibration_seed=947081,noise_seed=947083,measurement_sigma=sigma,controls=controls,calibration_equivalence=equivalence,cusp_fourier_reference_error=cusp_reference,independent_cylindrical_modes_error=independent,mode400_to600_error=mode_refinement,angular80_to112_error=angular_refinement,vacuum_commutator_error=commutator,scale_covariance_error=scaling,minimum_rate_factor=minimum_rate,interval_difference_halving_ratios=interval_order,ultrarelativistic_cusp_relative_error=ultra_error,numerical_warnings=numerical_warnings,noise=summary,noise_realizations=noise,seconds=time.time()-started,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args()
    report=run(args.generate,args.noise_trials);(ROOT/'results/circular-detector-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['noise_realizations','source_sha256']},indent=2))
