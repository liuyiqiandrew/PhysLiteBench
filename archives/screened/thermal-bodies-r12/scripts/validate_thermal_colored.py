"""Validate finite-mass thermal calorimeters r12; regeneration is explicit."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/thermal-bodies'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


oracle=load(TASK/'solution/model.py','thermal_oracle')
shortcut=load(ROOT/'scripts/thermal_colored_baseline.py','thermal_shortcut')
reference=load(TASK/'tests/reference.py','thermal_reference')


def error(actual,truth):return float(np.linalg.norm(actual-truth)/np.linalg.norm(truth))


def run(generate=False,noise_trials=256):
    start=time.time();inputs=reference.calibration_inputs()
    sigma=np.array([.003 if e['readout']=='mode_correlation' else .0005 for e in inputs])
    clean=reference.predict(inputs,.7)
    def records(values):return [dict(input=e,value=float(v),sigma=float(s)) for e,v,s in zip(inputs,values,sigma)]
    if generate:
        data=json.dumps(records(clean+np.random.default_rng(876031).normal(size=len(inputs))*sigma),indent=2)+'\n'
        for location in ['environment/data/calibration.json','tests/data/calibration.json']:(TASK/location).write_text(data)
    data=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert data==json.loads((TASK/'tests/data/calibration.json').read_text())
    groups=reference.hidden_inputs();truth={name:reference.predict(es,.7) for name,es in groups.items()}
    controls={}
    def errors(gamma,module):return {name:error(module.predict_at(es,gamma),truth[name]) for name,es in groups.items()}
    for name,module in [('oracle',oracle),('shortcut',shortcut)]:
        model=module.Model().fit(data);residual=(model.predict(inputs)-np.array([x['value'] for x in data]))/sigma
        controls[name]=dict(friction=model.friction,calibration_chi2=float(residual@residual)/(len(inputs)-1),hidden=errors(model.friction,module))
        assert abs(model.friction/.7-1)<.03 and controls[name]['calibration_chi2']<1.5
        assert (max(controls[name]['hidden'].values())<.05 if name=='oracle' else min(controls[name]['hidden'].values())>.05)
    equivalent=max(float(max(abs(oracle.predict_at(inputs,g)-shortcut.predict_at(inputs,g)))) for g in [.3,.51,.7,.93,1.2])
    assert equivalent==0
    independent=max(float(max(abs(oracle.predict_at(es,.7)-truth[name]))) for name,es in groups.items())
    nodes,weights=leggauss(96);nodes=(nodes+1)/2;weights=weights/2
    quadrature=0.;refinement=0.;corner_error=0.;balance=0.;energy_balance=0.;positive=np.inf;gibbs=0.;short_time=0.
    # Parameter and finite-mass corners, including fast coupler motion.
    for gamma in [.3,1.2]:
        for mass in [.05,.3]:
            for tau in [.15,2.]:
                for temperatures in [[2.,.5],[1.,1.]]:
                    e=reference.preparation([.8,1.4],1.2,tau,mass,temperatures)
                    A,S,_=oracle.stationary_state(gamma,*e['springs'],e['coupling'],tau,mass,*temperatures)
                    positive=min(positive,float(np.linalg.eigvalsh(S).min()))
                    currents=[]
                    energy=np.zeros((8,8));K=oracle.stiffness(e['springs'],e['coupling']);link=gamma/tau
                    energy[:2,:2]=K+link*np.eye(2);energy[2:4,2:4]=np.eye(2)
                    energy[4:6,4:6]=link*np.eye(2);energy[:2,4:6]=energy[4:6,:2]=-link*np.eye(2)
                    energy[6:8,6:8]=np.eye(2)/mass
                    damping=np.zeros((8,8));damping[6:8,6:8]=-2*gamma*np.eye(2)/mass**2
                    energy_balance=max(energy_balance,float(np.max(abs(A.T@energy+energy@A-damping))))
                    if temperatures[0]==temperatures[1]:gibbs=max(gibbs,float(max(abs((S-temperatures[0]*np.linalg.inv(energy)).ravel()))))
                    for bath in [0,1]:
                        outer=gamma*(temperatures[bath]/mass-S[6+bath,6+bath]/mass**2)
                        inner=link*(S[2+bath,4+bath]-S[2+bath,bath]);balance=max(balance,abs(outer-inner));currents.append(outer)
                        query=dict(e,readout='heat_variance',bath=bath,duration=2.83)
                        a=oracle.predict_at([query],gamma)[0];b=reference.predict([query],gamma)[0]
                        corner_error=max(corner_error,abs(a-b));c=oracle.heat_variance(A,S,gamma,tau,mass,bath,2.83,nodes,weights);quadrature=max(quadrature,abs(a-c))
                        refined=reference.heat_moments(query,gamma,rtol=2e-12)[1];refinement=max(refinement,abs(refined-b))
                        duration=1e-6;actual=oracle.heat_variance(A,S,gamma,tau,mass,bath,duration)/duration
                        leading=2*gamma*temperatures[bath]*S[6+bath,6+bath]/mass**2
                        short_time=max(short_time,abs(actual/leading-1))
                    balance=max(balance,abs(sum(currents)))
    assert independent<1e-8 and corner_error<1e-7 and quadrature<1e-8 and refinement<1e-7
    assert balance<1e-9 and energy_balance<1e-10 and positive>0 and gibbs<1e-9 and short_time<1e-3
    fit_checks=[]
    for true in [.3,.43,.7,1.03,1.2]:
        sample=records(reference.predict(inputs,true));fit=oracle.Model().fit(sample).friction;fit_checks.append([true,fit]);assert abs(fit-true)<1e-6
    rng=np.random.default_rng(876034);noise=[]
    for _ in range(noise_trials):
        sample=records(clean+rng.normal(size=len(inputs))*sigma);model=shortcut.Model().fit(sample);gamma=model.friction
        residual=(model.predict(inputs)-np.array([x['value'] for x in sample]))/sigma
        noise.append(dict(friction=gamma,chi2=float(residual@residual)/(len(inputs)-1),oracle=errors(gamma,oracle),shortcut=errors(gamma,shortcut)))
    summary=dict(calibration_passes=sum(n['chi2']<1.5 for n in noise),parameter_passes=sum(abs(n['friction']/.7-1)<.03 for n in noise),oracle_passes=sum(max(n['oracle'].values())<.05 for n in noise),shortcut_passes=sum(max(n['shortcut'].values())<.05 for n in noise),maximum_oracle_error=max(max(n['oracle'].values()) for n in noise),minimum_shortcut_error=min(min(n['shortcut'].values()) for n in noise),maximum_chi2=max(n['chi2'] for n in noise),friction_range=[min(n['friction'] for n in noise),max(n['friction'] for n in noise)])
    assert summary['calibration_passes']==noise_trials and summary['parameter_passes']==noise_trials
    assert summary['oracle_passes']==noise_trials and summary['shortcut_passes']==0
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[ROOT/'scripts/thermal_colored_baseline.py',Path(__file__).resolve()]
    return dict(revision=12,noise_trials=noise_trials,controls=controls,calibration_records=len(data),instrument_sigma=dict(mode_correlation=.003,heat_current=.0005),calibration_seed=876031,noise_seed=876034,calibration_equivalence=equivalent,independent_hidden_heat_moment_error=independent,domain_corner_reference_error=corner_error,quadrature48_to96_error=quadrature,direct_moment_tolerance_refinement=refinement,stationary_inner_outer_and_total_heat_balance=balance,deterministic_total_energy_identity_error=energy_balance,minimum_covariance_eigenvalue=positive,equilibrium_Gibbs_covariance_error=gibbs,short_time_heat_noise_relative_error=short_time,noiseless_parameter_recovery=fit_checks,noise=summary,noise_realizations=noise,seconds=time.time()-start,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--regenerate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args()
    report=run(args.regenerate,args.noise_trials);(ROOT/'results/thermal-r12-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['noise_realizations','source_sha256']},indent=2))
