"""Validate quantum residence-time calibration and scattering-delay control."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/quantum-residence'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def run(generate=False,noise_trials=256):
    start=time.time();g=load(TASK/'solution/model.py','oracle');b=load(ROOT/'scripts/quantum_residence_baseline.py','shortcut');r=load(TASK/'tests/reference.py','reference')
    inputs=r.calibration_inputs();true=r.TRUE_PARAMETER;sigma=.002;clean=r.predict(inputs,true)
    def records(values):return [dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,values)]
    if generate:
        data=json.dumps(records(clean+np.random.default_rng(945061).normal(0,sigma,len(inputs))),indent=2)+'\n'
        for name in ['environment/data/calibration.json','tests/data/calibration.json']:(TASK/name).write_text(data)
    data=json.loads((TASK/'environment/data/calibration.json').read_text());assert data==json.loads((TASK/'tests/data/calibration.json').read_text())
    groups=r.hidden_inputs();truth={name:r.predict(es,true) for name,es in groups.items()}
    def errors(module,width):
        return {name:float(np.linalg.norm(np.array([module.residence(e,width) for e in es])-truth[name])/np.linalg.norm(truth[name])) for name,es in groups.items()}
    controls={}
    for label,module in [('oracle',g),('shortcut',b)]:
        model=module.Model().fit(data);residual=(model.predict(inputs)-np.array([x['value'] for x in data]))/sigma
        controls[label]=dict(width=model.width,calibration_chi2=float(residual@residual)/(len(data)-1),hidden=errors(module,model.width))
        assert abs(model.width/true-1)<.03 and controls[label]['calibration_chi2']<1.5
        assert (max(controls[label]['hidden'].values())<.03 if label=='oracle' else min(controls[label]['hidden'].values())>.03)
    equivalence=0.;independent=0.;unitarity=0.;hermitian=0.;derivative=0.;probability_flow=0.;density_refinement=0.;absorption=0.;absorption_refinement=0.;minimum_delay=np.inf;minimum_residence=np.inf
    nodes,weights=leggauss(96)
    cases=[r.experiment(e,v) for e in [.5,1.3,2.4,3.5] for v in [0.,.7,2.3,4.5]]+[r.experiment(e,e) for e in [.5,1.7,3.5]]
    for width in [.6,.9,1.4]:
        for e in cases:
            energy,potential=e['energy'],e['potential'];S,dS=g.scattering(energy,potential,width);Q=-1j*S.conj().T@dS
            unitarity=max(unitarity,float(np.max(abs(S.conj().T@S-np.eye(2)))));hermitian=max(hermitian,float(np.max(abs(Q-Q.conj().T))))
            value=g.residence(e,width);target=r.population(energy,potential,width)
            independent=max(independent,abs(value-target));minimum_residence=min(minimum_residence,value);minimum_delay=min(minimum_delay,b.residence(e,width))
            fine=r.population(energy,potential,width,nodes,weights);density_refinement=max(density_refinement,abs(fine-target))
            h=1e-4
            def amplitudes(offset):
                a,_=r.amplitudes(energy+offset,potential,width);return np.array([[a[0],a[1]],[a[1],a[0]]])
            numerical=(amplitudes(-2*h)-8*amplitudes(-h)+8*amplitudes(h)-amplitudes(2*h))/(12*h)
            derivative=max(derivative,float(np.max(abs(numerical-dS))))
            a,q=r.amplitudes(energy,potential,width);x=np.linspace(0,width,21)
            psi=a[2]*np.cos(q*x)+a[3]*x*np.sinc(q*x/np.pi)
            slope=-a[2]*q*np.sin(q*x)+a[3]*np.cos(q*x)
            probability_flow=max(probability_flow,float(max(abs(np.imag(psi.conj()*slope)-np.sqrt(2*energy)*abs(a[1])**2))))
            def absorbed(scale):
                values=[r.weak_absorption(energy,potential,width,scale/f) for f in [1,2,4]]
                return (values[0]-6*values[1]+8*values[2])/3
            estimate=absorbed(.002);refined=absorbed(.001)
            absorption=max(absorption,abs(estimate-target));absorption_refinement=max(absorption_refinement,abs(refined-estimate))
            if potential==0:equivalence=max(equivalence,abs(g.residence(e,width)-b.residence(e,width)))
    assert independent<1e-10 and density_refinement<1e-10 and unitarity<1e-10 and hermitian<1e-10
    assert derivative<1e-8 and probability_flow<1e-10 and absorption<1e-6 and absorption_refinement<1e-6
    assert equivalence==0 and minimum_residence>0 and minimum_delay>0
    rng=np.random.default_rng(945063);noise=[]
    for _ in range(noise_trials):
        sample=records(clean+rng.normal(0,sigma,len(inputs)));model=b.Model().fit(sample)
        residual=(model.predict(inputs)-np.array([x['value'] for x in sample]))/sigma
        noise.append(dict(width=model.width,chi2=float(residual@residual)/(len(inputs)-1),oracle=errors(g,model.width),shortcut=errors(b,model.width)))
    summary=dict(calibration_passes=sum(n['chi2']<1.5 for n in noise),parameter_passes=sum(abs(n['width']/true-1)<.03 for n in noise),oracle_passes=sum(max(n['oracle'].values())<.03 for n in noise),shortcut_passes=sum(max(n['shortcut'].values())<.03 for n in noise),maximum_oracle_error=max(max(n['oracle'].values()) for n in noise),minimum_shortcut_error=min(min(n['shortcut'].values()) for n in noise),maximum_chi2=max(n['chi2'] for n in noise),width_range=[min(n['width'] for n in noise),max(n['width'] for n in noise)])
    assert summary['calibration_passes']==noise_trials and summary['parameter_passes']==noise_trials and summary['oracle_passes']==noise_trials and summary['shortcut_passes']==0
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[ROOT/'scripts/quantum_residence_baseline.py',Path(__file__).resolve()]
    return dict(revision=1,noise_trials=noise_trials,calibration_seed=945061,noise_seed=945063,measurement_sigma=sigma,controls=controls,calibration_equivalence=equivalence,independent_density_error=independent,density48_to96_error=density_refinement,scattering_unitarity_error=unitarity,smith_hermiticity_error=hermitian,independent_energy_derivative_error=derivative,interior_probability_current_error=probability_flow,weak_absorption_extrapolation_error=absorption,weak_absorption_refinement=absorption_refinement,minimum_physical_residence=minimum_residence,minimum_shortcut_delay=minimum_delay,noise=summary,noise_realizations=noise,seconds=time.time()-start,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args()
    report=run(args.generate,args.noise_trials);(ROOT/'results/quantum-residence-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['noise_realizations','source_sha256']},indent=2))
