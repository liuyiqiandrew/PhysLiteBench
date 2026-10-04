"""Validate external-wire noise against a closed two-state jump calculation."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/terminal-current-noise'


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


oracle=load('oracle',TASK/'solution/model.py')
shortcut=load('shortcut',ROOT/'scripts/terminal_current_noise_baseline.py')
ref=load('reference',TASK/'tests/reference.py')
META=json.loads((TASK/'tests/metadata.json').read_text())


def marked(e,rate,weights):
    a,b=oracle.transition_rates(e,rate);entry=a.sum();exit=b.sum();s=entry+exit
    p=np.array([exit,entry])/s;g=np.array([[-entry,exit],[entry,-exit]])
    j=np.array([[0.,-weights@b],[weights@a,0.]])
    j2=np.array([[0.,weights**2@b],[weights**2@a,0.]])
    return p,g,j,j2,s


def marked_psd(e,rate,weights):
    p,g,j,j2,s=marked(e,rate,weights);projection=np.outer(p,np.ones(2))
    resolvent=np.linalg.solve(1j*e['omega']*np.eye(2)-g+projection,np.eye(2)-projection)
    return float(2*np.ones(2)@j2@p+4*np.real(np.ones(2)@j@resolvent@j@p))


def admittance(e,rate,step=1e-5):
    weights=np.array([1-e['fraction'],-e['fraction']]);p,g,j,j2,s=marked(e,rate,weights)
    plus=marked(dict(e,voltage_left=e['voltage_left']+step),rate,weights)
    minus=marked(dict(e,voltage_left=e['voltage_left']-step),rate,weights)
    dg=(plus[1]-minus[1])/(2*step);dj=(plus[2]-minus[2])/(2*step)
    dp=np.linalg.solve(-1j*e['omega']*np.eye(2)-g+np.outer(p,np.ones(2)),dg@p)
    # Current is positive in electron units. Purely reactive geometric current has zero real part.
    return -(np.ones(2)@dj@p+np.ones(2)@j@dp)


def main():
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');args=p.parse_args();start=time.perf_counter()
    cal=ref.calibration_inputs();truth=ref.predict(cal);sigma=META['sigma']
    if args.generate:
        rng=np.random.default_rng(META['calibration_seed'])
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,truth+rng.normal(0,sigma,len(cal)))]
        for rel in ['environment/data/calibration.json','tests/data/calibration.json']:(TASK/rel).write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert len(records)==216 and all(r['sigma']==sigma for r in records)
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    unit=oracle.predict_at(cal,1.);wrong=shortcut.predict_at(cal,1.)
    cal_equal=float(np.max(abs(unit-wrong)));cal_ref=float(np.max(abs(unit-ref.predict(cal,1.))))
    assert cal_equal<2e-15 and cal_ref<2e-15
    groups=ref.hidden_inputs();target={k:ref.predict(v) for k,v in groups.items()};norm={k:np.linalg.norm(v) for k,v in target.items()}
    controls={}
    for label,module in [('oracle',oracle),('shortcut',shortcut)]:
        m=module.Model().fit(records);res=(m.predict(cal)-np.array([r['value'] for r in records]))/sigma
        hidden={k:float(np.linalg.norm(m.predict(v)-target[k])/norm[k]) for k,v in groups.items()}
        controls[label]={'rate':m.rate,'parameter_relative_error_max':abs(m.rate/ref.TRUE_PARAMETER-1),'calibration_chi2':float(res@res/(len(cal)-1)),'hidden':hidden}
        assert controls[label]['parameter_relative_error_max']<.03 and controls[label]['calibration_chi2']<1.5
        assert all((x<META['prediction_limit'])==(label=='oracle' or k=='zero_frequency') for k,x in hidden.items())
    rng=np.random.default_rng(META['noise_seed']);y=truth[None,:]+rng.normal(0,sigma,(256,len(cal)))
    rates=y@unit/(unit@unit);residual=(y-rates[:,None]*unit)/sigma;chi2=np.sum(residual**2,axis=1)/(len(cal)-1)
    errors=abs(rates/ref.TRUE_PARAMETER-1);assert chi2.max()<1.5 and errors.max()<.03
    extrema={}
    for label,module in [('oracle',oracle),('shortcut',shortcut)]:
        extrema[label]={}
        for k,v in groups.items():
            error=np.array([np.linalg.norm(module.predict_at(v,r)-target[k])/norm[k] for r in rates])
            extrema[label][k]={'min':float(error.min()),'max':float(error.max())}
            assert np.all((error<META['prediction_limit'])==(label=='oracle' or k=='zero_frequency'))
    maximum_error=0.;minimum_psd=1e9;minimum_short=1e9;dc_equality=0.;reversal=0.;scaling=0.;continuity=0.;stationarity=0.;count=0
    for offset,vl,vr,t,c,gl,gr,w,rate in itertools.product([-.4,.4],[-.8,.8],[-.8,.8],[.15,.4],[.15,.85],[.5,1.6],[.5,1.6],[0.,2.,10.],[.7,1.4]):
        e=ref.experiment(offset,vl,vr,t,c,gl,gr,w);answer=oracle.spectrum(e,rate);approx=shortcut.spectrum(e,rate)
        maximum_error=max(maximum_error,abs(answer-ref.spectrum(e,rate)));minimum_psd=min(minimum_psd,answer);minimum_short=min(minimum_short,approx)
        dc_equality=max(dc_equality,abs(oracle.spectrum(dict(e,omega=0.),rate)-shortcut.spectrum(dict(e,omega=0.),rate)))
        reversal=max(reversal,abs(answer-oracle.spectrum(dict(e,omega=-w),rate)))
        scaling=max(scaling,abs(oracle.spectrum(dict(e,omega=1.7*w),1.7*rate)-1.7*answer))
        p,g,j,j2,s=marked(e,rate,np.ones(2));stationarity=max(stationarity,float(np.max(abs(g@p))))
        charge_noise=4*p.prod()*s/(s*s+w*w)
        continuity=max(continuity,abs(marked_psd(e,rate,np.ones(2))-w*w*charge_noise))
        count+=1
    assert maximum_error<1e-13 and minimum_psd>0 and minimum_short>0 and dc_equality<1e-13
    assert reversal<1e-13 and scaling<1e-13 and stationarity<1e-14 and continuity<1e-13
    fdt_error=0.;fdt_refine=0.;equilibrium_count=0
    for offset,t,c,gl,gr,w in itertools.product([-.3,.1,.35],[.15,.4],[.15,.5,.85],[.6,1.4],[.7,1.5],[0.,1.,7.]):
        e=ref.experiment(offset,0.,0.,t,c,gl,gr,w);spectrum=oracle.spectrum(e,ref.TRUE_PARAMETER)
        first=4*t*admittance(e,ref.TRUE_PARAMETER).real;second=4*t*admittance(e,ref.TRUE_PARAMETER,5e-6).real
        fdt_error=max(fdt_error,abs(second-spectrum));fdt_refine=max(fdt_refine,abs(first-second));equilibrium_count+=1
    assert fdt_error<2e-9 and fdt_refine<2e-9
    integral_error=0.
    for e in [v[0] for v in groups.values()]:
        p,g,j,j2,s=marked(e,ref.TRUE_PARAMETER,np.array([1-e['fraction'],-e['fraction']]))
        project=np.outer(p,np.ones(2));amplitude=float(np.ones(2)@j@(np.eye(2)-project)@j@p)
        correlated=quad(lambda t:amplitude*np.exp(-s*t)*np.cos(e['omega']*t),0,np.inf,epsabs=1e-12)[0]
        integrated=float(2*np.ones(2)@j2@p+4*correlated)
        integral_error=max(integral_error,abs(integrated-ref.spectrum(e)))
    assert integral_error<2e-11
    recovery={}
    for rate in [.7,1.1,1.4]:
        records0=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,ref.predict(cal,rate))]
        recovery[str(rate)]=abs(oracle.Model().fit(records0).rate-rate);assert recovery[str(rate)]<1e-14
    high_error=0.
    for e in [v[0] for v in groups.values()]:
        p,g,j,j2,s=marked(e,ref.TRUE_PARAMETER,np.array([1-e['fraction'],-e['fraction']]))
        high_error=max(high_error,abs(oracle.spectrum(dict(e,omega=1e7),ref.TRUE_PARAMETER)-float(2*np.ones(2)@j2@p)))
    assert high_error<1e-12
    report={'task':'terminal-current-noise','revision':1,'noise_trials':256,'calibration_seed':META['calibration_seed'],'noise_seed':META['noise_seed'],
        'controls':controls,'noise':{'oracle_passes':256,'shortcut_rejections':256,'all_calibration_pass':True,'max_calibration_chi2':float(chi2.max()),'max_parameter_relative_error':float(errors.max())},'noise_extrema':extrema,
        'physical_checks':{'calibration_equivalence':cal_equal,'calibration_reference_error':cal_ref,'public_corner_count':count,'independent_closed_spectrum_error':maximum_error,'minimum_physical_psd':minimum_psd,'minimum_shortcut_psd':minimum_short,'zero_frequency_weight_equivalence':dc_equality,'frequency_reversal_error':reversal,'rate_time_scaling_error':scaling,'stationarity_error':stationarity,'charge_continuity_spectral_error':continuity,'equilibrium_cases':equilibrium_count,'voltage_response_FDT_error':fdt_error,'voltage_derivative_refinement':fdt_refine,'time_correlation_quadrature_error':integral_error,'high_frequency_jump_second_moment_error':high_error,'parameter_endpoint_recovery':recovery,'rate_information':float(unit@unit/sigma**2),'minimum_scored_signal':float(min(v.min() for k,v in target.items() if k!='zero_frequency')),'maximum_scored_signal':float(max(v.max() for k,v in target.items() if k!='zero_frequency'))},
        'seconds':time.perf_counter()-start,'source_sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'environment/data/calibration.json',TASK/'tests/metadata.json',ROOT/'scripts/terminal_current_noise_baseline.py',Path(__file__)]}}
    (ROOT/'results/terminal-current-noise-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
