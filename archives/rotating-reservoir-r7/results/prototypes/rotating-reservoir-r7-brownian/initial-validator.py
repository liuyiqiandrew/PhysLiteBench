"""Validate finite-bias rotor heat statistics and completed Gaussian current control."""
import argparse
import ast
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/rotating-reservoir'
ARCHIVE=ROOT/'archives/rotating-reservoir-r6'
RESULTS=ROOT/'results'
DIAGNOSTICS=('rotating_equal_temperature_bias','rotating_cold_a_positive_bias','rotating_hot_a_negative_bias')
ANCHOR='stationary_statistics_anchor'


def module(path):
    spec=importlib.util.spec_from_file_location('rotor_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def relative(actual,expected):return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_records(ref,metadata):
    rng=np.random.default_rng(metadata['calibration_seed']);gamma=ref.TRUE_PARAMETER
    return [dict(input=e,value=float(gamma*3.6/(2*gamma+3.6)*(e['temperature_a']-e['temperature_b'])+rng.normal(0,metadata['measurement_sigma'])),sigma=metadata['measurement_sigma']) for e in ref.calibration_inputs()]


def parameters(e,drag):
    return (e.get('heat_bias',0.),e['temperature_a'],e['temperature_b'],e['angular_speed'],e['flux_fraction'],e['flip_rate'],float(drag))


def renewal(chi,ta,tb,gamma=.67,rate=3.6):
    nodes,weights=laggauss(96);coefficient=chi*(ta-tb)-chi*chi*ta*tb
    def residual(theta):
        times=nodes/(rate+theta)
        return rate/(rate+theta)*(weights@(1+coefficient*(-np.expm1(-2*gamma*times)))**-.5)-1
    return float(brentq(residual,-rate+1e-4,2.,xtol=2e-13))


def retained_physics(oracle,shortcut):
    frozen=ARCHIVE/'tasks/rotating-reservoir/solution/model.py'
    names=['gaussian_moment','maxwell_statistics','bernoulli','collision_grid','grid_statistics','physical_statistics']
    def definitions(path):
        tree=ast.parse(path.read_text())
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    old=definitions(frozen);unchanged={}
    for label,path in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/rotating_reservoir_baseline.py')]:
        new=definitions(path);unchanged[label]={name:new[name]==old[name] for name in names}
        assert all(unchanged[label].values())
    assert (TASK/'environment/model.py').read_bytes()==(ROOT/'scripts/rotating_reservoir_baseline.py').read_bytes()
    public=definitions(TASK/'environment/model.py')
    assert 'tilted_grid' not in public and 'eigvals' not in (TASK/'environment/model.py').read_text()
    assert public['scgf']==definitions(ROOT/'scripts/rotating_reservoir_baseline.py')['scgf']
    prior=ARCHIVE/'results/rotating-reservoir-r6-validation.json';report=json.loads(prior.read_text())
    for rel,digest in report['source_sha256'].items():assert sha(ARCHIVE/rel)==digest
    return dict(retained_mean_noise_AST_identity=unchanged,previous_science_report_sha256=sha(prior),
        previous_full_domain_and_thermodynamic_evidence=report['scientific_checks'],
        public_completed_control_matches=True,public_has_no_exact_tilted_generator=True)


def scientific_checks(oracle,shortcut,ref):
    hidden=sum(ref.hidden_inputs().values(),[]);ref.state.cache_clear();ref.scgf.cache_clear()
    start=time.perf_counter();truth=ref.predict(hidden);cold=time.perf_counter()-start
    exact=oracle.predict_at(hidden,.67)
    refinements={}
    for label,kwargs in [('spectral91',dict(points=91,quadrature=100)),('spectral151',dict(points=151,quadrature=180)),
                         ('tail7',dict(extent=7.)),('tail11',dict(extent=11.)),('quadrature100',dict(quadrature=100))]:
        refinements[label]=float(max(abs(ref.predict(hidden,**kwargs)-truth)))
    dense=[];fv_refined=[];bias_cases=[]
    for e in hidden:
        if e['statistic']!='scgf':continue
        a=oracle.scgf(e,.67,e['heat_bias']);b=oracle.scgf(e,.67,e['heat_bias'],dense=True)
        c=oracle.scgf(e,.67,e['heat_bias'],points=241,extent=11.)
        dense.append(abs(a-b));fv_refined.append(abs(a-c))
        standard=ref.scgf(*parameters(e,.67));tail=ref.scgf(*parameters(e,.67),points=181,quadrature=220,extent=13.)
        robin=ref.scgf(*parameters(e,.67),gauge='full',boundary='reflecting')
        alternate=ref.scgf(*parameters(e,.67),boundary='gauge_neumann')
        bias_cases.append(dict(input=e,scgf=a,independent=standard,tail=tail,full_robin_gauge=robin,gauge_neumann=alternate))
    assert max(dense)<2e-10 and max(fv_refined)<5e-6
    assert max(abs(exact-truth))<5e-6 and max(refinements.values())<2e-7 and cold<45
    corners=[]
    for drag,ta,tb,omega,flip,eta,chi in itertools.product([.4,1.1],[.6,1.6],[.6,1.4],[-.65,.65],[.8,2.],[0.,1.],[-.35,.35]):
        e=dict(angular_speed=omega,flip_rate=flip,temperature_a=ta,temperature_b=tb,flux_fraction=eta,heat_bias=chi,statistic='scgf')
        value=oracle.scgf(e,drag,chi);reference=ref.scgf(*parameters(e,drag));fine=ref.scgf(*parameters(e,drag),points=151,quadrature=180,extent=11.)
        reverse=oracle.scgf(dict(e,angular_speed=-omega),drag,chi)
        corners.append(dict(input=e,drag=drag,oracle=value,reference=reference,refined=fine,reversal_error=abs(value-reverse)))
    assert max(abs(r['oracle']-r['reference']['value']) for r in corners)<3e-5
    assert max(abs(r['reference']['value']-r['refined']['value']) for r in corners)<1e-8
    assert min(r['reference']['spectral_gap'] for r in corners)>1.
    assert max(r['reference']['imaginary'] for r in corners)<1e-10
    assert max(r['reversal_error'] for r in corners)<2e-9
    anchors=[]
    for ta,tb,chi in itertools.product([.6,1.,1.6],[.6,1.,1.4],[-.35,.35]):
        e=dict(angular_speed=0.,flip_rate=1.,temperature_a=ta,temperature_b=tb,flux_fraction=0.)
        value=renewal(chi,ta,tb);continuous=ref.scgf(chi,ta,tb)['value'];grid=oracle.scgf(e,.67,chi)
        anchors.append(dict(ta=ta,tb=tb,chi=chi,renewal=value,continuous=continuous,grid=grid))
    assert max(abs(r['renewal']-r['continuous']) for r in anchors)<1e-9
    assert max(abs(r['renewal']-r['grid']) for r in anchors)<5e-6
    direct_cycles=[];nodes,weights=hermgauss(48)
    for ta,tb,chi,duration in itertools.product([.6,1.6],[.6,1.4],[-.35,.35],[.1,1.,4.]):
        gas=np.sqrt(2*tb)*nodes[:,None]
        after=np.exp(-.67*duration)*gas+np.sqrt(2*ta*(-np.expm1(-2*.67*duration)))*nodes[None,:]
        measured=float(weights@np.exp(-chi*(after*after-gas*gas)/2)@weights/np.pi)
        closed=(1+(-np.expm1(-2*.67*duration))*(chi*(ta-tb)-chi*chi*ta*tb))**-.5
        direct_cycles.append(abs(measured-closed))
    assert max(direct_cycles)<1e-10
    gc=[]
    for ta,tb,eta,chi in itertools.product([1.,1.2],[1.,1.1],[0.,1.],[-.15,.15]):
        partner=1/tb-1/ta-chi
        assert abs(chi)<1/max(ta,tb) and abs(partner)<1/max(ta,tb)
        a=ref.scgf(chi,ta,tb,eta=eta)['value'];b=ref.scgf(partner,ta,tb,eta=eta)['value']
        gc.append(abs(a-b))
    assert max(gc)<1e-9
    cumulants=[]
    for gamma,omega,flip,ta,tb,eta in [(.67,0.,1.,1.,1.,0.),(.4,.65,.8,.6,1.4,1.),
            (1.1,-.65,2.,1.6,.6,1.),(.67,.65,1.,1.4,1.4,1.),(.8,-.5,1.6,.8,1.2,.4)]:
        e=dict(angular_speed=omega,flip_rate=flip,temperature_a=ta,temperature_b=tb,flux_fraction=eta)
        stationary=ref.state(gamma,ta,tb,omega,flip,eta);zero=ref.scgf(0.,ta,tb,omega,eta,flip,gamma)['value'];estimate=[]
        for delta in [.002,.001]:
            plus=ref.scgf(delta,ta,tb,omega,eta,flip,gamma)['value'];minus=ref.scgf(-delta,ta,tb,omega,eta,flip,gamma)['value']
            estimate.append(dict(mean=-(plus-minus)/(2*delta),noise=(plus+minus-2*zero)/delta**2))
        recovered={key:(4*estimate[1][key]-estimate[0][key])/3 for key in ['mean','noise']}
        cumulants.append(dict(input=e,drag=gamma,zero=zero,recovered=recovered,stationary_mean=stationary['mean'],stationary_noise=stationary['noise']))
        assert oracle.scgf(e,gamma,0.)==shortcut.scgf(e,gamma,0.)==0.
    assert max(abs(r['recovered']['noise']-r['stationary_noise']) for r in cumulants)<2e-6
    rng=np.random.default_rng(713106);interiors=[];microscopic=[];detailed_balance=[]
    for _ in range(24):
        gamma=rng.uniform(.4,1.1);e=dict(angular_speed=rng.uniform(-.65,.65),flip_rate=rng.uniform(.8,2.),temperature_a=rng.uniform(.6,1.6),temperature_b=rng.uniform(.6,1.4),flux_fraction=rng.uniform(0,1),heat_bias=rng.uniform(-.35,.35),statistic='scgf')
        a=oracle.scgf(e,gamma,e['heat_bias']);b=ref.scgf(*parameters(e,gamma))['value']
        interiors.append(dict(input=e,drag=gamma,oracle=a,reference=b))
    for _ in range(256):
        v,g,u=rng.normal(size=3);tb=rng.uniform(.6,1.4);h=((v-u)**2-(g-u)**2)/2
        microscopic.append(abs((g*g-v*v)/2+h-u*(g-v)))
        detailed_balance.append(abs((-(g-u)**2+(v-u)**2)/(2*tb)-h/tb))
    assert max(abs(r['oracle']-r['reference']) for r in interiors)<2e-5
    assert max(microscopic+detailed_balance)<2e-12
    return dict(cold_reference_seconds=cold,independent_hidden_error=float(max(abs(exact-truth))),reference_refinements=refinements,
        dense_Perron_selection_error=float(max(dense)),finite_volume_refinement_error=float(max(fv_refined)),hidden_bias_cases=bias_cases,
        domain_corner_count=len(corners),corners=corners,full_corner_reference_error=float(max(abs(r['oracle']-r['reference']['value']) for r in corners)),
        full_corner_reference_refinement=float(max(abs(r['reference']['value']-r['refined']['value']) for r in corners)),
        minimum_principal_spectral_gap=float(min(r['reference']['spectral_gap'] for r in corners)),
        stationary_exponential_domain_proof='For |chi|<=.35 use f=exp(a*v²/2), a=.4-.5*chi. Uniform margins: .225<=a<=.575<.625<=1/Ta and .225<=a+chi<=.575<.7142857<=1/Tb. The marked jump gain divided by f decays quadratically in v with a finite incoming Gaussian integral; OU has negative quadratic leading term, motor flips leave f fixed. A finite uniform upper bound on L_chi f/f and finite stationary E[f] prove the measured exponential moments are finite. No maximal-strip claim is made.',
        public_domain_verified=True,renewal_anchors=anchors,direct_gaussian_cycle_error=float(max(direct_cycles)),
        admissible_zero_flow_GC_error=float(max(gc)),cumulant_anchors=cumulants,interior_cases=interiors,
        microscopic_heat_motor_identity_error=float(max(microscopic)),local_detailed_balance_error=float(max(detailed_balance)),
        retained_physics=retained_physics(oracle,shortcut))


def local_controls():
    output=RESULTS/'rotating-reservoir-r7-controls';output.mkdir(parents=True,exist_ok=True);results={}
    for name,path in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/rotating_reservoir_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='rotor-r7-control-') as directory:
            app=Path(directory)/'app';shutil.copytree(TASK/'environment',app);shutil.copyfile(path,app/'model.py')
            private=app/'private_tests';shutil.copytree(TASK/'tests',private)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',MODEL_PATH=str(app/'model.py'),METRICS_PATH=str(output/(name+'-metrics.json')),PYTHONPATH=str(app)+os.pathsep+str(private))
            start=time.perf_counter();run=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(app/'test_public.py'),str(private/'test_hidden.py')],cwd=app,env=env,text=True,capture_output=True,timeout=60)
            (output/(name+'-pytest.txt')).write_text(run.stdout+run.stderr)
            expected='9 passed' if name=='oracle' else '3 failed, 6 passed'
            assert expected in run.stdout and run.returncode==(0 if name=='oracle' else 1),run.stdout+run.stderr
            results[name]=dict(seconds=time.perf_counter()-start,returncode=run.returncode,expected=expected,metrics=json.loads((output/(name+'-metrics.json')).read_text()))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args()
    start=time.perf_counter();oracle=module(TASK/'solution/model.py');shortcut=module(ROOT/'scripts/rotating_reservoir_baseline.py');ref=module(TASK/'tests/reference.py');meta=json.loads((TASK/'tests/metadata.json').read_text())
    records=calibration_records(ref,meta);public=TASK/'environment/data/calibration.json';private=TASK/'tests/data/calibration.json'
    assert records==json.loads(public.read_text())==json.loads(private.read_text())
    assert public.read_bytes()==private.read_bytes()==(ARCHIVE/'tasks/rotating-reservoir/environment/data/calibration.json').read_bytes()
    assert len(records)==288 and meta['revision']==7
    inputs=ref.calibration_inputs();hidden=ref.hidden_inputs();truth={name:ref.predict(es) for name,es in hidden.items()}
    def scores(model):return {name:relative(model.predict(es),truth[name]) for name,es in hidden.items()}
    def chi(model,rs):return float(np.sum(((model.predict(inputs)-np.array([r['value'] for r in rs]))/meta['measurement_sigma'])**2)/(len(rs)-1))
    controls={}
    for name,source in [('oracle',oracle),('shortcut',shortcut)]:
        model=source.Model().fit(records);row=dict(parameter=model.drag,parameter_relative_error=abs(model.drag/.67-1),calibration_chi2=chi(model,records),hidden=scores(model))
        assert row['parameter_relative_error']<.03 and row['calibration_chi2']<1.5
        assert all(v<.04 if name=='oracle' or key==ANCHOR else v>.04 for key,v in row['hidden'].items())
        controls[name]=row
    print('rotor r7 fitted controls pass',flush=True)
    science=scientific_checks(oracle,shortcut,ref);print('rotor r7 independent/full-box/thermodynamic science pass',flush=True)
    rng=np.random.default_rng(meta['noise_seed']);noise={name:dict(hidden_min=1e10,hidden_max=0.,anchor_max=0.,parameter_error_max=0.,chi2_max=0.,successful_trials=0) for name in controls}
    true=np.array([.67*3.6/(2*.67+3.6)*(e['temperature_a']-e['temperature_b']) for e in inputs])
    for index in range(args.noise_trials):
        sample=[dict(input=e,value=float(y),sigma=.001) for e,y in zip(inputs,true+rng.normal(0,.001,len(inputs)))]
        for name,source in [('oracle',oracle),('shortcut',shortcut)]:
            model=source.Model().fit(sample);values=scores(model);diagnostic=[values[key] for key in DIAGNOSTICS];anchor=values[ANCHOR];error=abs(model.drag/.67-1);cal=chi(model,sample)
            assert error<.03 and cal<1.5 and anchor<.04
            assert max(diagnostic)<.04 if name=='oracle' else min(diagnostic)>.04
            row=noise[name];row['hidden_min']=min(row['hidden_min'],min(diagnostic));row['hidden_max']=max(row['hidden_max'],max(diagnostic));row['anchor_max']=max(row['anchor_max'],anchor);row['parameter_error_max']=max(row['parameter_error_max'],error);row['chi2_max']=max(row['chi2_max'],cal);row['successful_trials']+=1
        if (index+1)%16==0:print('rotor r7 direct noise refits',index+1,flush=True)
    report=dict(revision=7,calibration_preserved_byte_for_byte=True,calibration_records=288,calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],noise_trials=args.noise_trials,calibration_sigma=meta['measurement_sigma'],controls=controls,noise=noise,scientific_checks=science,local_controls=local_controls())
    report['seconds']=time.perf_counter()-start
    files=sorted(p for p in TASK.rglob('*') if p.is_file() and not any(part in ['__pycache__','.pytest_cache'] for part in p.parts));files += [ROOT/'scripts/rotating_reservoir_baseline.py',Path(__file__)]
    report['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in files}
    target=RESULTS/'rotating-reservoir-r7-validation.json';target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',report=str(target),sha256=sha(target),seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
