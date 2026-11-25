"""Validate hard-collision rotor flux and exact correlated Maxwell shortcut."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from itertools import product
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/rotating-reservoir'
ARCHIVE=ROOT/'archives/rotating-reservoir-r5'
RESULTS=ROOT/'results'


def module(path):
    spec=importlib.util.spec_from_file_location('rotor_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def calibration_records(ref,metadata):
    rng=np.random.default_rng(metadata['calibration_seed']);gamma=ref.TRUE_PARAMETER
    return [dict(input=e,value=float(gamma*3.6/(2*gamma+3.6)*(e['temperature_a']-e['temperature_b'])+rng.normal(0,metadata['measurement_sigma'])),sigma=metadata['measurement_sigma']) for e in ref.calibration_inputs()]


def relative(actual,expected):return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def scientific_checks(oracle,shortcut,ref):
    hidden=sum(ref.hidden_inputs().values(),[]);ref.state.cache_clear()
    start=time.perf_counter();truth=ref.predict(hidden);cold=time.perf_counter()-start
    exact=oracle.predict_at(hidden,.67)
    independent=float(max(abs(exact-truth)))
    refinements={}
    for name,kwargs in [('spectral91',dict(points=91,quadrature=100)),('spectral151',dict(points=151,quadrature=180)),
                        ('quadrature100',dict(quadrature=100)),('tail7',dict(extent=7.)),('tail11',dict(extent=11.))]:
        refinements[name]=float(max(abs(ref.predict(hidden,**kwargs)-truth)))
    refined=np.array([oracle.physical_statistics(e,.67,points=321)[e['statistic']] for e in hidden])
    fv_refinement=float(max(abs(refined-exact)))
    stationary=[];poisson=[];discrete_energy=[];continuum_energy=[];spectral_energy=[];entropy=[];tail=[];edge=[];probability=[];mass=[]
    reversal=[];noise=[];reference_relative=[];reference_absolute=[]
    corners=list(product([.4,.67,1.1],[-.65,0.,.65],[.8,2.],[.6,1.6],[.6,1.4],[0.,1.]))
    for drag,omega,flip,ta,tb,eta in corners:
        e=ref.experiment(omega,flip,ta,tb,eta)
        spectral=ref.state(drag,ta,tb,omega,flip,eta)
        coarse=oracle.grid_statistics(e,drag,241,details=True);fine=oracle.grid_statistics(e,drag,482,details=True)
        extrapolated={k:(4*fine[k]-coarse[k])/3 for k in ['mean','noise','motor','velocity_square']}
        actual=oracle.physical_statistics(e,drag)
        reference_absolute.extend([abs(actual[k]-spectral[k]) for k in ['mean','noise']])
        reference_relative.append(abs(actual['noise']-spectral['noise'])/spectral['noise'])
        qa=drag*(extrapolated['velocity_square']-ta)
        continuum_energy.append(abs(qa+extrapolated['mean']-extrapolated['motor']))
        spectral_energy.append(abs(drag*(spectral['velocity_square']-ta)+spectral['mean']-spectral['motor']))
        entropy.append(qa/ta+extrapolated['mean']/tb);noise.append(actual['noise'])
        for values in [coarse,fine]:
            stationary.append(values['stationary_residual']);poisson.append(values['poisson_residual'])
            discrete_energy.append(abs(values['bath_a_discrete']+values['mean']-values['motor']))
            probability.append(values['minimum_probability']);tail.append(values['outer_two_sigma_probability']);edge.append(values['edge_probability']);mass.extend([values['normalization_error'],values['conditional_mass_error']])
        reverse=oracle.physical_statistics(dict(e,angular_speed=-omega),drag)
        reversal.extend([abs(actual[k]-reverse[k]) for k in ['mean','noise']])
    assert min(noise)>0 and min(probability)>-1e-10 and min(entropy)>-2e-5
    assert max(tail)<1e-8 and max(edge)<1e-11
    assert max(reference_relative)<1e-5 and max(reference_absolute)<2e-5,(max(reference_relative),max(reference_absolute))
    assert max(continuum_energy)<2e-5 and max(spectral_energy)<2e-7
    assert max(stationary+poisson+discrete_energy+mass+reversal)<2e-8
    anchors=[];source_checks=[]
    for drag,temp,rate in product([.4,.67,1.1],[.6,1.,1.4],[3.6,2.,6.]):
        e=ref.experiment(0.,1.,temp,temp,0.)
        state=shortcut.maxwell_statistics(e,drag,rate)
        anchors.extend([abs(state['mean']),abs(state['noise']-2*drag*rate*temp**2/(2*drag+rate)),abs(state['shot']-rate*temp**2)])
    for drag,ta,tb,flip in product([.4,.67,1.1],[.6,1.6],[.6,1.4],[.8,2.]):
        e=ref.experiment(0.,flip,ta,tb,0.,'mean')
        closed=drag*3.6/(2*drag+3.6)*(ta-tb)
        anchors.extend([abs(oracle.predict_at([e],drag)[0]-closed),abs(shortcut.predict_at([e],drag)[0]-closed)])
    for drag,omega,flip,ta,tb,eta in corners:
        e=ref.experiment(omega,flip,ta,tb,eta)
        rate=shortcut.approximate_rate(tb,eta);values=shortcut.source_statistics(e,drag);m=values['moments']
        second=m[2]+m[7];signed=m[1]-m[6]
        motor=rate*(omega**2-omega*signed)
        source_checks.extend([values['stationary_residual'],abs(drag*(second-ta)+values['mean']-motor)])
        source_entropy=drag*(second-ta)/ta+values['mean']/tb
        assert source_entropy>-1e-10 and values['noise']>0
    assert max(anchors+source_checks)<2e-10
    isolated=[]
    for eta in [0.,.5,1.]:
        e=ref.experiment(0.,1.,1.,1.,eta)
        grid=oracle.grid_statistics(e,0.,241,details=True)
        spectral=ref.state(0.,1.,1.,0.,1.,eta)
        isolated.extend([abs(grid['mean']),abs(grid['noise']),abs(spectral['mean']),abs(spectral['noise'])])
    assert max(isolated)<1e-7
    rng=np.random.default_rng(610607);microscopic=[];detailed_balance=[];interiors=[]
    for _ in range(256):
        v,g,u=rng.normal(size=3);h=((v-u)**2-(g-u)**2)/2
        microscopic.append(abs((g*g-v*v)/2+h-u*(g-v)))
        # Reverse hard collision flux is identical; its rate ratio equals
        # exp(h/Tb), so calorimeter heat is the local detailed-balance entropy.
        tb=rng.uniform(.6,1.4)
        forward=np.exp(-(g-u)**2/(2*tb));reverse=np.exp(-(v-u)**2/(2*tb))
        detailed_balance.append(abs(np.log(forward/reverse)-h/tb))
    for _ in range(32):
        drag=rng.uniform(.4,1.1);e=ref.experiment(rng.uniform(-.65,.65),rng.uniform(.8,2),rng.uniform(.6,1.6),rng.uniform(.6,1.4),rng.uniform(0,1))
        values=oracle.physical_statistics(e,drag);reference=ref.state(drag,e['temperature_a'],e['temperature_b'],e['angular_speed'],e['flip_rate'],e['flux_fraction'])
        interiors.extend([abs(values[k]-reference[k]) for k in ['mean','noise']])
    assert max(interiors)<2e-5 and max(microscopic+detailed_balance)<2e-12
    assert independent<5e-6 and max(refinements.values())<2e-7 and fv_refinement<5e-6 and cold<45
    return dict(cold_reference_seconds=cold,independent_hidden_error=independent,reference_refinements=refinements,
        finite_volume_refinement_error=fv_refinement,domain_corner_count=len(corners),minimum_noise=float(min(noise)),minimum_fv_probability=float(min(probability)),
        maximum_outer_tail_probability=float(max(tail)),maximum_edge_probability=float(max(edge)),normalization_mass_error=float(max(mass)),
        fv_stationary_residual=float(max(stationary)),fv_poisson_residual=float(max(poisson)),discrete_first_law_error=float(max(discrete_energy)),
        extrapolated_continuum_first_law_error=float(max(continuum_energy)),spectral_first_law_error=float(max(spectral_energy)),
        minimum_thermal_entropy_production=float(min(entropy)),rotation_reversal_error=float(max(reversal)),
        full_corner_reference_absolute_error=float(max(reference_absolute)),full_corner_reference_noise_relative_error=float(max(reference_relative)),
        exact_Maxwell_closed_anchor_error=float(max(anchors)),source_moment_and_first_law_error=float(max(source_checks)),
        isolated_bounded_heat_zero_noise_error=float(max(isolated)),microscopic_motor_heat_identity_error=float(max(microscopic)),
        local_detailed_balance_error=float(max(detailed_balance)),independent_interior_count=32,independent_interior_error=float(max(interiors)),
        continuous_domain_moment_argument='For every even p>=2, OU contributes -p*gamma*|v|^p plus a lower-power diffusion term; the positive jump part is at most lambda*(M_p+|v|M_p+M_{p+1}), uniformly in eta in[0,1] and bounded incoming means/temperatures. Young bounds yield L|v|^p<=-c|v|^p+C. Linear event hazard and these moments imply nonexplosion. OU ellipticity and bounded-parameter Gaussian reset irreducibility yield geometric ergodicity and integrable heat cumulants. An analogous exp(a*v²) argument holds for a<1/(2*maxTa,Tb), justifying Gaussian tail control.',public_domain_verified=True)


def local_controls():
    output=RESULTS/'rotating-reservoir-r6-controls';output.mkdir(parents=True,exist_ok=True);results={}
    for name,path in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/rotating_reservoir_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='rotor-r6-control-') as directory:
            app=Path(directory)/'app';shutil.copytree(TASK/'environment',app);shutil.copyfile(path,app/'model.py')
            private=app/'private_tests';shutil.copytree(TASK/'tests',private)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                     MODEL_PATH=str(app/'model.py'),METRICS_PATH=str(output/(name+'-metrics.json')),PYTHONPATH=str(app)+os.pathsep+str(private))
            start=time.perf_counter();run=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(app/'test_public.py'),str(private/'test_hidden.py')],cwd=app,env=env,text=True,capture_output=True,timeout=60)
            (output/(name+'-pytest.txt')).write_text(run.stdout+run.stderr)
            expected='9 passed' if name=='oracle' else '3 failed, 6 passed'
            assert expected in run.stdout and run.returncode==(0 if name=='oracle' else 1),run.stdout+run.stderr
            results[name]=dict(seconds=time.perf_counter()-start,returncode=run.returncode,expected=expected,metrics=json.loads((output/(name+'-metrics.json')).read_text()))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--noise-trials',type=int,default=256);parser.add_argument('--regenerate-calibration',action='store_true');args=parser.parse_args()
    start=time.perf_counter();oracle=module(TASK/'solution/model.py');shortcut=module(ROOT/'scripts/rotating_reservoir_baseline.py');ref=module(TASK/'tests/reference.py');meta=json.loads((TASK/'tests/metadata.json').read_text())
    records=calibration_records(ref,meta)
    if args.regenerate_calibration:
        for p in [TASK/'environment/data/calibration.json',TASK/'tests/data/calibration.json']:p.write_text(json.dumps(records,indent=2)+'\n')
    assert records==json.loads((TASK/'environment/data/calibration.json').read_text())==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert len(records)==288 and (TASK/'environment/data/calibration.json').read_bytes()!=(ARCHIVE/'tasks/rotating-reservoir/environment/data/calibration.json').read_bytes()
    hidden=ref.hidden_inputs();truth={name:ref.predict(es) for name,es in hidden.items()};inputs=ref.calibration_inputs()
    def scores(model):return {name:relative(model.predict(es),truth[name]) for name,es in hidden.items()}
    def chi(model,rs):return float(np.sum(((model.predict(inputs)-np.array([r['value'] for r in rs]))/meta['measurement_sigma'])**2)/(len(rs)-1))
    controls={}
    for name,source in [('oracle',oracle),('shortcut',shortcut)]:
        model=source.Model().fit(records);row=dict(parameter=model.drag,parameter_relative_error=abs(model.drag/.67-1),calibration_chi2=chi(model,records),hidden=scores(model))
        assert row['parameter_relative_error']<.03 and row['calibration_chi2']<1.5
        assert all(value<.04 if name=='oracle' or key=='maxwell_anchor' else value>.04 for key,value in row['hidden'].items())
        controls[name]=row
    print('rotor r6 fitted controls pass',flush=True)
    rng=np.random.default_rng(meta['noise_seed']);noise={name:dict(hidden_min=1e10,hidden_max=0.,anchor_max=0.,parameter_error_max=0.,chi2_max=0.,successful_trials=0) for name in controls}
    true=np.array([.67*3.6/(2*.67+3.6)*(e['temperature_a']-e['temperature_b']) for e in inputs])
    for index in range(args.noise_trials):
        sample=[dict(input=e,value=float(y),sigma=.001) for e,y in zip(inputs,true+rng.normal(0,.001,len(inputs)))]
        for name,source in [('oracle',oracle),('shortcut',shortcut)]:
            model=source.Model().fit(sample);values=scores(model);diagnostic=[v for k,v in values.items() if k!='maxwell_anchor'];anchor=values['maxwell_anchor'];error=abs(model.drag/.67-1);cal=chi(model,sample)
            assert error<.03 and cal<1.5 and anchor<.04
            assert max(diagnostic)<.04 if name=='oracle' else min(diagnostic)>.04
            row=noise[name];row['hidden_min']=min(row['hidden_min'],min(diagnostic));row['hidden_max']=max(row['hidden_max'],max(diagnostic));row['anchor_max']=max(row['anchor_max'],anchor);row['parameter_error_max']=max(row['parameter_error_max'],error);row['chi2_max']=max(row['chi2_max'],cal);row['successful_trials']+=1
        if (index+1)%64==0:print('rotor r6 noise draws',index+1,flush=True)
    report=dict(revision=6,calibration_intentionally_regenerated_for_fixed_ring=True,calibration_records=288,calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],noise_trials=args.noise_trials,
        calibration_sigma=meta['measurement_sigma'],controls=controls,noise=noise,scientific_checks=scientific_checks(oracle,shortcut,ref),local_controls=local_controls())
    report['seconds']=time.perf_counter()-start
    files=sorted(p for p in TASK.rglob('*') if p.is_file() and not any(part in ['__pycache__','.pytest_cache'] for part in p.parts));files += [ROOT/'scripts/rotating_reservoir_baseline.py',Path(__file__)]
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (RESULTS/'rotating-reservoir-r6-validation.json').write_text(json.dumps(report,indent=2)+'\n');print('rotor r6 validation PASS',report['seconds'],flush=True)


if __name__=='__main__':main()
