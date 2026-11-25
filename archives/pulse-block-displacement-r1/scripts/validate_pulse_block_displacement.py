"""Author science and isolated local controls; no Docker or model evaluations."""
import argparse
import ast
from itertools import product
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
import traceback

import numpy as np
from numpy.polynomial.legendre import leggauss

BASE = Path(__file__).resolve().parents[1]
TASK = BASE/'tasks/pulse-block-displacement'
RESULTS = BASE/'results'
REPORT = {}


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle=load('pulse_oracle',TASK/'solution/model.py')
shortcut=load('pulse_shortcut',BASE/'scripts/pulse_block_displacement_baseline.py')
reference=load('pulse_reference',TASK/'tests/reference.py')
metadata=json.loads((TASK/'tests/metadata.json').read_text())


def nrmse(value,truth):
    return float(np.sqrt(np.mean((value-truth)**2)/np.mean(truth**2)))


def make_input(center,width,length,mode='displacement'):
    return {'mode':mode,'center':float(center),'width':float(width),'length':float(length)}


def local_controls():
    controls={}
    for name,path in [('oracle',TASK/'solution/model.py'),('shortcut',BASE/'scripts/pulse_block_displacement_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='pulse-block-control-') as folder:
            root=Path(folder)
            shutil.copytree(TASK/'environment',root/'app')
            shutil.copytree(TASK/'tests',root/'tests')
            shutil.copy2(path,root/'app/model.py')
            env=os.environ.copy()
            env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'app'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.monotonic()
            result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
                str(root/'app/test_public.py'),str(root/'tests/test_hidden.py')],cwd=root,env=env,
                capture_output=True,text=True,timeout=60)
            controls[name]={'returncode':result.returncode,'seconds':time.monotonic()-start,
                            'stdout':result.stdout,'stderr':result.stderr}
    (RESULTS/'pulse-block-displacement-r1-local-controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    assert controls['oracle']['returncode']==0 and '7 passed' in controls['oracle']['stdout']
    assert controls['shortcut']['returncode']==1 and '3 failed, 4 passed' in controls['shortcut']['stdout']
    return {name:{key:value for key,value in row.items() if key not in ('stdout','stderr')} for name,row in controls.items()}


def run(generate=False):
    start=time.monotonic();true=reference.TRUE_PARAMETER;sigma=metadata['measurement_sigma']
    settings=reference.calibration_inputs();inputs=settings*metadata['calibration_repeats']
    clean=reference.predict(inputs,true)
    if generate:
        values=clean+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(inputs))
        records=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,values)]
        text=json.dumps(records,indent=2)+'\n'
        for side in ('environment','tests'):(TASK/side/'data/calibration.json').write_text(text)
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert len(settings)==18 and len(records)==288 and [r['input'] for r in records]==inputs
    assert all(r['sigma']==sigma for r in records)
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    REPORT.update(status='science_in_progress',revision=1,model_evaluations=0,true_oscillator_strength=true,
                  distinct_calibration_settings=len(settings),calibration_repeats=metadata['calibration_repeats'],
                  calibration_count=len(records),sigma=sigma,prediction_limit=metadata['prediction_limit'],
                  calibration_seed=metadata['calibration_seed'],noise_seed=metadata['noise_seed'])
    groups=reference.hidden_inputs();truths={k:reference.predict(v,true) for k,v in groups.items()}
    actual={}
    for name,mod in [('oracle',oracle),('shortcut',shortcut)]:
        model=mod.Model().fit(records)
        residual=(model.predict(inputs)-np.array([r['value'] for r in records]))/sigma
        actual[name]={'oscillator_strength':model.oscillator_strength,
                     'parameter_relative_error':abs(model.oscillator_strength/true-1),
                     'calibration_chi2':float(residual@residual/(len(records)-1)),
                     'hidden':{k:nrmse(model.predict(v),truths[k]) for k,v in groups.items()}}
        assert actual[name]['parameter_relative_error']<.03 and actual[name]['calibration_chi2']<1.5
    REPORT['actual_data']=actual
    ref_error=max(float(np.max(abs(oracle.predict_at(v,true)-truths[k]))) for k,v in groups.items())
    refinement=max(float(np.max(abs(reference.predict(v,true,32768)-truths[k]))) for k,v in groups.items())
    cal_bias=float(np.max(abs(oracle.predict_at(inputs,true)-clean))/sigma)
    REPORT['scored_reference']={'maximum_absolute_error':ref_error,'maximum_refinement_change':refinement,
        'maximum_calibration_bias_in_sigma':cal_bias,'minimum_physical_displacement':float(min(truths[k].min() for k in groups if k!='transit_anchor')),
        'scope':'Independent Maxwell boundary solve plus time-domain centroids and exact global final center-of-energy balance; not a microscopic material-force simulation.'}
    assert ref_error<1e-8 and refinement<1e-8 and cal_bias<.001
    def functions(path):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)}
    assert functions(TASK/'environment/model.py')==functions(BASE/'scripts/pulse_block_displacement_baseline.py')
    rng=np.random.default_rng(metadata['noise_seed']);fits=[]
    max_chi=max_parameter=max_oracle=max_anchor=0.;min_source=float('inf')
    for _ in range(metadata['noise_trials']):
        noisy=clean+rng.normal(0,sigma,len(clean))
        rows=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,noisy)]
        for name,mod in [('oracle',oracle),('shortcut',shortcut)]:
            model=mod.Model().fit(rows);fits.append(model.oscillator_strength)
            chi=float(np.sum(((model.predict(inputs)-noisy)/sigma)**2)/(len(inputs)-1))
            parameter=abs(model.oscillator_strength/true-1)
            max_chi=max(max_chi,chi);max_parameter=max(max_parameter,parameter)
            assert chi<1.5 and parameter<.03
            for k,v in groups.items():
                error=nrmse(model.predict(v),truths[k])
                if name=='oracle':max_oracle=max(max_oracle,error);assert error<.04
                elif k=='transit_anchor':max_anchor=max(max_anchor,error);assert error<.04
                else:min_source=min(min_source,error);assert error>.04
    REPORT['noise']={'trials':metadata['noise_trials'],'all_expected_outcomes':True,'fit_min':min(fits),'fit_max':max(fits),
                     'maximum_calibration_chi2':max_chi,'maximum_parameter_relative_error':max_parameter,
                     'maximum_oracle_hidden_error':max_oracle,'minimum_source_diagnostic_error':min_source,
                     'maximum_source_anchor_error':max_anchor}
    coefficients=oracle.predict_at(settings,1.)-np.array([e['length'] for e in settings])
    max_fit=max_shared=max_cal_ref=0.;gap_min={k:float('inf') for k in groups if k!='transit_anchor'}
    recovery=[]
    for f in np.linspace(.2,.4,41):
        y=oracle.predict_at(settings,f)
        rows=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(settings,y)]
        estimates=[mod.Model().fit(rows).oscillator_strength for mod in (oracle,shortcut)]
        error=max(abs(x-f) for x in estimates);max_fit=max(max_fit,error)
        recovery.append({'truth':float(f),'estimates':estimates,'maximum_error':error})
        max_shared=max(max_shared,float(np.max(abs(y-shortcut.predict_at(settings,f)))))
        max_cal_ref=max(max_cal_ref,float(np.max(abs(y-reference.predict(settings,f)))))
        for k,v in groups.items():
            if k!='transit_anchor':gap_min[k]=min(gap_min[k],nrmse(shortcut.predict_at(v,f),oracle.predict_at(v,f)))
    REPORT['identifiability']={'proof':'For every known pulse, transit=L+f*C, with C=L*energy_average[(1+omega^2)/(omega^2-1)^2]>0. Any single setting globally identifies f, and the weighted objective is strictly convex.',
        'minimum_calibration_slope':float(coefficients.min()),'unit_sigma_information':float(coefficients@coefficients),
        'parameter_samples':41,'recoveries':recovery,'maximum_fit_error':max_fit,'maximum_shared_calibration_error':max_shared,
        'maximum_calibration_reference_error':max_cal_ref,'group_gap_minima':gap_min}
    assert max_fit<1e-12 and coefficients.min()>0 and max_shared<1e-12 and max_cal_ref<1e-8
    assert min(gap_min.values())>.25
    reference_cases=list(product(np.linspace(.2,.4,11),[1.5,1.7],[.04,.1],[.5,2.]))
    random=np.random.default_rng(171059)
    reference_cases += [(random.uniform(.2,.4),random.uniform(1.5,1.7),random.uniform(.04,.1),random.uniform(.5,2.)) for _ in range(48)]
    max_ref=max_refined=0.;reference_rows=[]
    for f,c,w,length in reference_cases:
        e=make_input(c,w,length)
        physical=oracle.predict_at([e],f)[0]
        ref=reference.predict([e],f)[0];fine=reference.predict([e],f,32768)[0]
        error=abs(ref-physical)/physical;change=abs(fine-ref)/physical
        max_ref=max(max_ref,error);max_refined=max(max_refined,change)
        reference_rows.append({'strength':float(f),'input':e,'physical':float(physical),'reference':float(ref),'relative_error':error,'relative_refinement':change})
    REPORT['domain_reference']={'cases':len(reference_rows),'maximum_relative_error':max_ref,'maximum_relative_refinement':max_refined,'rows':reference_rows}
    assert max_ref<1e-7 and max_refined<1e-7
    # Exact continuous bounds apply to all supported spectra, including unsampled controls.
    min_gap=min_value=min_source_value=float('inf');max_quad_change=0.;sweep=[]
    u,weights=leggauss(192);weights*= (1-u*u)**8;weights/=weights.sum()
    for f,c,w,length in product(np.linspace(.2,.4,41),np.linspace(1.5,1.7,5),[.04,.07,.1],[.5,1.,2.]):
        e=make_input(c,w,length);value=oracle.predict_at([e],f)[0];source=shortcut.predict_at([e],f)[0]
        phase,group=oracle.indices(c+w*u,f)
        refined=length*np.dot(weights,group-1)
        error=abs(source/value-1)
        min_gap=min(min_gap,error);min_value=min(min_value,value);min_source_value=min(min_source_value,source)
        max_quad_change=max(max_quad_change,abs(refined-value))
        sweep.append({'strength':float(f),'input':e,'physical':float(value),'source':float(source),'relative_gap':float(error)})
    continuous_gap=115/444
    REPORT['full_range_separation']={'sample_count':len(sweep),'sampled_minimum_gap':min_gap,
        'minimum_physical_signal':float(min_value),'minimum_source_signal':float(min_source_value),
        'continuous_gap_lower_bound':continuous_gap,'maximum_quadrature_refinement_change':max_quad_change,
        'bound_proof':'Set s=omega^2. The point-frequency source/physical ratio is (s-1)/(s+1)+f/(s-1). It increases in f. On s in[1.96,3.24], its minimum at f=.2 is473/888; its maximum at f=.4 is329/444 (the sole interior critical point is a minimum). The finite-band ratio is its positive physical-energy-weighted average.',
        'continuous_physical_lower_bound':.5*.2*4.24/2.24**2,
        'continuous_source_lower_bound':.5*.2/2.24*(1+.2*4.24/2.24**2),'rows':sweep}
    assert min_gap>continuous_gap and min_value>.05 and min_source_value>.05 and max_quad_change<1e-11
    omega=np.linspace(1.4,1.8,201);max_reflection=max_energy=max_oscillator=max_phase=max_derivative=0.
    phase_min=group_min=float('inf');phase_max=0.
    for f,length in product([.2,.3,.4],[.5,1,2]):
        phase,group=oracle.indices(omega,f)
        transmitted,reflected=reference.transmission(omega,f,length)
        max_reflection=max(max_reflection,float(np.max(abs(reflected))))
        max_energy=max(max_energy,float(np.max(abs(abs(transmitted)**2+abs(reflected)**2-1))))
        max_phase=max(max_phase,float(np.max(abs(transmitted-np.exp(1j*omega*phase*length)))))
        # Unit E=H amplitude: vacuum energy plus explicit electric and magnetic oscillator energies.
        electric=f/(1-omega*omega)
        magnetic=f/(1-omega*omega)
        stored=.5+(omega*omega+1)*(electric*electric+magnetic*magnetic)/(4*f)
        max_oscillator=max(max_oscillator,float(np.max(abs(stored-.5*group))))
        h=1e-5
        derivative=((omega+h)*oracle.indices(omega+h,f)[0]-(omega-h)*oracle.indices(omega-h,f)[0])/(2*h)
        max_derivative=max(max_derivative,float(np.max(abs(derivative-group))))
        phase_min=min(phase_min,float(phase.min()));phase_max=max(phase_max,float(phase.max()));group_min=min(group_min,float(group.min()))
    limits=[]
    for c,w,length in [(1.5,.04,.5),(1.6,.07,1.),(1.7,.1,2.)]:
        e=make_input(c,w,length);t=make_input(c,w,length,'transit')
        assert abs(oracle.predict_at([e],0)[0])+abs(shortcut.predict_at([e],0)[0])<1e-14
        assert abs(reference.predict([e],0)[0])<1e-10
        assert abs(oracle.predict_at([t],0)[0]-length)<1e-14
        f=.31;value=oracle.predict_at([e],f)[0]
        assert abs(oracle.predict_at([make_input(c,w,1.5*length)],f)[0]-1.5*value)<1e-12
        # Shrinking bandwidth approaches the corresponding monochromatic predictions.
        values=[]
        for width in [.04,.02,.01]:
            tiny=make_input(c,width,length)
            values.append(float(oracle.predict_at([tiny],f)[0]))
        exact=length*(oracle.indices(np.array([c]),f)[1][0]-1)
        assert all(abs(values[j+1]-exact)<abs(values[j]-exact) for j in range(2))
        limits.append({'center':c,'length':length,'widths':[.04,.02,.01],'physical':values,'monochromatic':float(exact)})
    REPORT['physical_limits']={'maximum_reflection_amplitude':max_reflection,'maximum_scattering_energy_error':max_energy,
        'maximum_matching_phase_error':max_phase,'maximum_explicit_oscillator_energy_error':max_oscillator,
        'maximum_group_derivative_error':max_derivative,'minimum_phase_index':phase_min,'maximum_phase_index':phase_max,
        'minimum_group_index':group_min,'vacuum_and_length_scaling_passed':True,'narrow_band_convergence':limits,
        'scope':'Vacuum strength0 and sub-.04 bandwidth are author limits outside the fitted/scored domain. The explicit oscillator energy check validates passive optical energy transport; it is not a microscopic force simulation.'}
    assert max(max_reflection,max_energy,max_phase,max_oscillator)<1e-12 and max_derivative<2e-9
    assert 0<phase_min<phase_max<1 and group_min>1
    REPORT['local_controls']=local_controls()
    REPORT['source_hashes']={str(path.relative_to(BASE)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [
        TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',BASE/'scripts/pulse_block_displacement_baseline.py',Path(__file__)]}
    REPORT['status']='science_and_local_controls_complete';REPORT['seconds']=time.monotonic()-start
    (RESULTS/'pulse-block-displacement-r1-validation.json').write_text(json.dumps(REPORT,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:REPORT[key] for key in ['status','actual_data','noise','scored_reference','local_controls','seconds']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true')
    try:run(parser.parse_args().generate)
    except Exception:
        REPORT['status']='author_check_failed';REPORT['traceback']=traceback.format_exc()
        (RESULTS/f'author-check-failure-{time.time_ns()}.json').write_text(json.dumps(REPORT,indent=2,allow_nan=False)+'\n')
        raise
