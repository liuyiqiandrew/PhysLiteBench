"""Author scientific checks and isolated local controls; no model runs."""
import argparse
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
import numpy as np
from scipy.special import expit, xlogy

BASE=Path(__file__).resolve().parents[1]
TASK=BASE/'tasks/fixed-number-band'
RESULTS=BASE/'results'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


oracle=load('band_oracle',TASK/'solution/model.py')
shortcut=load('band_shortcut',BASE/'scripts/fixed_number_band_baseline.py')
reference=load('band_reference',TASK/'tests/reference.py')
metadata=json.loads((TASK/'tests/metadata.json').read_text())


def nrmse(value,truth):
    return float(np.sqrt(np.mean((value-truth)**2)/np.mean(truth**2)))


def derivative(function,temperature,fraction=1e-3):
    h=temperature*fraction
    a,b,c,d=[function(temperature+j*h) for j in [-2,-1,1,2]]
    return (a-8*b+8*c-d)/(12*h)


def local_controls():
    report={}
    for name,path in [('oracle',TASK/'solution/model.py'),('shortcut',BASE/'scripts/fixed_number_band_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='fixed-number-band-') as folder:
            location=Path(folder)
            shutil.copytree(TASK/'environment',location/'app')
            shutil.copytree(TASK/'tests',location/'tests')
            shutil.copy2(path,location/'app/model.py')
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(location/'app'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.monotonic()
            result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(location/'app/test_public.py'),str(location/'tests/test_hidden.py')],cwd=location,env=env,capture_output=True,text=True,timeout=60)
            report[name]={'returncode':result.returncode,'seconds':time.monotonic()-start,'stdout':result.stdout,'stderr':result.stderr}
            if name=='oracle':assert result.returncode==0 and '7 passed' in result.stdout
            else:assert result.returncode==1 and '3 failed, 4 passed' in result.stdout
    (RESULTS/'fixed-number-band-r1-local-controls.json').write_text(json.dumps(report,indent=2)+'\n')
    return {name:{k:v for k,v in row.items() if k not in ('stdout','stderr')} for name,row in report.items()}


def run(generate=False):
    start=time.monotonic();true=reference.TRUE_PARAMETER
    inputs=reference.calibration_inputs()*metadata['calibration_repeats']
    clean=reference.predict(inputs,true);sigma=metadata['measurement_sigma']
    if generate:
        y=clean+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(inputs))
        rows=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,y)]
        text=json.dumps(rows,indent=2)+'\n'
        for side in ('environment','tests'):(TASK/side/'data/calibration.json').write_text(text)
    rows=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert [r['input'] for r in rows]==inputs and len(rows)==288
    assert all(r['sigma']==sigma for r in rows)
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    report={'revision':1,'status':'science_and_local_controls_complete','model_evaluations':0,
            'true_width':true,'calibration_seed':metadata['calibration_seed'],'noise_seed':metadata['noise_seed'],
            'calibration_count':len(rows),'distinct_settings':len(reference.calibration_inputs()),'sigma':sigma,'prediction_limit':.04}
    groups=reference.hidden_inputs();truths={k:reference.predict(v,true) for k,v in groups.items()}
    actual={}
    for name,mod in [('oracle',oracle),('shortcut',shortcut)]:
        model=mod.Model().fit(rows)
        residual=(model.predict(inputs)-np.array([r['value'] for r in rows]))/sigma
        actual[name]={'width':model.width,'parameter_relative_error':abs(model.width/true-1),
                      'calibration_chi2':float(residual@residual/(len(rows)-1)),
                      'hidden':{k:nrmse(model.predict(v),truths[k]) for k,v in groups.items()}}
        assert actual[name]['parameter_relative_error']<.03 and actual[name]['calibration_chi2']<1.5
    report['actual_data']=actual
    hidden_error=hidden_refinement=rounding=0.
    for name,experiments in groups.items():
        hidden_error=max(hidden_error,float(np.max(abs(oracle.predict_at(experiments,true)-truths[name]))))
        hidden_refinement=max(hidden_refinement,float(np.max(abs(reference.predict(experiments,true,16000)-truths[name]))))
        for e in experiments:
            for m in (8000,16000,32000,64000):rounding=max(rounding,abs(reference.canonical(true,e['temperature'],e['filling'],m)[2]-e['filling']))
    report['scored_reference']={'max_absolute_error':hidden_error,'max_larger_size_change':hidden_refinement,
                                'max_filling_rounding':rounding,'minimum_signal':float(min(x.min() for x in truths.values())),
                                'calibration_bias_sigma':float(np.max(abs(oracle.predict_at(inputs,true)-clean))/sigma)}
    assert hidden_error<1e-6 and hidden_refinement<1e-6 and rounding<1e-15
    assert report['scored_reference']['calibration_bias_sigma']<.01
    rng=np.random.default_rng(metadata['noise_seed'])
    max_chi=max_parameter=max_oracle=max_anchor=0.;min_shortcut=1e9;widths=[]
    for _ in range(metadata['noise_trials']):
        noisy=clean+rng.normal(0,sigma,len(clean))
        records=[{'input':e,'value':float(y),'sigma':sigma} for e,y in zip(inputs,noisy)]
        for name,mod in [('oracle',oracle),('shortcut',shortcut)]:
            model=mod.Model().fit(records);widths.append(model.width)
            chi=float(np.sum(((model.predict(inputs)-noisy)/sigma)**2)/(len(inputs)-1))
            parameter=abs(model.width/true-1)
            assert chi<1.5 and parameter<.03
            max_chi=max(max_chi,chi);max_parameter=max(max_parameter,parameter)
            for group,experiments in groups.items():
                error=nrmse(model.predict(experiments),truths[group])
                if name=='oracle':max_oracle=max(max_oracle,error);assert error<.04
                elif group=='half_filling_anchor':max_anchor=max(max_anchor,error);assert error<.04
                else:min_shortcut=min(min_shortcut,error);assert error>.04
    report['noise']={'trials':metadata['noise_trials'],'all_expected_outcomes':True,'width_min':min(widths),'width_max':max(widths),
                     'max_calibration_chi2':max_chi,'max_parameter_relative_error':max_parameter,'max_oracle_hidden_error':max_oracle,
                     'min_shortcut_diagnostic_error':min_shortcut,'max_shortcut_anchor_error':max_anchor}
    # Analytic monotonicity establishes uniqueness throughout the fitted interval.
    max_recovery=max_calibration_difference=max_cal_formula=0.;max_slope=-1e9;min_gap=1e9
    temperatures=np.array([e['temperature'] for e in reference.calibration_inputs()])
    for width in np.linspace(.8,1.2,41):
        y=oracle.predict_at(inputs,width)
        max_calibration_difference=max(max_calibration_difference,float(np.max(abs(y-shortcut.predict_at(inputs,width)))))
        records=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,y)]
        for mod in (oracle,shortcut):max_recovery=max(max_recovery,abs(mod.Model().fit(records).width-width))
        x=width/(4*temperatures);formula=x*x/np.cosh(x)**2
        slope=2*formula/width*(1-x*np.tanh(x));max_slope=max(max_slope,float(slope.max()))
        max_cal_formula=max(max_cal_formula,float(np.max(abs(oracle.predict_at(reference.calibration_inputs(),width)-formula))))
        for group,experiments in groups.items():
            if group!='half_filling_anchor':min_gap=min(min_gap,nrmse(shortcut.predict_at(experiments,width),oracle.predict_at(experiments,width)))
    report['identifiability']={'proof':'At half filling C=x^2*sech(x)^2, x=W/(4T)>=4/3. dC/dW=2*C/W*(1-x*tanh(x))<0 throughout calibration domain.',
                               'width_samples':41,'max_noiseless_width_error':max_recovery,'least_negative_derivative':max_slope,
                               'max_half_filling_formula_error':max_cal_formula,'max_calibration_model_difference':max_calibration_difference,
                               'min_diagnostic_group_gap_over_width':min_gap}
    assert max_recovery<1e-7 and max_slope<0 and max_calibration_difference<1e-12 and min_gap>.04
    # Full continuous-domain response versus fixed-filling temperature derivative;
    # also validate that the supplied source really is the fixed-mu heat response.
    random=np.random.default_rng(152043)
    controls=list(product([.8,1.2],[.1,.6],[.1,.5,.9]))
    controls += [tuple(random.uniform([.8,.1,.1],[1.2,.6,.9])) for _ in range(48)]
    max_derivative=max_derivative_refine=max_source_derivative=max_hole=max_number=0.;min_physical=1.;min_difference=1.
    for width,temp,n in controls:
        e={'temperature':temp,'filling':n}
        physical=oracle.predict_at([e],width)[0];source=shortcut.predict_at([e],width)[0]
        energies,mu,f=oracle.state(width,temp,n)
        def energy(t):
            levels,_,occupations=oracle.state(width,t,n)
            return float(np.mean(levels*occupations))
        coarse=derivative(energy,temp);fine=derivative(energy,temp,5e-4)
        max_derivative=max(max_derivative,abs(fine-physical));max_derivative_refine=max(max_derivative_refine,abs(fine-coarse))
        def entropy(t):
            occ=expit((mu-energies)/t)
            return float(-np.mean(xlogy(occ,occ)+xlogy(1-occ,1-occ)))
        max_source_derivative=max(max_source_derivative,abs(temp*derivative(entropy,temp,5e-4)-source))
        reflection={'temperature':temp,'filling':1-n}
        for mod,value in [(oracle,physical),(shortcut,source)]:max_hole=max(max_hole,abs(mod.predict_at([reflection],width)[0]-value))
        max_number=max(max_number,abs(np.mean(f)-n));min_physical=min(min_physical,physical);min_difference=min(min_difference,source-physical)
    report['full_domain_response']={'cases':len(controls),'max_fixed_filling_energy_derivative_error':max_derivative,
                                    'max_derivative_step_change':max_derivative_refine,'max_fixed_mu_entropy_derivative_error':max_source_derivative,
                                    'max_particle_hole_error':max_hole,'max_filling_residual':max_number,
                                    'minimum_physical_capacity':min_physical,'minimum_source_minus_physical':min_difference}
    assert max(max_derivative,max_derivative_refine,max_source_derivative)<1e-8
    assert min_physical>0 and min_difference>-1e-12
    # Independent finite canonical sums: rational fillings preserve N exactly at
    # every size. Additional finite-energy derivative tests verify normalization.
    rational=list(product([.8,1.2],[.1,.25,.6],[.1,.3,.5,.7,.9]))
    max_canonical=max_canonical_refine=max_canonical_derivative=0.;finite_errors=np.zeros(4)
    for width,temp,n in rational:
        e={'temperature':temp,'filling':n};physical=oracle.predict_at([e],width)[0]
        capacities=np.array([reference.canonical(width,temp,n,m)[1] for m in [8000,16000,32000,64000]])
        finite_errors=np.maximum(finite_errors,abs(capacities-physical))
        coarse=reference.predict([e],width)[0];fine=reference.predict([e],width,16000)[0]
        max_canonical=max(max_canonical,abs(coarse-physical));max_canonical_refine=max(max_canonical_refine,abs(fine-coarse))
        def finite_energy(t):return reference.canonical(width,t,n,4000)[0]
        d=derivative(finite_energy,temp)
        max_canonical_derivative=max(max_canonical_derivative,abs(d-reference.canonical(width,temp,n,4000)[1]))
        for m in [8000,16000,32000,64000]:assert abs(reference.canonical(width,temp,n,m)[2]-n)<1e-15
    report['canonical_domain']={'cases':len(rational),'band_sizes':[8000,16000,32000,64000],
                                'max_finite_size_errors':finite_errors.tolist(),'max_base_extrapolation_error':max_canonical,
                                'max_larger_size_change':max_canonical_refine,'max_fixed_N_energy_derivative_error':max_canonical_derivative,
                                'rounding':'All these rational fillings, the scored inputs and calibration have exactly integer N at every tested size.'}
    assert max_canonical<1e-6 and max_canonical_refine<1e-6 and max_canonical_derivative<1e-7
    # Analytic independent example and high/low temperature limits, outside scored range.
    width=np.log(9.);ex={'temperature':1.,'filling':.3}
    exact=9*np.log(9.)**2/272;open_value=9*np.log(9.)**2/200
    example_error=max(abs(oracle.predict_at([ex],width)[0]-exact),abs(shortcut.predict_at([ex],width)[0]-open_value))
    high=[];low=[]
    for n in [.1,.3,.5,.7,.9]:
        errors=[]
        for t in [20.,40.,80.]:
            e={'temperature':t,'filling':n};physical=oracle.predict_at([e],1.)[0];source=shortcut.predict_at([e],1.)[0]
            errors.append((abs(physical*t*t-n*(1-n)/4),abs(source-n*(1-n)*np.log(n/(1-n))**2)))
        assert all(errors[i+1][0]<errors[i][0] and errors[i+1][1]<errors[i][1] for i in (0,1))
        high.append({'filling':n,'error_pairs':errors})
        values=oracle.predict_at([{'temperature':t,'filling':n} for t in [.08,.06,.04]],1.)
        assert values[0]>values[1]>values[2]>0
        low.append({'filling':n,'capacities':values.tolist()})
    report['limits']={'analytic_example_max_error':example_error,'high_temperature':high,'low_temperature':low,
                      'qualification':'These limit controls are author-only checks, outside the declared task temperature range.'}
    assert example_error<1e-12
    report['local_controls']=local_controls();report['seconds']=time.monotonic()-start
    report['source_hashes']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',BASE/'scripts/fixed_number_band_baseline.py',Path(__file__)]}
    (RESULTS/'fixed-number-band-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true')
    run(parser.parse_args().generate)
