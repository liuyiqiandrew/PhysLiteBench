"""Validate fixed-speed active-particle inference against a two-point-equivalent Gaussian model."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import j0

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/active-trap'


def module(path):
    spec = importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def metrics(model, records, true):
    inputs = [r['input'] for r in records]
    values = np.array([r['value'] for r in records])
    sigma = np.array([r['sigma'] for r in records])
    return {'parameter': model.rotational_diffusion,
            'parameter_relative_error': abs(model.rotational_diffusion/true-1),
            'calibration_chi2': float(np.sum(((model.predict(inputs)-values)/sigma)**2)/(len(records)-1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--noise-trials', type=int, default=256)
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/active-trap-validation/summary.json')
    args = parser.parse_args()
    started = time.time()
    oracle = module(TASK/'solution/model.py')
    shortcut = module(ROOT/'scripts/active_trap_baseline.py')
    reference = module(TASK/'tests/reference.py')
    metadata = json.loads((TASK/'tests/metadata.json').read_text())
    true = reference.TRUE_PARAMETER
    inputs = reference.calibration_inputs()*2
    exact = reference.predict(inputs, true)
    sigma = metadata['measurement_relative_sigma']*np.maximum(abs(exact), metadata['measurement_scale_floor'])
    if args.generate:
        values = exact + np.random.default_rng(metadata['calibration_seed']).normal(0, sigma)
        records = [dict(input=e, value=float(y), sigma=float(s)) for e,y,s in zip(inputs,values,sigma)]
        for folder in ['environment', 'tests']:
            (TASK/folder/'data/calibration.json').write_text(json.dumps(records, indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records] == inputs
    hidden = reference.hidden_inputs()
    truth = {n: reference.predict(es, true) for n,es in hidden.items()}
    def scores(predict):
        return {n: float(np.sqrt(np.mean((predict(es)-truth[n])**2))) for n,es in hidden.items()}
    report = {'revision':1, 'calibration_seed':metadata['calibration_seed'], 'noise_seed':metadata['noise_seed'],
              'noise_trials':args.noise_trials, 'controls':{}}
    for name, source in [('oracle', oracle), ('shortcut', shortcut)]:
        model = source.Model().fit(records)
        result = metrics(model, records, true)
        result['calibration_reference_max_error'] = float(max(abs(np.array([source.predict_one(e,true) for e in inputs])-exact)))
        result['hidden'] = scores(model.predict)
        assert result['calibration_chi2'] < 1.5 and result['parameter_relative_error'] < .03
        assert result['calibration_reference_max_error'] < 1e-10
        assert (max(result['hidden'].values()) < .025 if name=='oracle' else min(result['hidden'].values()) > .025)
        report['controls'][name] = result
    rng = np.random.default_rng(metadata['noise_seed'])
    parameters=[];chi2s=[];noise_scores={'physical_reference':[], 'shortcut':[]}
    for _ in range(args.noise_trials):
        values = exact+rng.normal(0,sigma)
        sample=[dict(input=e,value=float(y),sigma=float(s)) for e,y,s in zip(inputs,values,sigma)]
        good=oracle.Model().fit(sample);bad=shortcut.Model().fit(sample)
        result=metrics(good,sample,true)
        assert abs(good.rotational_diffusion-bad.rotational_diffusion)<1e-12
        parameters.append(good.rotational_diffusion);chi2s.append(result['calibration_chi2'])
        noise_scores['physical_reference'].extend(scores(lambda es:reference.predict(es,good.rotational_diffusion)).values())
        noise_scores['shortcut'].extend(scores(bad.predict).values())
    report['noise']={'parameter_min':min(parameters),'parameter_max':max(parameters),
                     'parameter_relative_error_max':float(max(abs(np.array(parameters)/true-1))),
                     'calibration_chi2_max':max(chi2s),'calibration_pass_fraction':float(np.mean(np.array(chi2s)<1.5)),
                     'hidden':{n:{'min':min(v),'max':max(v)} for n,v in noise_scores.items()},
                     'method':'256 fits and independent moment-reference hidden evaluations at every fitted rate; direct angular-oracle checks at the extrema and across the full allowed domain.'}
    assert report['noise']['parameter_relative_error_max']<.03
    assert report['noise']['calibration_pass_fraction']>=.99
    assert max(noise_scores['physical_reference'])<.025 and min(noise_scores['shortcut'])>.025
    extrema={}
    for name,source in [('oracle',oracle),('shortcut',shortcut)]:
        values=[]
        for rate in [min(parameters),max(parameters)]:
            model=source.Model();model.rotational_diffusion=rate
            values.extend(scores(model.predict).values())
        extrema[name]={'min':min(values),'max':max(values)}
    report['hidden_parameter_extrema_check']=extrema
    report['physical_checks']=physical_checks(oracle,shortcut,reference)
    report['seconds']=time.time()-started
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('active-trap PASS',args.output, 'seconds',report['seconds'])


def physical_checks(oracle, shortcut, reference):
    points=[]
    for rate in [.4,.65,1.1]:
        for trap in [.4,1.2,2.5]:
            for argument in [0.,3.1,7.]:
                speed=2.4
                points.append((rate,{'readout':'trap_fourier','trap_rate':trap,'speed':speed,
                                     'diffusion':.012,'wavenumber':argument*trap/speed}))
    independent=[];angular=[];series=[];normalization=[];bounds=[]
    for rate,e in points:
        a=oracle.characteristic(e,rate)
        b=reference.characteristic(e,rate)
        independent.append(abs(a-b));bounds.append(abs(a))
        series.append(abs(reference.characteristic(e,rate,48)-b))
        if e['wavenumber']:
            angular.append(abs(oracle.characteristic(e,rate,modes=28,cutoff=28,tolerance=2e-11)-a))
        else:normalization.append(abs(a-1))
    passive=[];persistent=[];variance=[]
    for trap in [.4,1.3,2.5]:
        e={'readout':'trap_fourier','trap_rate':trap,'speed':0.,'diffusion':.07,'wavenumber':2.7}
        passive.append(abs(oracle.characteristic(e,.65)-np.exp(-.07*2.7**2/(2*trap))))
        e.update(speed=1.7,wavenumber=4.*trap/1.7)
        limit=j0(4.)*np.exp(-.07*e['wavenumber']**2/(2*trap))
        persistent.append(abs(oracle.characteristic(e,0.)-limit))
        m=reference.radial_moments(trap,.65,2)
        exact=.07/trap+.5*(1.7/trap)**2*m[1]
        e['readout']='trap_variance'
        variance.append(abs(exact-oracle.second_moment(e,.65)))
    # Characteristic functions must define positive-semidefinite Fourier Gram matrices.
    e={'readout':'trap_fourier','trap_rate':1.2,'speed':1.7,'diffusion':.012,'wavenumber':0.}
    qs=np.linspace(0,4.,13)
    gram=np.array([[reference.characteristic(dict(e,wavenumber=float(abs(a-b))),.65) for b in qs] for a in qs])
    gram_min=float(np.linalg.eigvalsh(gram)[0])
    assert max(independent)<2e-8 and max(angular)<2e-8 and max(series)<1e-11
    assert max(passive)<1e-12 and max(persistent)<2e-8 and max(variance)<1e-12
    assert max(normalization)<1e-12 and max(bounds)<=1+1e-9 and gram_min>-1e-10
    return {'oracle_reference_full_domain_error':max(independent),'angular_20_vs_28_error':max(angular),
            'moment_36_vs_48_error':max(series),'probability_normalization_error':max(normalization),
            'characteristic_absolute_max':max(bounds),'fourier_gram_min_eigenvalue':gram_min,
            'passive_gaussian_limit_error':max(passive),'zero_rotation_ring_bessel_limit_error':max(persistent),
            'exact_second_moment_error':max(variance),'omitted_propulsion_history_bound':7*np.exp(-25),
            'both_controls_have_identical_propulsion_covariance':'<v_i(t) v_j(0)> = delta_ij speed^2 exp(-rotational_diffusion*|t|)/2'}


if __name__=='__main__':main()
