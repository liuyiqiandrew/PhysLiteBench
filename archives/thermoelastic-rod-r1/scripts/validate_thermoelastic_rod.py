"""Independent thermoelastic constraints, controls, and calibration-noise validation."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = 'thermoelastic-rod'


def module(path):
    spec = importlib.util.spec_from_file_location('check_'+str(abs(hash(str(path)))), path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def metrics(model, records, true):
    residual = (model.predict([r['input'] for r in records])-np.array([r['value'] for r in records]))/np.array([r['sigma'] for r in records])
    return dict(parameter=model.conductivity, parameter_relative_error=abs(model.conductivity/true-1),
                calibration_chi2=float(residual@residual)/(len(records)-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--noise-trials', type=int, default=256)
    parser.add_argument('--output', type=Path, default=ROOT/'jobs'/(TASK+'-validation')/'summary.json')
    args = parser.parse_args()
    task = ROOT/'tasks'/TASK
    reference = module(task/'tests/reference.py')
    oracle_module = module(task/'solution/model.py'); oracle=oracle_module.Model
    shortcut=module(ROOT/'scripts/thermoelastic_rod_baseline.py').Model
    meta=json.loads((task/'tests/metadata.json').read_text()); true=meta['true_parameter']
    inputs=reference.calibration_inputs()*2
    noiseless=reference.predict(inputs,true)
    sigma=meta['measurement_sigma_fraction']*max(abs(noiseless))
    def records_for(seed):
        values=noiseless+np.random.default_rng(seed).normal(0,sigma,len(inputs))
        return [dict(input=e,value=float(y),sigma=float(sigma)) for e,y in zip(inputs,values)]
    if args.generate:
        data=json.dumps(records_for(meta['calibration_seed']),indent=2)+'\n'
        for folder in ['environment','tests']:(task/folder/'data/calibration.json').write_text(data)
    public,private=task/'environment/data/calibration.json',task/'tests/data/calibration.json'
    assert public.read_bytes()==private.read_bytes()
    records=json.loads(public.read_text()); assert [r['input'] for r in records]==inputs
    hidden=reference.hidden_inputs(); truths={k:reference.predict(v,true) for k,v in hidden.items()}
    def scores(model):
        return {k:float(np.sqrt(np.mean((model.predict(hidden[k])-y)**2))) for k,y in truths.items()}
    report=dict(revision=1,calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],noise_trials=args.noise_trials,measurement_sigma=float(sigma),controls={})
    for name,cls in [('oracle',oracle),('shortcut',shortcut)]:
        exact=cls();exact.conductivity=true
        assert max(abs(exact.predict(inputs)-noiseless))<1e-12
        model=cls().fit(records); result=metrics(model,records,true);result['hidden']=scores(model)
        assert result['parameter_relative_error']<.03 and result['calibration_chi2']<1.5
        assert (max(result['hidden'].values())<meta['prediction_limit'] if name=='oracle' else min(result['hidden'].values())>meta['prediction_limit'])
        report['controls'][name]=result
    exact=oracle();exact.conductivity=true
    all_inputs=[e for es in hidden.values() for e in es]
    ref128=reference.predict(all_inputs,true)
    ref256=reference.predict(all_inputs,true,cells=256)
    prediction=exact.predict(all_inputs)
    convergence=float(max(abs(ref128-ref256))); agreement=float(max(abs(prediction-ref256)))
    assert convergence<.001 and agreement<.002
    energies=[]; long_time=[]
    for es in hidden.values():
        e=es[0]
        initial_energy=5.*e['mean']+6.*e['bath_initial']
        for time in [0.,.3,2.,10.,65.]:
            mean,bath=exact.predict([dict(e,observable='mean',time=time),dict(e,observable='bath',time=time)])
            energies.append(abs(5*mean+6*bath-initial_energy))
        end=exact.predict([dict(e,observable='mean',time=2000.),dict(e,observable='bath',time=2000.)])
        long_time.extend(abs(end-initial_energy/11.))
    assert max(energies)<1e-8 and max(long_time)<1e-8
    report['physical_checks']=dict(finite_volume_refinement_max_difference=convergence,oracle_reference_max_error=agreement,closed_energy_max_error=float(max(energies)),equilibrium_max_error=float(max(long_time)))
    rng=np.random.default_rng(meta['noise_seed']); parameters=[]; chi2=[]; errors=[]
    for i in range(args.noise_trials):
        values=noiseless+rng.normal(0,sigma,len(inputs))
        sample=[dict(input=e,value=float(y),sigma=float(sigma)) for e,y in zip(inputs,values)]
        model=oracle().fit(sample); m=metrics(model,sample,true)
        parameters.append(model.conductivity);chi2.append(m['calibration_chi2']);errors.append(m['parameter_relative_error'])
        if i in [0,args.noise_trials-1]:assert abs(shortcut().fit(sample).conductivity-model.conductivity)<1e-6
    sensitivity={}
    for name,cls in [('oracle',oracle),('shortcut',shortcut)]:
        values=[]
        for p in [min(parameters),max(parameters)]:
            model=cls();model.conductivity=p;values.extend(scores(model).values())
        sensitivity[name]=dict(min=min(values),max=max(values))
    assert max(errors)<.03 and np.mean(np.array(chi2)<1.5)>=.99
    assert sensitivity['oracle']['max']<meta['prediction_limit'] and sensitivity['shortcut']['min']>meta['prediction_limit']
    report['noise']=dict(parameter_min=min(parameters),parameter_max=max(parameters),parameter_relative_error_max=max(errors),calibration_chi2_max=max(chi2),calibration_pass_fraction=float(np.mean(np.array(chi2)<1.5)),hidden_at_parameter_extrema=sensitivity)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(TASK,'PASS',args.output)


if __name__=='__main__':main()
