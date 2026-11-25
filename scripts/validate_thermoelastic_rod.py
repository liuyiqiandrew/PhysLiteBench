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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs'/(TASK+'-r3-validation')/'summary.json')
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
    report=dict(revision=meta['revision'],calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],noise_trials=args.noise_trials,measurement_sigma=float(sigma),controls={})
    for name,cls in [('oracle',oracle),('shortcut',shortcut)]:
        exact=cls();exact.conductivity=true
        assert max(abs(exact.predict(inputs)-noiseless))<1e-4
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
    baseline=shortcut();baseline.conductivity=true
    calibration_equivalence=float(max(abs(exact.predict(inputs)-baseline.predict(inputs))))
    assert calibration_equivalence<1e-12
    x=oracle_module.LENGTH*(np.arange(512)+.5)/512
    basis=np.sqrt(2)*np.cos(np.arange(oracle_module.MODES+1)[:,None]*np.pi*x[None,:]/oracle_module.LENGTH)
    basis[0]=1.
    alpha=oracle_module.expansion(x)
    elastic=oracle_module.REFERENCE_TEMPERATURE*oracle_module.YOUNG
    alpha_modes=basis[1:]@alpha/len(x)
    metric,mechanical,coupling,mobility=oracle_module.material_operators()
    weights=np.r_[oracle_module.VOLUME*oracle_module.HEAT_FIXED_STRAIN,
                  np.zeros(oracle_module.MODES),elastic*oracle_module.VOLUME*alpha_modes,6.]
    capacity_uniform=oracle_module.VOLUME*(oracle_module.HEAT_FIXED_STRAIN+elastic*np.var(alpha))
    def full_state(e,time):
        rates,vectors,mass=oracle_module.spectrum(true,e['contact'])
        initial=oracle_module.initial_state(e)
        return vectors@(np.exp(-rates*time)*(vectors.T@(mass@initial)))
    energies=[];long_time=[];availability_increases=[];spectral_refinement=[]
    for es in hidden.values():
        e=es[0];initial=oracle_module.initial_state(e)
        initial_energy=weights@initial
        last=.5*initial@metric@initial
        for time in [0.,.3,2.,10.,80.]:
            state=full_state(e,time)
            energies.append(abs(weights@state-initial_energy))
            availability=.5*state@metric@state
            availability_increases.append(availability-last);last=availability
        end=full_state(e,3000.)
        if e['contact']:
            long_time.extend(abs(end[[0,-1]]-initial_energy/(capacity_uniform+6.)))
        else:
            long_time.append(abs(end[0]-(initial_energy-6.*e['bath_initial'])/capacity_uniform))
            long_time.append(abs(end[-1]-e['bath_initial']))
        refined=[]
        for e in es:
            rates,vectors,mass=oracle_module.spectrum(true,e['contact'],128)
            initial=oracle_module.initial_state(e,128)
            state=vectors@(np.exp(-rates*e['time'])*(vectors.T@(mass@initial)))
            vals={'mean':state[0],'first':state[1]/np.sqrt(2),'second':state[2]/np.sqrt(2),'bath':state[-1]}
            refined.append(vals[e['observable']])
        spectral_refinement.append(float(max(abs(exact.predict(es)-refined))))
    assert max(energies)<1e-7 and max(long_time)<1e-7 and max(availability_increases)<1e-10
    assert max(spectral_refinement)<.002
    # A low-mode temperature/strain snapshot probes local total-stress uniformity.
    theta=np.zeros(oracle_module.MODES+1);theta[:3]=[.2,.1,.35]
    strain=np.zeros(oracle_module.MODES);strain[:3]=[.0001,.0003,-.0002]
    force=coupling@theta-strain
    stress_errors={};dissipation_errors={};eigen_min={}
    for name,mod in [('oracle',oracle_module),('shortcut',module(ROOT/'scripts/thermoelastic_rod_baseline.py'))]:
        _,_,_,mob=mod.material_operators()
        strain_rate=oracle_module.YOUNG*mob@force
        theta_x=theta@basis;strain_x=strain@basis[1:];rate_x=strain_rate@basis[1:]
        stress=oracle_module.YOUNG*(strain_x-alpha*theta_x)+oracle_module.mechanical_viscosity(x)*rate_x
        stress_errors[name]=float(np.std(stress)/(oracle_module.YOUNG*np.linalg.norm(force)))
        eigen_min[name]=float(np.linalg.eigvalsh(mob).min())
        # Each model is passive in its own positive mobility law.
        power=-oracle_module.YOUNG**2*oracle_module.VOLUME*force@mob@force
        mechanical_derivative=oracle_module.YOUNG*oracle_module.VOLUME*(-force)@strain_rate
        dissipation_errors[name]=float(abs(power-mechanical_derivative)/abs(power))
        assert eigen_min[name]>0 and power<0 and dissipation_errors[name]<1e-12
    assert stress_errors['oracle']<1e-5 and stress_errors['shortcut']>.01
    assert abs(np.mean(rate_x))<1e-12
    report['physical_checks']=dict(finite_volume_refinement_max_difference=convergence,oracle_reference_max_error=agreement,spectral_refinement_max_difference=max(spectral_refinement),closed_linear_energy_max_error=float(max(energies)),equilibrium_max_error=float(max(long_time)),availability_increase_max=float(max(availability_increases)),calibration_shortcut_equivalence=calibration_equivalence,total_stress_relative_nonuniformity=stress_errors,mobility_minimum_eigenvalues=eigen_min,mechanical_dissipation_relative_errors=dissipation_errors,uniform_rod_capacity=capacity_uniform)
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
