"""Coupled flow/current controls, independent collocation, and noise validation."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK='electroviscous-flow'


def module(path):
    spec=importlib.util.spec_from_file_location('check_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--noise-trials',type=int,default=256)
    parser.add_argument('--output',type=Path,default=ROOT/'jobs'/f'{TASK}-validation'/'summary.json')
    args=parser.parse_args();task=ROOT/'tasks'/TASK
    ref=module(task/'tests/reference.py');oracle=module(task/'solution/model.py');shortcut=module(ROOT/'scripts/electroviscous_flow_baseline.py')
    meta=json.loads((task/'tests/metadata.json').read_text());true=meta['true_parameter'];limit=meta['prediction_limit']
    inputs=ref.calibration_inputs();noiseless=ref.predict(inputs,true);sigma=.004*max(abs(noiseless))
    def records_for(values):return [dict(input=e,value=float(v),sigma=float(sigma)) for e,v in zip(inputs,values)]
    if args.generate:
        values=noiseless+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs))
        data=json.dumps(records_for(values),indent=2)+'\n'
        for folder in ['environment','tests']:(task/folder/'data/calibration.json').write_text(data)
    public=task/'environment/data/calibration.json';private=task/'tests/data/calibration.json'
    assert public.read_bytes()==private.read_bytes()
    records=json.loads(public.read_text());assert [r['input'] for r in records]==inputs
    hidden=ref.hidden_inputs();truth={k:ref.predict(v,true) for k,v in hidden.items()}
    def scores(model):return {k:float(np.sqrt(np.mean((model.predict(hidden[k])-v)**2)/np.mean(v**2))) for k,v in truth.items()}
    def calibration(model,records):
        residual=(model.predict(inputs)-np.array([r['value'] for r in records]))/sigma
        return dict(parameter=model.viscosity,parameter_relative_error=abs(model.viscosity/true-1),calibration_chi2=float(residual@residual)/(len(records)-1))
    report=dict(revision=meta['revision'],calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],noise_trials=args.noise_trials,measurement_sigma=float(sigma),controls={})
    exact=oracle.Model();exact.viscosity=true
    base=shortcut.Model();base.viscosity=true
    cal_error=float(max(abs(exact.predict(inputs)-base.predict(inputs))));assert cal_error<1e-16
    for name,cls in [('oracle',oracle.Model),('shortcut',shortcut.Model)]:
        model=cls().fit(records);result=calibration(model,records);result['hidden']=scores(model)
        assert result['parameter_relative_error']<.03 and result['calibration_chi2']<1.5
        assert (max(result['hidden'].values())<limit if name=='oracle' else min(result['hidden'].values())>limit)
        report['controls'][name]=result
    experiments=[e for group in hidden.values() for e in group]
    expected=ref.predict(experiments,true);actual=exact.predict(experiments)
    reference_error=float(max(abs(actual-expected))/max(abs(expected)))
    refined=ref.predict(experiments,true,tolerance=2e-10)
    refinement=float(max(abs(refined-expected))/max(abs(expected)))
    assert reference_error<1e-6 and refinement<1e-6
    x=oracle.NODES;w=oracle.WEIGHTS;eta=true
    neutrality=[];reciprocity=[];feedback=[];currents=[];powers=[];weak_current=[];positive_baseline=[];ratios=[]
    for group in hidden.values():
        e=group[-3];h=e['half_gap']*1e-6;c=e['concentration'];g=e['pressure_gradient'];charge=e['surface_charge']
        solution=oracle.equilibrium(e['half_gap'],c,charge)
        psi,dpsi=solution(x);wall=solution(1.)[0];rho=-2*oracle.FARADAY*c*np.sinh(psi)
        sigma_local=2*oracle.FARADAY**2*c*oracle.DIFFUSIVITY/(oracle.GAS_CONSTANT*oracle.TEMPERATURE)*np.cosh(psi)
        up=g*h*h*(1-x*x)/(2*eta);ue=oracle.PERMITTIVITY*oracle.THERMAL_VOLTAGE*(psi-wall)/eta
        conductance=w@sigma_local;ip=w@(rho*up);additional=w@(rho*ue);field=-ip/(conductance+additional)
        u=up+ue*field;du=-g*h*h*x/eta+oracle.PERMITTIVITY*oracle.THERMAL_VOLTAGE*dpsi*field/eta
        neutrality.append(abs((w@rho)*h+charge)/abs(charge))
        reciprocity.append(abs(ip/g-w@ue)/abs(ip/g))
        gradient_conductance=oracle.PERMITTIVITY**2/eta*w@((oracle.THERMAL_VOLTAGE*dpsi/h)**2)
        feedback.append(abs(additional-gradient_conductance)/additional)
        currents.append(abs(w@(rho*u+sigma_local*field))/abs(ip))
        dissipated=eta/h**2*(w@(du*du))+(w@sigma_local)*field*field
        powers.append(abs(g*(w@u)-dissipated)/(g*(w@u)))
        approximate_field=-ip/conductance
        weak_current.append(abs(w@(rho*(up+ue*approximate_field)+sigma_local*approximate_field))/abs(ip))
        positive_baseline.append(float(w@(up+ue*approximate_field)/(w@up)))
        ratios.append(float(additional/conductance))
        flipped=dict(e,surface_charge=-charge);reversed_g=dict(e,pressure_gradient=-g)
        assert abs(exact.predict([e])[0]-exact.predict([flipped])[0])<1e-15
        assert abs(exact.predict([e])[0]+exact.predict([reversed_g])[0])<1e-15
    assert max(neutrality)<1e-8 and max(reciprocity)<1e-8 and max(feedback)<1e-8
    assert max(currents)<1e-12 and max(powers)<1e-8 and min(positive_baseline)>0
    # Public-domain corners retain physical current closure and positive conductance.
    corner_errors=[]
    for h in [.2,.45]:
        for c in [.006,.03]:
            for q in [-.00035,.00035]:
                e=ref.reading(1e5,h,c,q)
                for eta_corner in [.0008,.002]:
                    model=oracle.Model();model.viscosity=eta_corner
                    a=model.predict([e])[0];b=ref.predict([e],eta_corner)[0]
                    assert a>0 and b>0
                    corner_errors.append(abs(a/b-1))
    report['physical_checks']=dict(calibration_shortcut_equivalence=cal_error,oracle_reference_relative_error=reference_error,reference_refinement_relative_error=refinement,
        transverse_charge_balance_relative_error=max(neutrality),hydraulic_electric_reciprocity_relative_error=max(reciprocity),feedback_conductance_gradient_identity_relative_error=max(feedback),
        open_circuit_current_relative_error=max(currents),pressure_power_dissipation_relative_error=max(powers),shortcut_uncancelled_current_fraction_min=min(weak_current),
        shortcut_positive_mean_flow_fraction_min=min(positive_baseline),feedback_to_ohmic_conductance=ratios,parameter_geometry_corner_relative_error=max(corner_errors))
    rng=np.random.default_rng(meta['noise_seed']);parameters=[];chi2=[];errors=[]
    for _ in range(args.noise_trials):
        sample=records_for(noiseless+rng.normal(0,sigma,len(inputs)))
        model=oracle.Model().fit(sample);result=calibration(model,sample)
        assert abs(shortcut.Model().fit(sample).viscosity-model.viscosity)<1e-15
        parameters.append(model.viscosity);chi2.append(result['calibration_chi2']);errors.append(result['parameter_relative_error'])
    sensitivity={}
    for name,cls in [('oracle',oracle.Model),('shortcut',shortcut.Model)]:
        values=[]
        for eta in [min(parameters),max(parameters)]:
            model=cls();model.viscosity=eta;values.extend(scores(model).values())
        sensitivity[name]=dict(min=min(values),max=max(values))
    assert max(errors)<.03 and np.mean(np.array(chi2)<1.5)>=.99
    assert sensitivity['oracle']['max']<limit and sensitivity['shortcut']['min']>limit
    report['noise']=dict(parameter_min=min(parameters),parameter_max=max(parameters),parameter_relative_error_max=max(errors),calibration_chi2_max=max(chi2),calibration_pass_fraction=float(np.mean(np.array(chi2)<1.5)),hidden_at_parameter_extrema=sensitivity)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(TASK,'PASS',args.output)


if __name__=='__main__':main()
