"""Validate insulating-duct current closure, independent discretizations, and noise."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import splu

ROOT=Path(__file__).resolve().parents[1]
TASK='insulating-mhd-duct'


def module(path):
    spec=importlib.util.spec_from_file_location('check_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def metrics(model,records,true):
    residual=(model.predict([r['input'] for r in records])-np.array([r['value'] for r in records]))/np.array([r['sigma'] for r in records])
    return dict(parameter=model.viscosity,parameter_relative_error=abs(model.viscosity/true-1),calibration_chi2=float(residual@residual)/(len(records)-1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--noise-trials',type=int,default=256)
    parser.add_argument('--output',type=Path,default=ROOT/'jobs'/(TASK+'-r2-validation')/'summary.json')
    args=parser.parse_args();task=ROOT/'tasks'/TASK
    reference=module(task/'tests/reference.py');om=module(task/'solution/model.py');oracle=om.Model
    shortcut=module(ROOT/'scripts/insulating_mhd_duct_baseline.py').Model
    meta=json.loads((task/'tests/metadata.json').read_text());true=meta['true_parameter']
    inputs=reference.calibration_inputs()*2;noiseless=reference.predict(inputs,true)
    sigma=meta['measurement_sigma_fraction']*max(abs(noiseless))
    if args.generate:
        y=noiseless+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs))
        data=json.dumps([dict(input=e,value=float(v),sigma=float(sigma)) for e,v in zip(inputs,y)],indent=2)+'\n'
        for folder in ['environment','tests']:(task/folder/'data/calibration.json').write_text(data)
    public,private=task/'environment/data/calibration.json',task/'tests/data/calibration.json'
    assert public.read_bytes()==private.read_bytes()
    records=json.loads(public.read_text());assert [r['input'] for r in records]==inputs
    axis_oracle=oracle();axis_oracle.viscosity=true
    axis_shortcut=shortcut();axis_shortcut.viscosity=true
    axis_equality=float(np.max(abs(axis_oracle.predict(inputs)-axis_shortcut.predict(inputs))))
    assert axis_equality<1e-14
    hidden=reference.hidden_inputs();truths={k:reference.predict(v,true) for k,v in hidden.items()}
    def scores(model):
        return {k:float(np.linalg.norm(model.predict(hidden[k])-y)/np.linalg.norm(y)) for k,y in truths.items()}
    report=dict(revision=2,calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],noise_trials=args.noise_trials,measurement_sigma=float(sigma),calibration_control_agreement_max=axis_equality,controls={})
    for name,cls in [('oracle',oracle),('shortcut',shortcut)]:
        exact=cls();exact.viscosity=true
        agreement=float(np.linalg.norm(exact.predict(inputs)-noiseless)/np.linalg.norm(noiseless));assert agreement<1e-3
        model=cls().fit(records);result=metrics(model,records,true);result['hidden']=scores(model);result['calibration_reference_relative_error']=agreement
        assert result['parameter_relative_error']<.03 and result['calibration_chi2']<1.5
        assert (max(result['hidden'].values())<meta['prediction_limit'] if name=='oracle' else min(result['hidden'].values())>meta['prediction_limit'])
        report['controls'][name]=result
    exact=oracle();exact.viscosity=true
    agreement=[];refinement=[];time_refinement=[]
    for name,es in hidden.items():
        y=reference.predict(es,true,nx=144,ny=96)
        refinement.append(float(np.linalg.norm(y-truths[name])/np.linalg.norm(y)))
        agreement.append(float(np.linalg.norm(exact.predict(es)-y)/np.linalg.norm(y)))
        coarse_time=reference.predict(es,true,nx=48,ny=32,steps=1200)
        fine_time=reference.predict(es,true,nx=48,ny=32,steps=2400)
        time_refinement.append(float(np.linalg.norm(coarse_time-fine_time)/np.linalg.norm(fine_time)))
    assert max(refinement)<.001 and max(agreement)<.001 and max(time_refinement)<.001
    experiments=[e for group in hidden.values() for e in group]
    symmetry=float(max(abs(exact.predict(experiments)-exact.predict([dict(e,magnetic_x=-e['magnetic_x'],magnetic_y=-e['magnetic_y']) for e in experiments]))))
    reversal=float(max(abs(exact.predict(experiments)+exact.predict([dict(e,pressure_gradient=-e['pressure_gradient']) for e in experiments]))))
    assert symmetry<1e-14 and reversal<1e-14
    reflection=float(max(abs(exact.predict(experiments)-exact.predict([dict(e,magnetic_x=-e['magnetic_x'],x=1-e['x']) for e in experiments]))))
    assert reflection<1e-14
    # Independent discrete charge and mechanical/electrical power identities.
    viscous,cx,cy,g=reference.operators(24,16)
    u=np.random.default_rng(281).normal(0,1e-4,cx.shape[1]);bx,by=.35,-.45
    c=by*cx-bx*cy
    phi=splu(g.T@g).solve(-g.T@(c@u))
    current=-1e5*(g@phi+c@u)
    divergence=g.T@current
    charge_error=float(np.linalg.norm(divergence)/np.linalg.norm(g.T@(1e5*c@u)))
    power=float(u@(c.T@current));heat=float(current@current/1e5)
    dissipation_error=abs(power+heat)/heat
    assert charge_error<1e-10 and dissipation_error<1e-10 and power<0
    m,n,k2,mean,qx,qy,qxy=om.operators(12,16)
    correct=bx*bx*qx+by*by*qy+bx*by*qxy
    approximate=bx*bx*qx+by*by*qy
    eigen_min=[float(np.linalg.eigvalsh(x).min()) for x in [correct,approximate]]
    assert min(eigen_min)>-1e-12
    # Independent-axis currents each satisfy the same no-current wall closure.
    channel_heat=0.;channel_power=0.
    for ci in [by*cx,-bx*cy]:
        field=splu(g.T@g).solve(-g.T@(ci@u))
        ji=-1e5*(g@field+ci@u)
        channel_heat+=float(ji@ji/1e5);channel_power+=float(u@(ci.T@ji))
    channel_power_error=abs(channel_heat+channel_power)/channel_heat
    assert channel_power_error<1e-10
    # Refining velocity and current modes checks the shared-field spectral solve.
    spectral_refinement=[]
    for es in hidden.values():
        refined=[]
        mm,nn,_,avg,*_=om.operators(52,72)
        for e in es:
            weight=avg if e['observable']=='mean' else 2*np.sin(mm*np.pi*e['x'])*np.sin(nn*np.pi*e['y'])
            value=0.
            for ids,rates,basis,forcing in om.spectrum(true,e['magnetic_x'],e['magnetic_y'],52,72):
                value+=weight[ids]@(basis@(-np.expm1(-rates*e['time'])/rates*forcing))
            refined.append(e['pressure_gradient']/1000*value)
        spectral_refinement.append(float(np.linalg.norm(np.array(refined)-exact.predict(es))/np.linalg.norm(refined)))
    assert max(spectral_refinement)<.001
    report['physical_checks']=dict(reference_spatial_refinement_relative_max=max(refinement),oracle_reference_relative_max=max(agreement),reference_time_refinement_relative_max=max(time_refinement),spectral_refinement_relative_max=max(spectral_refinement),field_reversal_max_error=symmetry,pressure_reversal_max_error=reversal,reflection_max_error=reflection,charge_balance_relative_error=charge_error,dissipation_relative_error=dissipation_error,shortcut_independent_channel_dissipation_relative_error=channel_power_error,magnetic_operator_minimum_eigenvalues=eigen_min)
    rng=np.random.default_rng(meta['noise_seed']);parameters=[];chi2=[];errors=[]
    for i in range(args.noise_trials):
        y=noiseless+rng.normal(0,sigma,len(inputs));sample=[dict(input=e,value=float(v),sigma=float(sigma)) for e,v in zip(inputs,y)]
        model=oracle().fit(sample);m=metrics(model,sample,true)
        parameters.append(model.viscosity);chi2.append(m['calibration_chi2']);errors.append(m['parameter_relative_error'])
        if i in [0,args.noise_trials-1]:assert abs(shortcut().fit(sample).viscosity-model.viscosity)<1e-10
    sensitivity={}
    for name,cls in [('oracle',oracle),('shortcut',shortcut)]:
        values=[]
        for p in [min(parameters),max(parameters)]:
            model=cls();model.viscosity=p;values.extend(scores(model).values())
        sensitivity[name]=dict(min=min(values),max=max(values))
    assert max(errors)<.03 and np.mean(np.array(chi2)<1.5)>=.99
    assert sensitivity['oracle']['max']<meta['prediction_limit'] and sensitivity['shortcut']['min']>meta['prediction_limit']
    report['noise']=dict(parameter_min=min(parameters),parameter_max=max(parameters),parameter_relative_error_max=max(errors),calibration_chi2_max=max(chi2),calibration_pass_fraction=float(np.mean(np.array(chi2)<1.5)),hidden_at_parameter_extrema=sensitivity)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(TASK,'PASS',args.output)


if __name__=='__main__':main()
