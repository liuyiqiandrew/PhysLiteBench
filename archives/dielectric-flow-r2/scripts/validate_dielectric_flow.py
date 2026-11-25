"""Scientific controls for dielectric-gradient and free-charge forcing."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/dielectric-flow'


def module(path):
    spec=importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))),path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def key(e):return json.dumps({k:e[k] for k in ['permittivity_modes','conductivity_modes','field']},sort_keys=True)


def scientific_checks(good,bad,ref,inputs,hidden,truth):
    equivalence=float(np.max(abs(good.predict_at(inputs,1.1)-bad.predict_at(inputs,1.1))))
    errors=[];refinement=[];spectral=[];current=[];curl=[];gauss_cal=[];divergence=[];means=[];mean_force=[];work=[];maxwell=[];reversal=[];translation=[];charge_ratio=[]
    unique={key(e):e for e in inputs+sum(hidden.values(),[])}
    for name,es in hidden.items():
        errors.append(float(np.max(abs(good.predict_at(es,1.1)-truth[name]))))
        sample=es[::4]
        refinement.append(float(np.max(abs(ref.predict(sample,1.1,points=192)-ref.predict(sample,1.1)))))
    for k,e in unique.items():
        x,y,epsilon,sigma,E=good.electric_state(k)
        _,_,kx,ky,_=good.geometry(len(x))
        current.append(float(np.max(abs(good.divergence(sigma*E,kx,ky)))))
        curl.append(float(np.max(abs(good.gradient(E[1],kx,ky)[0]-good.gradient(E[0],kx,ky)[1]))))
        rho=good.divergence(epsilon*E,kx,ky)
        if e['conductivity_modes']==e['permittivity_modes']:gauss_cal.append(float(np.max(abs(rho))))
        stress=np.array([[epsilon*(E[0]**2-.5*np.sum(E*E,axis=0)),epsilon*E[0]*E[1]],
                         [epsilon*E[0]*E[1],epsilon*(E[1]**2-.5*np.sum(E*E,axis=0))]])
        full=good.electric_force(epsilon,E,kx,ky)
        direct=np.array([good.divergence(row,kx,ky) for row in stress])
        maxwell.append(float(np.max(abs(full-direct))))
        mean_force.append(float(np.max(abs(full.mean(axis=(1,2))))))
        for source in [good,bad]:
            _,_,u=source.unit_velocity(k)
            f=source.electric_force(epsilon,E,kx,ky)
            divergence.append(float(np.max(abs(good.divergence(u,kx,ky)))))
            means.append(float(np.max(abs(u.mean(axis=(1,2))))))
            diss=sum(np.mean(np.sum(good.gradient(a,kx,ky)**2,axis=0)) for a in u)
            work.append(abs(float(np.mean(np.sum(u*f,axis=0))-diss)))
        # Largest chosen steady velocity at the lowest allowed viscosity.
        speed=np.max(np.sqrt(np.sum(good.unit_velocity(k)[2]**2,axis=0)))/.7
        charge_ratio.append(float(epsilon.max()/sigma.min()*speed*np.sqrt(8)))
        reverse=dict(e,field=[-v for v in e['field']])
        reversal.append(float(np.max(abs(good.unit_velocity(k)[2]-good.unit_velocity(key(reverse))[2]))))
        shift=np.array([.37,-.21]);translated=json.loads(json.dumps(e))
        for kind in ['permittivity_modes','conductivity_modes']:
            for m in translated[kind]:m['phase']+=float(np.dot(m['wave'],shift))
        translated['detector_phase']+=float(np.dot(e['detector_wave'],shift))
        translation.append(float(abs(good.predict_at([e],1.1)[0]-good.predict_at([translated],1.1)[0])))
        a=good.unit_velocity(k,49);b=good.unit_velocity(k,65)
        q=e['detector_wave'];p=e['detector_phase'];c=e['component']
        spectral.append(float(abs(2*np.mean(a[2][c]*np.cos(q[0]*a[0]+q[1]*a[1]+p))-2*np.mean(b[2][c]*np.cos(q[0]*b[0]+q[1]*b[1]+p)))))
    zero=ref.experiment([],[],field=(.3,.1));uniform=float(np.max(abs(good.unit_velocity(key(zero))[2])))
    zero_field=dict(inputs[0],field=[0.,0.]);zero_error=float(np.max(abs(good.unit_velocity(key(zero_field))[2])))
    corner=ref.experiment([ref.mode((2,2),.35,.7),ref.mode((2,-1),.35,-.3)],
                          [ref.mode((1,2),-.35,.3),ref.mode((-2,1),.35,1.1)],field=(.32,.24))
    corners=[dict(corner,component=c,detector_wave=list(q),detector_phase=p) for c in [0,1] for q in [(1,0),(0,1),(2,2),(3,-2)] for p in [0.,-.7]]
    corner_error=float(np.max(abs(good.predict_at(corners,.7)-ref.predict(corners,.7,points=192))))
    # The independent finite-volume force is an exact conservative divergence.
    fv=ref.fields(key(corner),96);h=2*np.pi/96
    face_div=lambda a:sum((a[i]-np.roll(a[i],1,i))/h for i in [0,1])
    fv_current=float(np.max(abs(face_div(fv[3]))))
    fv_incompressibility=float(np.max(abs(face_div(fv[2]))))
    fv_meanforce=float(np.max(abs(fv[4].mean(axis=(1,2)))))
    result={'calibration_closure_equivalence':equivalence,'oracle_reference_max_error':max(errors),
            'reference96_to192_error':max(refinement),'spectral49_to65_error':max(spectral),
            'current_divergence_max':max(current),'electric_curl_max':max(curl),
            'proportional_profile_free_charge_max':max(gauss_cal),'both_models_velocity_divergence_max':max(divergence),
            'both_models_mean_velocity_max':max(means),'maxwell_force_mean_max':max(mean_force),
            'both_models_viscous_work_error':max(work),'maxwell_stress_force_identity_error':max(maxwell),
            'field_reversal_velocity_error':max(reversal),'translation_identity_error':max(translation),
            'uniform_material_velocity_max':uniform,'zero_field_velocity_max':zero_error,
            'extreme_allowed_pattern_reference_error':corner_error,'fv_current_divergence_max':fv_current,
            'fv_velocity_divergence_max':fv_incompressibility,'fv_mean_force_max':fv_meanforce,
            'sampled_charge_convection_ratio_max':max(charge_ratio),
            'calibration_signal_rms':float(np.sqrt(np.mean(good.predict_at(inputs,1.1)**2)))}
    assert equivalence<1e-10 and max(errors)<2e-7 and max(refinement)<2e-7
    assert max(spectral)<1e-7 and max(current)<1e-7 and max(curl)<1e-10 and max(gauss_cal)<1e-10
    assert max(divergence)<1e-10 and max(means)<1e-12 and max(work)<1e-10
    assert max(mean_force)<1e-9 and max(maxwell)<1e-6 and max(reversal)<1e-10 and max(translation)<1e-8
    assert uniform<1e-12 and zero_error<1e-12 and corner_error<2e-6
    assert fv_current<1e-7 and fv_incompressibility<1e-10 and fv_meanforce<1e-12
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256)
    parser.add_argument('--output',type=Path,default=ROOT/'results/dielectric-flow-r2-validation.json')
    args=parser.parse_args();started=time.monotonic()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/dielectric_flow_baseline.py');ref=module(TASK/'tests/reference.py')
    metadata=json.loads((TASK/'tests/metadata.json').read_text());true=ref.TRUE_PARAMETER
    inputs=ref.calibration_inputs();exact=ref.predict(inputs,true);sigma=metadata['measurement_sigma']
    if args.generate:
        observed=exact+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(exact))
        records=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,observed)]
        for side in ['environment','tests']:(TASK/side/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs
    hidden=ref.hidden_inputs();truth={name:ref.predict(es,true) for name,es in hidden.items()}
    def scores(model):return {name:float(np.sqrt(np.mean((model.predict(es)-truth[name])**2))/max(np.sqrt(np.mean(truth[name]**2)),1e-8)) for name,es in hidden.items()}
    def calibration(model,sample):
        residual=(model.predict(inputs)-np.array([r['value'] for r in sample]))/sigma
        return float(residual@residual)/(len(sample)-1)
    report={'revision':2,'calibration_seed':metadata['calibration_seed'],'noise_seed':metadata['noise_seed'],'noise_trials':args.noise_trials,'controls':{}}
    for label,source in [('oracle',good),('shortcut',bad)]:
        model=source.Model().fit(records)
        result={'parameter':model.viscosity,'parameter_relative_error':abs(model.viscosity/true-1),
                'calibration_chi2':calibration(model,records),'hidden':scores(model)}
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        assert (max(result['hidden'].values())<.04 if label=='oracle' else min(result['hidden'].values())>.04)
        report['controls'][label]=result
    rng=np.random.default_rng(metadata['noise_seed']);parameters=[];chi2=[]
    for _ in range(args.noise_trials):
        observed=exact+rng.normal(0,sigma,len(exact));sample=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,observed)]
        a=good.Model().fit(sample);b=bad.Model().fit(sample)
        assert abs(a.viscosity-b.viscosity)<1e-11
        parameters.append(a.viscosity);chi2.append(calibration(a,sample))
    report['noise']={'parameter_min':min(parameters),'parameter_max':max(parameters),'parameter_relative_error_max':float(max(abs(np.array(parameters)/true-1))),
                     'calibration_chi2_max':max(chi2),'calibration_pass_fraction':float(np.mean(np.array(chi2)<1.5))}
    assert report['noise']['parameter_relative_error_max']<.03 and report['noise']['calibration_pass_fraction']>=.99
    extrema={}
    for label,source in [('oracle',good),('shortcut',bad)]:
        values=[]
        for parameter in [min(parameters),max(parameters)]:
            m=source.Model();m.viscosity=parameter;values.extend(scores(m).values())
        extrema[label]={'min':min(values),'max':max(values)}
    assert extrema['oracle']['max']<.04 and extrema['shortcut']['min']>.04
    report['hidden_parameter_extrema_check']=extrema
    report['independent_checks']=scientific_checks(good,bad,ref,inputs,hidden,truth)
    report['seconds']=time.monotonic()-started
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps({'dielectric-flow':report},indent=2)+'\n')
    print('dielectric-flow PASS',args.output,report['seconds'])


if __name__=='__main__':main()
