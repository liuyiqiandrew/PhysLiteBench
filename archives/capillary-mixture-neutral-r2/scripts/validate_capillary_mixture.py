"""Variable-viscosity Stokes controls and independent variational checks."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/capillary-mixture'


def module(path):
    spec=importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def metrics(model,records,true):
    values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
    prediction=model.predict([r['input'] for r in records])
    return {'parameter':model.viscosity,'parameter_relative_error':abs(model.viscosity/true-1),
            'calibration_chi2':float(np.sum(((prediction-values)/sigma)**2)/(len(records)-1))}


def prediction(source,experiments,parameter,points):
    out=[]
    for e in experiments:
        prep={k:e[k] for k in ['modes','viscosity_contrast']}
        x,y,u=source.unit_velocity(json.dumps(prep,sort_keys=True),points)
        q=e['detector_wave'];weight=np.cos(q[0]*x+q[1]*y+e['detector_phase'])
        out.append(2*np.mean(u[e['component']]*weight)/parameter)
    return np.array(out)


def physical_checks(good,bad,ref,truth):
    reference_errors=[];reference_refinement=[];oracle_refinement=[]
    divergence=[];means=[];work_errors=[];residuals=[];dissipation=[];missing=[]
    for name,es in ref.hidden_inputs().items():
        reference_errors.append(float(max(abs(good.predict_at(es,.004)-truth[name]))))
        reference_refinement.append(float(max(abs(ref.predict(es,.004,16,81)-truth[name]))))
        oracle_refinement.append(float(max(abs(prediction(good,es,.004,65)-good.predict_at(es,.004)))))
        for beta in [3.5,5.5,7.]:
            e=dict(es[0],viscosity_contrast=beta)
            prep={k:e[k] for k in ['modes','viscosity_contrast']};key=json.dumps(prep,sort_keys=True)
            x,y,kx,ky,inv=good.grid(49)
            c=sum(m['amplitude']*np.cos(m['wave'][0]*x+m['wave'][1]*y+m['phase']) for m in e['modes'])
            eta=np.exp(beta*c);lap=np.fft.ifft2(-(kx*kx+ky*ky)*np.fft.fft2(c)).real
            force=good.project(-.2*lap*good.gradient(c,kx,ky),kx,ky,inv)
            for source in [good,bad]:
                _,_,u=source.unit_velocity(key)
                du=np.array([good.gradient(u[i],kx,ky) for i in range(2)])
                strain=du+du.swapaxes(0,1)
                stress=eta*(strain if source is good else du)
                divstress=np.array([good.divergence(stress[i],kx,ky) for i in range(2)])
                residual=force+good.project(divstress,kx,ky,inv)
                residuals.append(float(np.linalg.norm(residual)/np.linalg.norm(force)))
                power=float(np.mean(np.sum(force*u,axis=0)))
                loss=float(np.mean(np.sum(.5*eta*strain**2 if source is good else eta*du**2,axis=(0,1))))
                work_errors.append(abs(power-loss));dissipation.append(loss)
                divergence.append(float(np.max(abs(good.divergence(u,kx,ky)))))
                means.append(float(np.max(abs(u.mean(axis=(1,2))))))
                if source is bad:
                    omitted=np.array([good.divergence(eta*du.swapaxes(0,1)[i],kx,ky) for i in range(2)])
                    missing.append(float(np.linalg.norm(good.project(omitted,kx,ky,inv))/np.linalg.norm(force)))
    corner=ref.experiment([ref.mode((3,3),.14,.8),ref.mode((3,-2),.13,-.3),ref.mode((-2,1),.08,1.2)],7.)
    corners=[dict(corner,component=d,detector_wave=q,detector_phase=p) for d in [0,1]
             for q in [[0,5],[6,1],[1,4],[3,3],[3,-2],[-2,1]] for p in [0.,-np.pi/2]]
    corner_error=float(max(abs(prediction(good,corners,.002,49)-prediction(good,corners,.002,81))))
    shifted=[];shift=np.array([.23,-.41]);es=ref.hidden_inputs()['oblique_waves']
    for e in es:
        modes=[dict(m,phase=m['phase']+np.dot(m['wave'],shift)) for m in e['modes']]
        shifted.append(dict(e,modes=modes,detector_phase=e['detector_phase']+np.dot(e['detector_wave'],shift)))
    translation=float(max(abs(good.predict_at(es,.004)-good.predict_at(shifted,.004))))
    e=dict(es[0],viscosity_contrast=4.)
    scaled=dict(e,viscosity_contrast=5.,modes=[dict(m,amplitude=.8*m['amplitude']) for m in e['modes']])
    scaling=float(abs(good.predict_at([scaled],.004)[0]-.8**2*good.predict_at([e],.004)[0]))
    calibration_equivalence=float(max(abs(good.predict_at(ref.calibration_inputs(),.004)-bad.predict_at(ref.calibration_inputs(),.004))))
    rotation=np.array([[0.,-1.],[1.,0.]])
    rotation_stress=float(np.max(abs(rotation+rotation.T)))
    assert max(reference_errors)<1e-5 and max(reference_refinement)<1e-5 and max(oracle_refinement)<1e-6
    assert corner_error<.001 and max(residuals)<1e-8 and max(work_errors)<1e-12
    assert max(divergence)<1e-10 and max(means)<1e-12 and min(dissipation)>0 and min(missing)>.01
    assert translation<1e-8 and scaling<1e-9 and calibration_equivalence<1e-10 and rotation_stress==0
    return {'oracle_variational_reference_max_error':max(reference_errors),
            'reference_cutoff12_vs16_error':max(reference_refinement),'oracle49_vs65_error':max(oracle_refinement),
            'high_wave_corner49_vs81_error':corner_error,'velocity_divergence_max':max(divergence),
            'mean_velocity_max':max(means),'both_closure_relative_stokes_residual':max(residuals),
            'both_closure_work_dissipation_error':max(work_errors),'both_closure_dissipation_min':min(dissipation),
            'shortcut_missing_projected_stress_relative_min':min(missing),'translation_covariance_error':translation,
            'composition_scaling_error':scaling,'uniform_viscosity_calibration_equivalence':calibration_equivalence,
            'local_rigid_rotation_newtonian_stress':rotation_stress,
            'local_rigid_rotation_shortcut_stress_norm':float(np.linalg.norm(rotation))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256)
    parser.add_argument('--output',type=Path,default=ROOT/'jobs/capillary-mixture-r2-validation/summary.json')
    args=parser.parse_args();started=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/capillary_mixture_baseline.py');ref=module(TASK/'tests/reference.py')
    metadata=json.loads((TASK/'tests/metadata.json').read_text());true=ref.TRUE_PARAMETER;limit=metadata['prediction_limit']
    inputs=ref.calibration_inputs();exact=ref.predict(inputs,true);sigma=metadata['measurement_sigma']
    if args.generate:
        values=exact+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
        for directory in ['environment','tests']:(TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs();truth={n:ref.predict(es,true) for n,es in hidden.items()}
    def scores(model):return {n:float(np.sqrt(np.mean((model.predict(es)-truth[n])**2))) for n,es in hidden.items()}
    report={'revision':2,'calibration_seed':metadata['calibration_seed'],'noise_seed':metadata['noise_seed'],
            'noise_trials':args.noise_trials,'measurement_sigma':sigma,'prediction_limit':limit,'controls':{}}
    for label,source in [('oracle',good),('shortcut',bad)]:
        m=source.Model().fit(records);result=metrics(m,records,true);result['hidden']=scores(m)
        result['calibration_reference_max_error']=float(max(abs(source.predict_at(inputs,true)-exact)))
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03 and result['calibration_reference_max_error']<1e-10
        assert (max(result['hidden'].values())<limit if label=='oracle' else min(result['hidden'].values())>limit)
        report['controls'][label]=result
    rng=np.random.default_rng(metadata['noise_seed']);parameters=[];chi2s=[]
    for _ in range(args.noise_trials):
        values=exact+rng.normal(0,sigma,len(inputs));sample=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
        a=good.Model().fit(sample);b=bad.Model().fit(sample);result=metrics(a,sample,true)
        assert abs(a.viscosity-b.viscosity)<1e-12
        parameters.append(a.viscosity);chi2s.append(result['calibration_chi2'])
    report['noise']={'parameter_min':min(parameters),'parameter_max':max(parameters),
                     'parameter_relative_error_max':float(max(abs(np.array(parameters)/true-1))),
                     'calibration_chi2_max':max(chi2s),'calibration_pass_fraction':float(np.mean(np.array(chi2s)<1.5))}
    assert report['noise']['parameter_relative_error_max']<.03 and report['noise']['calibration_pass_fraction']>=.99
    extrema={}
    for label,source in [('oracle',good),('shortcut',bad)]:
        values=[]
        for parameter in [min(parameters),max(parameters)]:
            model=source.Model();model.viscosity=parameter;values.extend(scores(model).values())
        extrema[label]={'min':min(values),'max':max(values)}
    assert extrema['oracle']['max']<limit and extrema['shortcut']['min']>limit
    report['hidden_parameter_extrema_check']=extrema
    report['physical_checks']=physical_checks(good,bad,ref,truth)
    report['seconds']=time.time()-started
    paths=[TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/metadata.json',TASK/'environment/data/calibration.json',ROOT/'scripts/capillary_mixture_baseline.py',Path(__file__)]
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('capillary-mixture r2 PASS',args.output,report['seconds'])


if __name__=='__main__':main()
