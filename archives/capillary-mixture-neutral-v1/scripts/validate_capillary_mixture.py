"""Scientific controls for conservative composition and capillary flow."""
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


def project(states,times,experiments):
    x=np.arange(states.shape[1])*2*np.pi/states.shape[1];x,y=np.meshgrid(x,x,indexing='ij')
    out=[]
    for e in experiments:
        q=e['detector_wave'];phase=q[0]*x+q[1]*y
        detector=np.cos(phase) if e['quadrature']=='cosine' else np.sin(phase)
        out.append(2*np.mean(states[times.index(e['time'])]*detector))
    return np.array(out)


def metrics(model,records,true):
    values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
    prediction=model.predict([r['input'] for r in records])
    return {'parameter':model.mobility,'parameter_relative_error':abs(model.mobility/true-1),
            'calibration_chi2':float(np.sum(((prediction-values)/sigma)**2)/(len(records)-1))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256)
    parser.add_argument('--output',type=Path,default=ROOT/'jobs/capillary-mixture-validation/summary.json')
    args=parser.parse_args();started=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/capillary_mixture_baseline.py');ref=module(TASK/'tests/reference.py')
    metadata=json.loads((TASK/'tests/metadata.json').read_text());true=ref.TRUE_PARAMETER
    inputs=ref.calibration_inputs();exact=ref.predict(inputs,true);sigma=metadata['measurement_sigma']
    if args.generate:
        values=exact+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
        for directory in ['environment','tests']:(TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs
    hidden=ref.hidden_inputs();truth={n:ref.predict(es,true) for n,es in hidden.items()}
    def scores(model):return {n:float(np.sqrt(np.mean((model.predict(es)-truth[n])**2))) for n,es in hidden.items()}
    report={'revision':1,'calibration_seed':metadata['calibration_seed'],'noise_seed':metadata['noise_seed'],
            'noise_trials':args.noise_trials,'controls':{}}
    for label,source in [('oracle',good),('shortcut',bad)]:
        m=source.Model().fit(records);result=metrics(m,records,true);result['hidden']=scores(m)
        result['calibration_reference_max_error']=float(max(abs(source.calibration_predict(inputs,true)-exact)))
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03 and result['calibration_reference_max_error']<1e-12
        assert (max(result['hidden'].values())<.003 if label=='oracle' else min(result['hidden'].values())>.003)
        report['controls'][label]=result
    rng=np.random.default_rng(metadata['noise_seed']);parameters=[];chi2s=[]
    for _ in range(args.noise_trials):
        values=exact+rng.normal(0,sigma,len(inputs));sample=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
        a=good.Model().fit(sample);b=bad.Model().fit(sample);result=metrics(a,sample,true)
        assert abs(a.mobility-b.mobility)<1e-12
        parameters.append(a.mobility);chi2s.append(result['calibration_chi2'])
    report['noise']={'parameter_min':min(parameters),'parameter_max':max(parameters),
                     'parameter_relative_error_max':float(max(abs(np.array(parameters)/true-1))),
                     'calibration_chi2_max':max(chi2s),'calibration_pass_fraction':float(np.mean(np.array(chi2s)<1.5))}
    assert report['noise']['parameter_relative_error_max']<.03 and report['noise']['calibration_pass_fraction']>=.99
    extrema={}
    for label,source in [('oracle',good),('shortcut',bad)]:
        values=[]
        for parameter in [min(parameters),max(parameters)]:
            model=source.Model();model.mobility=parameter;values.extend(scores(model).values())
        extrema[label]={'min':min(values),'max':max(values)}
    assert extrema['oracle']['max']<.003 and extrema['shortcut']['min']>.003
    report['hidden_parameter_extrema_check']=extrema
    report['physical_checks']=physical_checks(good,bad,ref,truth)
    report['seconds']=time.time()-started
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('capillary-mixture PASS',args.output,report['seconds'])


def physical_checks(good,bad,ref,truth):
    reference_errors=[];space_errors=[];time_errors=[];mass=[];increases=[];work_errors=[];divergence=[];means=[];extremes=[]
    for name,es in ref.hidden_inputs().items():
        times=sorted({e['time'] for e in es})
        a=good.profile(es[0],times,.04);p=project(a,times,es)
        reference_errors.append(float(max(abs(p-truth[name]))))
        refined=ref.predict(es,.04,points=192,max_step=.0025)
        space_errors.append(float(max(abs(refined-truth[name]))))
        finer=project(good.profile(es[0],times,.04,points=64,max_step=.01),times,es)
        time_errors.append(float(max(abs(finer-p))))
        for source in [good,bad]:
            fields=source.profile(es[0],np.linspace(0,4,17),.04)
            x,y,kx,ky,k2,inv,mask=source.grid(fields.shape[1]);energies=[]
            for c in fields:
                state=np.fft.fft2(c);cx=np.fft.ifft2(1j*kx*state).real;cy=np.fft.ifft2(1j*ky*state).real
                energies.append(float(np.mean(.5*c*c+.1*(cx*cx+cy*cy))))
                mass.append(abs(float(np.mean(c))));extremes.extend([float(c.min()),float(c.max())])
                n,vx,vy=source.velocity_and_advection(state,len(c));ux=np.fft.fft2(vx);uy=np.fft.fft2(vy)
                divergence.append(float(max(abs(np.fft.ifft2(1j*(kx*ux+ky*uy)).ravel()))))
                means.extend([abs(float(vx.mean())),abs(float(vy.mean()))])
                if source is good:
                    muhat=(1+.2*k2)*state;mu=np.fft.ifft2(muhat).real
                    dc=np.fft.ifft2(-.04*k2*muhat+n).real
                    gradient=[np.fft.ifft2(1j*k*muhat).real for k in [kx,ky]]
                    vg=[np.fft.ifft2(1j*k*u).real for k in [kx,ky] for u in [ux,uy]]
                    diss=.04*sum(np.mean(g*g) for g in gradient)+.004*sum(np.mean(g*g) for g in vg)
                    work_errors.append(abs(float(np.mean(mu*dc)+diss)))
            increases.append(float(max(np.diff(energies))))
    one_mode=[]
    for k in [1,2,3]:
        e=ref.experiment([ref.mode([k,0],.3,.4)],2.,[k,0]);times=[0.,.3,2.]
        fields=good.profile(e,times,.02)
        es=[dict(e,time=t) for t in times]
        one_mode.append(float(max(abs(project(fields,times,es)-good.calibration_predict(es,.02)))))
    # Equal Laplacian eigenvalues also produce a pressure-only reversible force.
    e=ref.experiment([ref.mode([1,0],.2),ref.mode([0,1],.15,.7)],1.,[1,0])
    f=good.profile(e,[0.],.04)[0];force=good.velocity_and_advection(np.fft.fft2(f),len(f))
    same_shell=float(max(np.max(abs(force[1])),np.max(abs(force[2]))))
    corners=[]
    for parameter in [.02,.09]:
        e=ref.experiment([ref.mode([3,3],.14,.8),ref.mode([3,-2],.13,-.3),ref.mode([-2,1],.08,1.2)],.3,[3,3])
        es=[dict(e,time=t,detector_wave=q,quadrature=z) for t in [.04,.15,.3] for q in [[3,3],[3,-2],[-2,1],[0,5]] for z in ['cosine','sine']]
        times=sorted({e['time'] for e in es})
        a=project(good.profile(e,times,parameter),times,es)
        b=project(good.profile(e,times,parameter,points=64,max_step=.005),times,es)
        corners.append(float(max(abs(a-b))))
    assert max(reference_errors)<2e-4 and max(space_errors)<2e-4 and max(time_errors)<2e-5
    assert max(mass)<1e-12 and max(increases)<1e-12 and max(work_errors)<1e-10
    assert max(divergence)<1e-10 and max(means)<1e-12 and max(one_mode)<1e-12 and same_shell<1e-10
    assert max(corners)<.0002 and min(extremes)>-.5 and max(extremes)<.5
    return {'oracle_fv128_absolute_error':max(reference_errors),'fv128_vs192_and_time_refinement':max(space_errors),
            'spectral48_vs64_and_time_refinement':max(time_errors),'high_wavenumber_corner_refinement':max(corners),
            'mean_composition_drift':max(mass),'maximum_free_energy_increase_both_controls':max(increases),
            'instantaneous_thermodynamic_work_identity_error':max(work_errors),'velocity_divergence_max':max(divergence),
            'mean_velocity_max':max(means),'pure_stripe_exact_calibration_error':max(one_mode),
            'equal_length_wavevectors_velocity_max':same_shell,'composition_range':[min(extremes),max(extremes)]}


if __name__=='__main__':main()
