"""Boundary-value, energy and noisy-calibration checks for electron-fluid heating."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/hydrodynamic-heating'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


good=load(TASK/'solution/model.py','good')
bad=load(ROOT/'scripts/hydrodynamic_heating_baseline.py','bad')
ref=load(TASK/'tests/reference.py','reference')


def local_maxwell_heat(experiment,plasma):
    """Independent local two-transverse-wave slab solution."""
    omega=experiment['frequency'];d=experiment['thickness'];cosine=np.cos(experiment['angle']);k=omega*np.sin(experiment['angle'])
    eps=1-plasma**2/(omega*(omega+.06j));q=np.sqrt(eps*omega**2-k*k+0j);admittance=q/(omega*eps)
    phase=np.exp(1j*q*d)
    matrix=np.array([[admittance+cosine,(-admittance+cosine)*phase],[(admittance-cosine)*phase,-admittance-cosine]])
    amplitudes=np.linalg.solve(matrix,[2*cosine,0])
    nodes,weights=np.polynomial.legendre.leggauss(128);lo,hi=d*np.array(experiment['window']);z=(lo+hi)/2+(hi-lo)*nodes/2
    forward=amplitudes[0]*np.exp(1j*q*z);backward=amplitudes[1]*np.exp(-1j*q*(z-d))
    electric=np.array([admittance*(forward-backward),-k/(omega*eps)*(forward+backward)])
    return (hi-lo)/2*np.dot(weights,.06*plasma**2/(omega**2+.06**2)*np.sum(abs(electric)**2,axis=0))/cosine


def run(count):
    records=json.loads((TASK/'tests/data/calibration.json').read_text());inputs=[r['input'] for r in records];sigma=np.array([r['sigma'] for r in records]);clean=ref.predict(inputs,1.)
    hidden=ref.hidden_inputs();truth={key:ref.predict(es,1.) for key,es in hidden.items()}
    def scores(module,plasma):return {key:float(np.sqrt(np.mean((module.predict_at(es,plasma)-truth[key])**2)/np.mean(truth[key]**2))) for key,es in hidden.items()}
    controls={}
    for label,module in [('oracle',good),('shortcut',bad)]:
        model=module.Model().fit(records);res=(model.predict(inputs)-np.array([r['value'] for r in records]))/sigma
        controls[label]=dict(plasma_frequency=model.plasma_frequency,calibration_chi2=float(res@res/(len(records)-1)),hidden=scores(module,model.plasma_frequency))
    rng=np.random.default_rng(617193);samples=[]
    for _ in range(count):
        observed=clean+sigma*rng.normal(size=len(inputs));noisy=[dict(input=e,value=float(v),sigma=float(s)) for e,v,s in zip(inputs,observed,sigma)]
        a=good.Model().fit(noisy);b=bad.Model().fit(noisy)
        assert abs(a.plasma_frequency-b.plasma_frequency)<1e-9
        res=(a.predict(inputs)-observed)/sigma
        samples.append(dict(plasma_frequency=a.plasma_frequency,chi2=float(res@res/(len(inputs)-1)),oracle=scores(good,a.plasma_frequency),shortcut=scores(bad,b.plasma_frequency)))
    all_inputs=inputs+sum(hidden.values(),[])
    corners=[ref.experiment(d,w,a,v) for d in [.2,.8] for w in [.7,1.,1.5] for a in [0.,-1.,1.] for v in [[0.,.15],[.35,.65],[.85,1.],[0.,1.]]]
    reference_error=corner_error=refinement=energy=additivity=mirror=fields_error=boundary_residual=0.;minimum_heat=1.;maximum_heat=0.;normal_longitudinal=0.
    for plasma in [.85,1.,1.15]:
        reference_error=max(reference_error,float(max(abs(good.predict_at(all_inputs,plasma)-ref.predict(all_inputs,plasma)))))
        corner_error=max(corner_error,float(max(abs(good.predict_at(corners,plasma)-ref.predict(corners,plasma)))))
        for e in corners+sum(hidden.values(),[]):
            heat=good.absorbed_fraction(e,plasma);refined=good.absorbed_fraction(e,plasma,128)
            refinement=max(refinement,abs(heat-refined));minimum_heat=min(minimum_heat,heat)
            mirror=max(mirror,abs(heat-good.absorbed_fraction(dict(e,angle=-e['angle']),plasma)))
            coeff=good.amplitudes(e,plasma);front,back=good.mode_fields([0,e['thickness']],e,plasma)@coeff
            normal=np.cos(e['angle']);reflection=front[2]-1;transmission=back[2];full=good.absorbed_fraction(dict(e,window=[0.,1.]),plasma)
            maximum_heat=max(maximum_heat,full)
            energy=max(energy,abs(full-(1-abs(reflection)**2-abs(transmission)**2)))
            boundary_residual=max(boundary_residual,abs(front[0]+normal*front[2]-2*normal),abs(back[0]-normal*back[2]),abs(front[4]),abs(back[4]))
            pieces=sum(good.absorbed_fraction(dict(e,window=v),plasma) for v in [[0.,.3],[.3,.7],[.7,1.]])
            additivity=max(additivity,abs(full-pieces))
            generator,initial=ref.boundary_state(e,plasma);omega=e['frequency'];lateral=omega*np.sin(e['angle'])
            for z in [.13*e['thickness'],.61*e['thickness']]:
                ex,h,jz,rho=expm(generator*z)@initial;ez=(jz-1j*lateral*h)/(1j*omega);jx=(plasma**2*ex-.01j*lateral*rho)/(.06-1j*omega)
                value=good.mode_fields([z],e,plasma)[0]@coeff
                fields_error=max(fields_error,float(max(abs(value-np.array([ex,ez,h,jx,jz])))))
            if e['angle']==0:normal_longitudinal=max(normal_longitudinal,float(max(abs(coeff[2:]))))
    equivalence=max(float(max(abs(good.predict_at(inputs,p)-bad.predict_at(inputs,p)))) for p in [.85,1.,1.15])
    noiseless=[]
    for plasma in [.85,.9,1.,1.1,1.15]:
        clean_values=ref.predict(inputs,plasma);model=good.Model().fit([dict(input=e,value=float(v),sigma=1e-5) for e,v in zip(inputs,clean_values)])
        noiseless.append(dict(true=plasma,fitted=model.plasma_frequency))
    profile=[]
    for plasma in np.linspace(.85,1.15,121):
        res=(good.predict_at(inputs,plasma)-clean)/sigma;profile.append([float(plasma),float(res@res)])
    minima=[i for i in range(1,len(profile)-1) if profile[i][1]<profile[i-1][1] and profile[i][1]<profile[i+1][1]]
    # Resolve decreasing electron-pressure boundary layers using the anchored
    # four-mode basis, and compare to a separately derived local Maxwell slab.
    limit_cases=[ref.experiment(.5,.8,.7,[0.,1.]),ref.experiment(.5,.8,.7,[.2,.8])]
    limits=[]
    try:
        for beta in [.1,.05,.02,.01,.005]:
            good.PRESSURE_SPEED=beta
            rows=[]
            for e in limit_cases:
                value=good.absorbed_fraction(e,1.,256);local=local_maxwell_heat(e,1.)
                rows.append(dict(input=e,hydrodynamic=value,local=local,relative_error=abs(value-local)/abs(local),quadrature128_to256=abs(value-good.absorbed_fraction(e,1.,128))))
            limits.append(dict(pressure_speed=beta,cases=rows))
    finally:good.PRESSURE_SPEED=.1
    local={}
    for label,source in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/hydrodynamic_heating_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='hydro-heating-'+label+'-') as directory:
            base=Path(directory);shutil.copytree(TASK/'environment',base/'app');shutil.copytree(TASK/'tests',base/'tests');shutil.copy2(source,base/'app/model.py')
            result=subprocess.run([sys.executable,'-m','pytest','-q',str(base/'app/test_public.py'),str(base/'tests/test_hidden.py')],cwd=base/'app',env=dict(os.environ,PYTHONPATH=str(base/'app'),OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True)
            local[label]=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    report=dict(revision=1,noise_trials=count,calibration_seed=617191,noise_seed=617193,controls=controls,physical_checks=dict(calibration_equivalence=equivalence,all_scored_reference_error=reference_error,corner_reference_error=corner_error,quadrature64_to128_error=refinement,full_energy_balance_error=energy,regional_additivity_error=additivity,signed_incidence_symmetry_error=mirror,independent_pointwise_field_error=fields_error,boundary_residual=boundary_residual,minimum_regional_heating=minimum_heat,maximum_total_heating=maximum_heat,normal_incidence_longitudinal_amplitude=normal_longitudinal,pressure_speed_local_limits=limits),noiseless_recovery=noiseless,calibration_profile=profile,calibration_profile_local_minima=minima,
        noise=dict(calibration_passes=sum(s['chi2']<1.5 for s in samples),parameter_passes=sum(abs(s['plasma_frequency']-1)<.03 for s in samples),oracle_passes=sum(max(s['oracle'].values())<.03 for s in samples),shortcut_passes=sum(max(s['shortcut'].values())<.03 for s in samples),maximum_oracle_error=max(max(s['oracle'].values()) for s in samples),minimum_shortcut_error=min(min(s['shortcut'].values()) for s in samples),maximum_chi2=max(s['chi2'] for s in samples),plasma_frequency_range=[min(s['plasma_frequency'] for s in samples),max(s['plasma_frequency'] for s in samples)]),local_controls=local,noise_realizations=samples)
    assert equivalence<1e-12 and reference_error<1e-8 and corner_error<1e-7 and refinement<1e-10
    assert energy<1e-10 and additivity<1e-10 and mirror<1e-12 and fields_error<1e-7 and boundary_residual<1e-11
    assert minimum_heat>=0 and maximum_heat<=1+1e-10 and normal_longitudinal<1e-12
    assert len(minima)==1 and abs(profile[minima[0]][0]-1)<1e-12
    assert max(abs(r['fitted']-r['true']) for r in noiseless)<1e-7
    assert all(limits[-1]['cases'][i]['relative_error']<limits[0]['cases'][i]['relative_error'] for i in range(2))
    assert all(row['quadrature128_to256']<1e-10 for item in limits for row in item['cases'])
    assert all(report['noise'][key]==count for key in ['calibration_passes','parameter_passes','oracle_passes'])
    assert report['noise']['shortcut_passes']==0 and report['noise']['minimum_shortcut_error']>.03
    assert local['oracle']['returncode']==0 and local['shortcut']['returncode']==1
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args()
    if args.generate:
        inputs=ref.calibration_inputs();clean=ref.predict(inputs,1.);rng=np.random.default_rng(617191)
        records=[dict(input=e,value=float(v+1e-5*rng.normal()),sigma=1e-5) for e,v in zip(inputs,clean)]
        for area in ['environment','tests']:(TASK/area/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    report=run(args.noise_trials)
    (ROOT/'results/hydrodynamic-heating-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['noise_realizations','calibration_profile','local_controls']},indent=2))
