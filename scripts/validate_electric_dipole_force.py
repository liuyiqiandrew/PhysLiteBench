"""Validate electric point-dipole force using independent Maxwell stress."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/electric-dipole-force'


def module(path):
    spec=importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def physical_checks(good,bad,ref):
    import copy
    from numpy.polynomial.legendre import leggauss
    cal=ref.calibration_inputs();hidden=sum(ref.hidden_inputs().values(),[])
    agreement=max(abs(good.predict_at(hidden,1.1)-ref.predict(hidden)))
    radius_error=max(float(max(abs(ref.predict(hidden,radius=r)-ref.predict(hidden)))) for r in [.25,.7])
    angular_error=float(max(abs(ref.predict(hidden,order=32)-ref.predict(hidden))))
    cal_equivalence=float(max(abs(good.predict_at(cal,1.1)-bad.predict_at(cal,1.1))))
    cal_reference=float(max(abs(good.predict_at(cal,1.1)-ref.predict(cal))))
    passivity=max(abs(good.polarizability(s).imag-abs(good.polarizability(s))**2/(6*np.pi)) for s in np.linspace(.6,1.6,31))
    analytic=[]
    for theta in [.31,.63,.97]:
        e=ref.experiment([ref.beam(theta),ref.beam(-theta)])
        alpha=good.polarizability(1.1)
        analytic.extend([abs(good.predict_at([e],1.1)[0]-2*alpha.imag*np.cos(theta)**3),abs(bad.predict_at([e],1.1)[0]-2*alpha.imag*np.cos(theta))])
    symmetry=[]
    delta=np.array([.21,-.13,.17]);angle=.73;c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    for e in hidden:
        translated=copy.deepcopy(e);translated['position']=(np.array(e['position'])+delta).tolist()
        for w in translated['waves']:
            amplitude=(np.array(w['real'])+1j*np.array(w['imag']))*np.exp(-1j*np.array(w['direction'])@delta)
            w['real']=amplitude.real.tolist();w['imag']=amplitude.imag.tolist()
        rotated=copy.deepcopy(e)
        for key in ['position','axis']:rotated[key]=(rotation@e[key]).tolist()
        for w in rotated['waves']:
            for key in ['direction','real','imag']:w[key]=(rotation@w[key]).tolist()
        for source in [good,bad]:symmetry.extend(abs(source.predict_at([translated,rotated],1.1)-source.predict_at([e],1.1)[0]))
    # Surface flux and self-force checks use the full retarded dipole fields.
    self_force=[];incident_force=[];radiated=[];power_balance=[]
    z,w=leggauss(32);phi=np.arange(64)*np.pi/32;zz,pp=np.meshgrid(z,phi,indexing='ij')
    n=np.stack([np.sqrt(1-zz*zz)*np.cos(pp),np.sqrt(1-zz*zz)*np.sin(pp),zz],axis=-1).reshape(-1,3)
    def traction(E,H):
        return .5*np.real(E*np.sum(E.conj()*n,axis=1)[:,None]+H*np.sum(H.conj()*n,axis=1)[:,None]-.5*n*np.sum(abs(E)**2+abs(H)**2,axis=1)[:,None])
    def flux(E,H):return .5*np.real(np.sum(np.cross(E,H.conj())*n,axis=1))
    for e in hidden[9:12]:
        center,_=ref.fields(e['waves'],np.array(e['position']));dipole=good.polarizability(1.1)*center
        for r in [.25,.7]:
            weights=np.repeat(w,64)*np.pi/32*r*r
            E,H=ref.fields(e['waves'],np.array(e['position'])+r*n);proj=(n@dipole)[:,None]*n;scale=np.exp(1j*r)/(4*np.pi)
            Es=scale*((dipole-proj)/r+(3*proj-dipole)*(1/r**3-1j/r**2));Hs=scale*(1/r+1j/r**2)*np.cross(n,dipole)
            self_force.append(float(max(abs(weights@traction(Es,Hs)))));incident_force.append(float(max(abs(weights@traction(E,H)))))
            radiated.append(abs(weights@flux(Es,Hs)-np.vdot(dipole,dipole).real/(12*np.pi)))
            power_balance.append(abs(weights@(flux(E+Es,H+Hs)-flux(E,H))))
    corners=[]
    for theta in [.31,1.23]:
        for phase in [-2.7,2.4]:
            for x in [-2.,2.]:
                e=ref.experiment([ref.beam(theta,.3j,.8),ref.beam(-theta,.4,.9,phase)],(x,.13,-.27))
                for strength in [.6,1.6]:corners.append(abs(good.predict_at([e],strength)[0]-ref.predict([e],strength)[0]))
    for e in cal+hidden:
        assert abs(np.linalg.norm(e['axis'])-1)<1e-14 and max(abs(np.array(e['position'])))<=2
        for wave in e['waves']:
            k=np.array(wave['direction']);v=np.array(wave['real'])+1j*np.array(wave['imag'])
            assert abs(np.linalg.norm(k)-1)<1e-14 and abs(k@v)<1e-14 and np.linalg.norm(v)<=1+1e-14 and k[2]>=.3
    noiseless=[]
    for true in [.601,1.1,1.599]:
        exact=good.predict_at(cal,true);records=[dict(input=e,value=float(y),sigma=.00005) for e,y in zip(cal,exact)]
        noiseless.append(abs(good.Model().fit(records).response_strength/true-1))
        profile=[float(np.sum((good.predict_at(cal,g)-exact)**2)) for g in np.linspace(.6,1.6,181)]
        index=int(np.argmin(profile));assert all(np.diff(profile[:index+1])<0) and all(np.diff(profile[index:])>0)
    assert max(agreement,radius_error,angular_error,cal_reference,max(corners))<1e-11
    assert cal_equivalence<1e-14 and passivity<1e-14 and max(analytic)<1e-14 and max(symmetry)<1e-12
    assert max(self_force+incident_force+radiated+power_balance)<1e-11 and max(noiseless)<1e-7
    return {'oracle_maxwell_stress_reference_max':float(agreement),'stress_radius_change_max':radius_error,'stress_angular_refinement_max':angular_error,
        'calibration_reference_max':cal_reference,'calibration_control_difference':cal_equivalence,'optical_theorem_residual':float(passivity),
        'crossed_TM_analytic_limit_max':float(max(analytic)),'translation_rotation_error_both_controls':float(max(symmetry)),
        'dipole_self_force_max':max(self_force),'incident_closed_surface_force_max':max(incident_force),'radiated_power_error':float(max(radiated)),
        'lossless_total_power_flux_residual':float(max(power_balance)),'full_domain_corner_reference_max':float(max(corners)),
        'noiseless_fit_relative_errors':noiseless,'calibration_objective_full_domain_unimodal':True,'public_input_domain_verified':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256)
    args=parser.parse_args();start=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/electric_dipole_force_baseline.py');ref=module(TASK/'tests/reference.py')
    metadata=json.loads((TASK/'tests/metadata.json').read_text());true=ref.TRUE_PARAMETER;sigma=metadata['measurement_sigma'];limit=metadata['prediction_limit']
    inputs=ref.calibration_inputs();exact=ref.predict(inputs,true)
    if args.generate:
        values=exact+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
        for directory in ['environment','tests']:(TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text()) and [r['input'] for r in records]==inputs
    assert all(r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs();truth={name:ref.predict(es,true) for name,es in hidden.items()}
    def chi(model,rs):
        residual=(model.predict(inputs)-np.array([r['value'] for r in rs]))/sigma
        return float(residual@residual/(len(rs)-1))
    def scores(model):return {name:float(np.sqrt(np.mean((model.predict(es)-truth[name])**2)/np.mean(truth[name]**2))) for name,es in hidden.items()}
    report={'revision':1,'calibration_seed':metadata['calibration_seed'],'noise_seed':metadata['noise_seed'],'noise_trials':args.noise_trials,'controls':{}}
    for label,source in [('oracle',good),('shortcut',bad)]:
        model=source.Model().fit(records);result={'parameter':model.response_strength,'parameter_relative_error':abs(model.response_strength/true-1),
            'calibration_chi2':chi(model,records),'hidden':scores(model)}
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        assert (max(result['hidden'].values())<limit if label=='oracle' else min(result['hidden'].values())>limit)
        report['controls'][label]=result
    rng=np.random.default_rng(metadata['noise_seed']);parameters=[];chi2s=[];noise_oracle=[];noise_shortcut=[]
    for _ in range(args.noise_trials):
        sample=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))]
        a=good.Model().fit(sample);b=bad.Model().fit(sample);assert abs(a.response_strength-b.response_strength)<1e-10
        parameters.append(a.response_strength);chi2s.append(chi(a,sample))
        noise_oracle.append(max(scores(a).values()));noise_shortcut.append(min(scores(b).values()))
        assert noise_oracle[-1]<limit and noise_shortcut[-1]>limit
    assert max(chi2s)<1.5 and max(abs(np.array(parameters)/true-1))<.03
    report['noise']={'parameter_min':min(parameters),'parameter_max':max(parameters),'parameter_relative_error_max':float(max(abs(np.array(parameters)/true-1))),
        'calibration_chi2_max':max(chi2s),'calibration_pass_fraction':1.,'oracle_hidden_passes':args.noise_trials,'shortcut_hidden_passes':0,
        'oracle_hidden_max':max(noise_oracle),'shortcut_hidden_min':min(noise_shortcut)}
    extrema={}
    for label,source in [('oracle',good),('shortcut',bad)]:
        errors=[]
        for parameter in [min(parameters),max(parameters)]:
            model=source.Model();model.response_strength=parameter;errors.extend(scores(model).values())
        extrema[label]={'min':min(errors),'max':max(errors)}
    assert extrema['oracle']['max']<limit and extrema['shortcut']['min']>limit
    report['hidden_parameter_extrema_check']=extrema;report['physical_checks']=physical_checks(good,bad,ref);report['seconds']=time.time()-start
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/metadata.json',TASK/'environment/data/calibration.json',ROOT/'scripts/electric_dipole_force_baseline.py',Path(__file__)]}
    output=ROOT/'jobs/electric-dipole-force-validation';output.mkdir(parents=True,exist_ok=True);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'results/electric-dipole-force-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('electric-dipole-force PASS',output/'summary.json',report['seconds'])
    print(json.dumps(report['controls'],indent=2));print(json.dumps(report['physical_checks'],indent=2))


if __name__=='__main__':main()
