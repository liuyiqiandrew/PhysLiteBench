"""Validate pressure storage from compatible poroelastic deformation."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1];TASK=ROOT/'tasks/poroelastic-relaxation'


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


oracle=load('oracle',TASK/'solution/model.py');shortcut=load('shortcut',ROOT/'scripts/poroelastic_relaxation_baseline.py');ref=load('reference',TASK/'tests/reference.py')
META=json.loads((TASK/'tests/metadata.json').read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args();start=time.perf_counter()
    cal=ref.calibration_inputs();truth=ref.predict(cal);sigma=META['sigma']
    if args.generate:
        rng=np.random.default_rng(META['calibration_seed']);records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,truth+rng.normal(0,sigma,len(cal)))]
        for rel in ['environment/data/calibration.json','tests/data/calibration.json']:(TASK/rel).write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text());assert len(records)==216 and all(r['sigma']==sigma for r in records)
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    cal_equal=float(np.max(abs(oracle.predict_at(cal,ref.TRUE_PARAMETER)-shortcut.predict_at(cal,ref.TRUE_PARAMETER))))
    cal_ref=float(np.max(abs(oracle.predict_at(cal,ref.TRUE_PARAMETER)-truth)));assert cal_equal==0 and cal_ref<1e-8
    groups=ref.hidden_inputs();target={k:ref.predict(v) for k,v in groups.items()};norm={k:np.linalg.norm(v) for k,v in target.items()}
    controls={}
    for label,module in [('oracle',oracle),('shortcut',shortcut)]:
        m=module.Model().fit(records);res=(m.predict(cal)-np.array([r['value'] for r in records]))/sigma
        hidden={k:float(np.linalg.norm(m.predict(v)-target[k])/norm[k]) for k,v in groups.items()}
        controls[label]={'viscosity':m.viscosity,'parameter_relative_error_max':abs(m.viscosity/ref.TRUE_PARAMETER-1),'calibration_chi2':float(res@res/(len(cal)-1)),'hidden':hidden}
        assert controls[label]['parameter_relative_error_max']<.03 and controls[label]['calibration_chi2']<1.5
        assert all((x<META['prediction_limit'])==(label=='oracle' or k=='uniform_anchors') for k,x in hidden.items())
    rng=np.random.default_rng(META['noise_seed']);y=truth[None,:]+rng.normal(0,sigma,(256,len(cal)))
    times=np.array([e['time'] for e in cal]);initial=np.array([e['initial_pressure'] for e in cal]);storage,g=oracle.coefficients(cal[0]);decay=g*times/storage
    viscosities=[];chi2=[];global_grid_gap=[]
    grid=np.linspace(.0008,.0015,301);grid_predictions=initial[None,:]*np.exp(-decay[None,:]/grid[:,None])
    for values in y:
        loss=lambda eta:float(np.sum(((initial*np.exp(-decay/eta)-values)/sigma)**2))
        fit=minimize_scalar(loss,bounds=(.0008,.0015),method='bounded',options={'xatol':1e-14})
        viscosities.append(fit.x);chi2.append(fit.fun/(len(cal)-1));global_grid_gap.append(float(np.min(np.sum(((grid_predictions-values)/sigma)**2,axis=1))-fit.fun))
    viscosities=np.array(viscosities);relative=abs(viscosities/ref.TRUE_PARAMETER-1)
    assert max(chi2)<1.5 and relative.max()<.03 and min(global_grid_gap)>-1e-7
    extrema={}
    for label,module in [('oracle',oracle),('shortcut',shortcut)]:
        extrema[label]={}
        for k,v in groups.items():
            error=np.array([np.linalg.norm(module.predict_at(v,eta)-target[k])/norm[k] for eta in viscosities]);extrema[label][k]={'min':float(error.min()),'max':float(error.max())}
            assert np.all((error<META['prediction_limit'])==(label=='oracle' or k=='uniform_anchors'))
    ref_error=0.;relative_storage=0.;force_balance=0.;compatibility=0.;energy_identity=0.;dissipation=0.;reversal=0.;zero_time=0.;min_storage=1.;max_storage=0.;max_strain=0.;source_compatibility=0.;count=0
    for mode in itertools.product(range(-2,3),repeat=2):
        e=ref.experiment(mode);s,g=oracle.coefficients(e);p,energy,strain=ref.fluid_stiffness(e)
        relative_storage=max(relative_storage,abs(s*p-1));min_storage=min(min_storage,s);max_storage=max(max_storage,s)
        energy_identity=max(energy_identity,abs(2*energy/p-1))
        stress=ref.C@strain-ref.A*p*ref.Q
        k=2*np.pi*np.array(mode)/.01
        if any(mode):
            b=np.array([[k[0],0.],[0.,k[1]],[k[1],k[0]]])
            force_balance=max(force_balance,float(np.linalg.norm(b.T@stress)/(np.linalg.norm(b)*p)))
            compatibility=max(compatibility,abs(float(k[1]**2*strain[0]+k[0]**2*strain[1]-k[0]*k[1]*strain[2]))/(np.dot(k,k)*np.linalg.norm(strain)))
            local=np.linalg.solve(ref.C,ref.A*ref.Q)
            source_compatibility=max(source_compatibility,abs(float(k[1]**2*local[0]+k[0]**2*local[1]-k[0]*k[1]*local[2]))/(np.dot(k,k)*np.linalg.norm(local)))
        else:force_balance=max(force_balance,float(np.linalg.norm(stress)/p))
        max_strain=max(max_strain,float(np.max(abs(strain/p*3000))))
        for eta,t,a in itertools.product([.0008,.0011,.0015],[0.,.1,.5,4.],[1000.,3000.]):
            e=ref.experiment(mode,t,a);exact=oracle.predict_at([e],eta)[0];independent=ref.predict([e],eta)[0];ref_error=max(ref_error,abs(exact-independent))
            reversal=max(reversal,abs(exact-oracle.predict_at([dict(e,mode=(-np.array(mode)).tolist())],eta)[0]))
            if t==0:zero_time=max(zero_time,abs(exact-a))
            # Fluid-energy loss equals Darcy plus leakage dissipation for this Fourier coefficient.
            z=exact/p;zdot=-g/eta*exact
            energy_dot=2*energy*z*zdot
            loss=g/eta*exact*exact
            dissipation=max(dissipation,abs(energy_dot+loss)/max(loss,1e-90));assert 0<=exact<=a
            count+=1
    assert ref_error<1e-8 and relative_storage<1e-13 and force_balance<1e-13 and compatibility<1e-13
    assert energy_identity<1e-13 and dissipation<1e-13 and reversal<1e-12 and zero_time==0 and max_strain<.001
    assert source_compatibility>.1 and min_storage>0
    uniform=1/ref.M+ref.A**2*ref.Q@np.linalg.solve(ref.C,ref.Q)
    axis_errors={}
    for mode,index in [((1,0),0),((0,1),1)]:
        expected=1/ref.M+ref.A**2/(ref.C[index,index]-ref.C[index,2]**2/ref.C[2,2])
        axis_errors[str(mode)]=abs(oracle.coefficients(ref.experiment(mode))[0]/expected-1)
    assert max(axis_errors.values())<1e-14
    compliance_sensitivities={}
    for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]:
        delta=1e3;plus=ref.C.copy();minus=ref.C.copy()
        plus[i,j]+=delta;minus[i,j]-=delta
        if i!=j:plus[j,i]+=delta;minus[j,i]-=delta
        sensitivity=(ref.Q@np.linalg.solve(plus,ref.Q)-ref.Q@np.linalg.solve(minus,ref.Q))/(2*delta)
        compliance_sensitivities[str((i,j))]=float(sensitivity)
        assert abs(sensitivity)>1e-17
    recovery={}
    for eta in [.0008,.0011,.0015]:
        clean=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,ref.predict(cal,eta))]
        recovery[str(eta)]=abs(oracle.Model().fit(clean).viscosity/eta-1);assert recovery[str(eta)]<1e-7
    # Monotone response to viscosity guarantees noiseless identifiability in this interval.
    derivative=initial*np.exp(-decay/ref.TRUE_PARAMETER)*decay/ref.TRUE_PARAMETER**2
    assert np.all(derivative>0)
    report={'task':'poroelastic-relaxation','revision':1,'noise_trials':256,'calibration_seed':META['calibration_seed'],'noise_seed':META['noise_seed'],'controls':controls,
        'noise':{'oracle_passes':256,'shortcut_rejections':256,'all_calibration_pass':True,'max_calibration_chi2':max(chi2),'max_parameter_relative_error':float(relative.max()),'minimum_grid_objective_minus_fit':min(global_grid_gap)},'noise_extrema':extrema,
        'physical_checks':{'calibration_equivalence':cal_equal,'calibration_energy_reference_error':cal_ref,'public_cases':count,'independent_pressure_error_Pa':ref_error,'inverse_storage_identity_error':relative_storage,'force_balance_relative_error':force_balance,'compatible_strain_relative_error':compatibility,'shortcut_incompatible_strain_diagnostic':source_compatibility,'energy_identity_relative_error':energy_identity,'Darcy_leakage_dissipation_relative_error':dissipation,'mode_reversal_error':reversal,'initial_condition_error':zero_time,'minimum_storage_per_Pa':min_storage,'maximum_storage_per_Pa':max_storage,'uniform_storage_per_Pa':float(uniform),'maximum_strain_at_public_amplitude':max_strain,'axis_plane_strain_limit_relative_error':axis_errors,'local_compliance_elastic_entry_sensitivities':compliance_sensitivities,'parameter_endpoint_recovery':recovery,'viscosity_information':float(derivative@derivative/sigma**2),'minimum_scored_signal_Pa':float(min(v.min() for k,v in target.items() if k!='uniform_anchors')),'maximum_scored_signal_Pa':float(max(v.max() for k,v in target.items() if k!='uniform_anchors'))},'seconds':time.perf_counter()-start,
        'source_sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'environment/data/calibration.json',TASK/'tests/metadata.json',ROOT/'scripts/poroelastic_relaxation_baseline.py',Path(__file__)]}}
    (ROOT/'results/poroelastic-relaxation-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
