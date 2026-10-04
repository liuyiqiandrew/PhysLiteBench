"""Validate the equilibrium molecular force slope and both completed controls."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigh,expm

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/spin-strain-response'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

oracle=load('oracle',TASK/'solution/model.py')
shortcut=load('shortcut',ROOT/'scripts/spin_strain_response_baseline.py')
reference=load('reference',TASK/'tests/reference.py')
META=json.loads((TASK/'tests/metadata.json').read_text())


def displaced_force(e,coupling,x,offset=0.):
    h=(e['anisotropy']-coupling*x)*reference.Q+e['transverse']*reference.SX+e['longitudinal']*reference.SZ+offset*np.eye(3)
    density=expm(-h/e['temperature']);density/=np.trace(density)
    return float(coupling*np.trace(reference.Q@density))


def displacement_slope(e,coupling,step=.001):
    f=lambda x:displaced_force(e,coupling,x)
    derivative=lambda h:(-f(2*h)+8*f(h)-8*f(-h)+f(-2*h))/(12*h)
    return (16*derivative(step/2)-derivative(step))/15


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args();start=time.perf_counter()
    cal=reference.calibration_inputs();truth=reference.predict(cal);sigma=META['sigma']
    if args.generate:
        rng=np.random.default_rng(META['calibration_seed'])
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,truth+rng.normal(0,sigma,len(cal)))]
        for rel in ['environment/data/calibration.json','tests/data/calibration.json']:(TASK/rel).write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    assert len(records)==216 and all(r['sigma']==sigma for r in records)
    unit=oracle.predict_at(cal,1.);wrong_unit=shortcut.predict_at(cal,1.)
    calibration_error=float(np.max(abs(unit-wrong_unit)));assert calibration_error<1e-14
    reference_calibration_error=float(np.max(abs(reference.predict(cal,1.)-unit)));assert reference_calibration_error<1e-12
    groups=reference.hidden_inputs();targets={k:reference.predict(v) for k,v in groups.items()}
    scales={k:float(np.sqrt(np.mean(y*y))) for k,y in targets.items()}
    hidden_error=max(float(np.max(abs(oracle.predict_at(v,reference.TRUE_PARAMETER)-targets[k]))) for k,v in groups.items());assert hidden_error<1e-12
    controls={}
    for label,module in [('oracle',oracle),('shortcut',shortcut)]:
        model=module.Model().fit(records)
        residual=(model.predict(cal)-np.array([r['value'] for r in records]))/sigma
        errors={k:float(np.sqrt(np.mean((model.predict(v)-targets[k])**2))/scales[k]) for k,v in groups.items()}
        controls[label]={'coupling':model.coupling,'parameter_relative_error_max':abs(model.coupling/reference.TRUE_PARAMETER-1),'calibration_chi2':float(residual@residual/(len(cal)-1)),'hidden':errors}
        assert controls[label]['calibration_chi2']<1.5 and controls[label]['parameter_relative_error_max']<.03
        assert all((v<META['prediction_limit'])==(label=='oracle' or k=='commuting_anchors') for k,v in errors.items())
    rng=np.random.default_rng(META['noise_seed']);y=truth[None,:]+rng.normal(0,sigma,(256,len(cal)))
    coupling_squared=y@unit/(unit@unit);couplings=np.sqrt(coupling_squared)
    residual=(y-coupling_squared[:,None]*unit[None,:])/sigma
    chi2=np.sum(residual**2,axis=1)/(len(cal)-1);relative=abs(couplings/reference.TRUE_PARAMETER-1)
    assert np.max(chi2)<1.5 and np.max(relative)<.03 and np.min(couplings)>.8 and np.max(couplings)<1.4
    extrema={}
    for label,module in [('oracle',oracle),('shortcut',shortcut)]:
        extrema[label]={}
        for k,v in groups.items():
            base=module.predict_at(v,1.)
            errors=np.sqrt(np.mean((coupling_squared[:,None]*base-targets[k])**2,axis=1))/scales[k]
            extrema[label][k]={'min':float(errors.min()),'max':float(errors.max())}
            assert np.all((errors<META['prediction_limit'])==(label=='oracle' or k=='commuting_anchors'))
    corner_error=0.;finite_step_error=0.;refinement_error=0.;commuting_closed_error=0.;gauge_error=0.;symmetry_error=0.;min_response=1.;min_quantum_correction=1.;max_response=0.;count=0
    for d,h,z,t,g in itertools.product([.7,1.,1.3],[0.,.7,1.5],[-.7,0.,.7],[.2,.5,.8],[.8,1.1,1.4]):
        e=reference.experiment(d,h,z,t);exact=float(oracle.predict_at([e],g)[0]);ref=float(reference.predict([e],g)[0]);approx=float(shortcut.predict_at([e],g)[0])
        corner_error=max(corner_error,abs(exact-ref));min_response=min(min_response,exact);max_response=max(max_response,exact);min_quantum_correction=min(min_quantum_correction,approx-exact)
        displaced=displacement_slope(e,g);refined=displacement_slope(e,g,.0005)
        finite_step_error=max(finite_step_error,abs(displaced-exact));refinement_error=max(refinement_error,abs(refined-displaced))
        gauge_error=max(gauge_error,abs(displaced_force(e,g,.02,.73)-displaced_force(e,g,.02)))
        reversed_e=reference.experiment(d,-h,-z,t);symmetry_error=max(symmetry_error,abs(oracle.response(e)-oracle.response(reversed_e)))
        if h==0.:
            w=2*np.exp(-d/t)*np.cosh(z/t);closed=g*g*w/(t*(1+w)**2)
            commuting_closed_error=max(commuting_closed_error,abs(closed-exact))
        count+=1
    assert corner_error<1e-11 and finite_step_error<1e-9 and refinement_error<1e-9
    assert min_response>0 and min_quantum_correction>-1e-12 and gauge_error<1e-12 and symmetry_error<1e-12 and commuting_closed_error<1e-12
    e=reference.experiment(1.,1.2,.3,1e4);classical_error=abs(oracle.response(e)/shortcut.response(e)-1);assert classical_error<1e-7
    e=reference.experiment(1.,1.2,.3,.005);h=e['anisotropy']*oracle.Q+e['transverse']*oracle.SX+e['longitudinal']*oracle.SZ;energy,V=eigh(h);A=V.T@oracle.Q@V
    ground=2*sum(A[0,j]**2/(energy[j]-energy[0]) for j in [1,2]);ground_error=abs(oracle.response(e)-ground);assert ground_error<1e-12
    # Exact positive-parameter identifiability and numerical endpoint recovery.
    recoveries={}
    for parameter in [.8,1.1,1.4]:
        noiseless=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,reference.predict(cal,parameter))]
        fit=oracle.Model().fit(noiseless).coupling;recoveries[str(parameter)]=abs(fit-parameter);assert abs(fit-parameter)<1e-12
    signal=np.concatenate([v for k,v in targets.items() if k!='commuting_anchors'])
    report={'task':'spin-strain-response','revision':1,'noise_trials':256,'calibration_seed':META['calibration_seed'],'noise_seed':META['noise_seed'],'controls':controls,'noise':{'oracle_passes':256,'shortcut_rejections':256,'all_calibration_pass':True,'max_calibration_chi2':float(chi2.max()),'max_coupling_relative_error':float(relative.max())},'noise_extrema':extrema,'physical_checks':{'commuting_calibration_equality':calibration_error,'calibration_reference_error':reference_calibration_error,'independent_hidden_error':hidden_error,'public_corner_count':count,'corner_frechet_error':corner_error,'finite_displacement_error':finite_step_error,'finite_displacement_refinement':refinement_error,'commuting_closed_form_error':commuting_closed_error,'minimum_response':min_response,'maximum_response':max_response,'minimum_quantum_variance_correction':min_quantum_correction,'energy_gauge_error':gauge_error,'field_reversal_error':symmetry_error,'high_temperature_relative_error':classical_error,'zero_temperature_virtual_transition_error':ground_error,'noiseless_endpoint_parameter_recovery':recoveries,'squared_coupling_information':float(unit@unit/sigma**2),'minimum_scored_signal':float(signal.min()),'maximum_scored_signal':float(signal.max())},'seconds':time.perf_counter()-start,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'environment/data/calibration.json',TASK/'tests/metadata.json',ROOT/'scripts/spin_strain_response_baseline.py',Path(__file__)]}}
    for p in [ROOT/'jobs/spin-strain-response-validation/summary.json',ROOT/'results/spin-strain-response-validation.json']:
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
