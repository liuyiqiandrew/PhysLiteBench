"""Validate the four-level rotor against its finite-gap laboratory Hamiltonian."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1];TASK=ROOT/'tasks/geometric-rotor'
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
oracle=load('oracle',TASK/'solution/model.py')
shortcut=load('shortcut',ROOT/'scripts/geometric_rotor_baseline.py')
reference=load('reference',TASK/'tests/reference.py')
META=json.loads((TASK/'tests/metadata.json').read_text())
ANCHOR='central_holonomy_anchors'


def fast_predict(inputs,inertia):
    energy=np.array([oracle.levels(e['theta'],e['q'],e['r']) for e in inputs])/inertia
    temperature=np.array([e['temperature'] for e in inputs])[:,None]
    w=np.exp(-(energy-energy.min(axis=1)[:,None])/temperature)
    return np.sum(w*energy,axis=1)/w.sum(axis=1)


def infer(inputs,y,sigma):
    def objective(I):
        r=(fast_predict(inputs,I)-y)/sigma
        return float(r@r)
    opt=minimize_scalar(objective,bounds=(.8,1.2),method='bounded',options={'xatol':1e-13})
    return float(min([.8,1.2,opt.x],key=objective))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args();start=time.perf_counter()
    inputs=reference.calibration_inputs();truth=reference.predict(inputs);sigma=META['sigma']
    if args.generate:
        rng=np.random.default_rng(META['calibration_seed']);noisy=truth+rng.normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,noisy)]
        for rel in ['environment/data/calibration.json','tests/data/calibration.json']:(TASK/rel).write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text());assert len(records)==288 and all(r['sigma']==sigma for r in records)
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    nominal=np.array([r['value'] for r in records]);groups=reference.hidden_inputs();targets={k:reference.predict(v) for k,v in groups.items()};scales={k:np.sqrt(np.mean(v*v)) for k,v in targets.items()}
    cal_equality=float(np.max(abs(oracle.predict_at(inputs,1.07)-shortcut.predict_at(inputs,1.07))));assert cal_equality<1e-11
    controls={}
    for label,m in [('oracle',oracle),('shortcut',shortcut)]:
        model=m.Model().fit(records);r=(model.predict(inputs)-nominal)/sigma
        errors={k:float(np.sqrt(np.mean((model.predict(v)-targets[k])**2))/scales[k]) for k,v in groups.items()}
        controls[label]={'inertia':model.inertia,'parameter_relative_error_max':abs(model.inertia/reference.TRUE_PARAMETER-1),'calibration_chi2':float(r@r/(len(r)-1)),'hidden':errors}
        assert controls[label]['calibration_chi2']<1.5 and controls[label]['parameter_relative_error_max']<.03
        assert all((x<META['prediction_limit'])==(label=='oracle' or k==ANCHOR) for k,x in errors.items())
    fast_match=abs(infer(inputs,nominal,sigma)-controls['oracle']['inertia']);assert fast_match<1e-8
    rng=np.random.default_rng(META['noise_seed']);noisy=truth[None,:]+rng.normal(0,sigma,(256,len(inputs)))
    fitted=np.array([infer(inputs,y,sigma) for y in noisy]);chi=np.array([np.sum(((fast_predict(inputs,I)-y)/sigma)**2)/(len(inputs)-1) for I,y in zip(fitted,noisy)])
    rel=abs(fitted/reference.TRUE_PARAMETER-1);assert rel.max()<.03 and chi.max()<1.5
    extrema={}
    for label,m in [('oracle',oracle),('shortcut',shortcut)]:
        extrema[label]={}
        for k,v in groups.items():
            errors=np.array([np.sqrt(np.mean((m.predict_at(v,I)-targets[k])**2))/scales[k] for I in fitted])
            extrema[label][k]={'min':float(errors.min()),'max':float(errors.max())}
            assert np.all((errors<META['prediction_limit'])==(label=='oracle' or k==ANCHOR))
    profiles={};grid=np.linspace(.8,1.2,161);allcurves=np.array([fast_predict(inputs,x) for x in grid]);monotonic=float(np.max(np.diff(allcurves,axis=0)));assert monotonic<0
    for true in [.8,1.07,1.2]:
        y=fast_predict(inputs,true);objectives=np.sum(((allcurves-y)/sigma)**2,axis=1)
        fitted_true=infer(inputs,y,sigma);profiles[str(true)]={'recovered':fitted_true,'absolute_error':abs(fitted_true-true),'grid_minima':int(sum(objectives[j]<=objectives[max(0,j-1)] and objectives[j]<=objectives[min(len(grid)-1,j+1)] for j in range(len(grid))))}
        assert abs(fitted_true-true)<1e-7 and profiles[str(true)]['grid_minima']==1
    reference_error=0.;source_scalar_error=0.;gap_refinement=0.;fourier_refinement=0.;projection_A=0.;projection_phi=0.;minmetric=1.;noncommuting=0.;cases=0
    for I,q,r,theta,T in itertools.product([.8,1.07,1.2],[1,2,3],[1,2],[.3,.85,np.pi/2],[.04,.3]):
        e=reference.experiment(theta,T,q,r);value=oracle.predict_at([e],I)[0];ref=reference.response(e,I)
        reference_error=max(reference_error,abs(value-ref));cases+=1
        source_value=shortcut.predict_at([e],I)[0]
        assert value>0 and source_value>0
        n=np.arange(-24,25);shift=((q+r)%2)/2
        scalar=np.diag((n-shift)**2/(2*I)+np.sin(theta)**2*(r*r+q*q/2)/(8*I))
        for j in range(len(n)-2*r):scalar[j,j+2*r]=scalar[j+2*r,j]=-np.sin(theta)**2*q*q/(32*I)
        levels=eigh(scalar,eigvals_only=True);weights=np.exp(-(levels-levels[0])/T)
        source_scalar_error=max(source_scalar_error,abs(source_value-float(weights@levels/weights.sum())))
    for e in [reference.experiment(.45,.07,2,1),reference.experiment(.4,.1,3,2),reference.experiment(np.pi/2,.3,3,2)]:
        a=reference.response(e,1.07);gap_refinement=max(gap_refinement,abs(a-reference.response(e,1.07,base_gap=2048)))
        fourier_refinement=max(fourier_refinement,abs(a-reference.response(e,1.07,cutoff=24)))
    Y=np.array([[0,-1j],[1j,0]])
    for phi,theta,q,r in itertools.product([.2,1.1,2.7],[.3,.85,np.pi/2],[1,2,3],[1,2]):
        U,dU=shortcut.frame(phi,theta,q,r);V=U[:,:2];dV=dU[:,:2]
        A=1j*V.conj().T@dV;expected=(q*np.cos(r*phi)*shortcut.X-q*np.cos(theta)*np.sin(r*phi)*Y+r*np.cos(theta)*shortcut.Z)/2
        C=dV.conj().T@(np.eye(4)-V@V.conj().T)@dV;expectedC=np.eye(2)*np.sin(theta)**2*(q*q*np.sin(r*phi)**2+r*r)/4
        projection_A=max(projection_A,float(np.max(abs(A-expected))));projection_phi=max(projection_phi,float(np.max(abs(C-expectedC))));minmetric=min(minmetric,float(eigh(C,eigvals_only=True).min()))
    assert reference_error<3e-6 and gap_refinement<3e-6 and fourier_refinement<1e-9
    assert source_scalar_error<1e-11
    assert projection_A<1e-12 and projection_phi<1e-12 and minmetric>=-1e-12
    # A separate finite-temperature zero-texture physical limit: the full
    # projected spectrum reduces to an ordinary rotor; the source may not.
    zero_e=reference.experiment(0.,.12,2,1);n=np.arange(-24,25);energy=n*n/(2*1.07);w=np.exp(-energy/.12);free=float(w@energy/w.sum())
    zero_correct=oracle.predict_at([zero_e],1.07)[0];zero_shortcut=shortcut.predict_at([zero_e],1.07)[0]
    assert abs(zero_correct-free)<1e-10
    for theta,q,r in [(.6,2,1),(.8,3,2)]:
        def A(phi):return (q*np.cos(r*phi)*shortcut.X-q*np.cos(theta)*np.sin(r*phi)*Y+r*np.cos(theta)*shortcut.Z)/2
        noncommuting=max(noncommuting,float(np.linalg.norm(A(.2)@A(1.2)-A(1.2)@A(.2))))
    selected=np.concatenate([v for k,v in targets.items() if k!=ANCHOR]);assert selected.min()>.05
    report={'task':'geometric-rotor','revision':2,'noise_trials':256,'controls':controls,'noise':{'oracle_passes':256,'shortcut_rejections':256,'all_calibration_pass':True,'max_calibration_chi2':float(chi.max()),'max_inertia_relative_error':float(rel.max()),'fast_fit_vs_model_error':fast_match},'noise_extrema':extrema,'physical_checks':{'calibration_closure_equality':cal_equality,'starter_matrix_vs_scalar_prototype_error':source_scalar_error,'reference_cases':cases,'max_original_lab_reference_error':reference_error,'gap_refinement':gap_refinement,'fourier_refinement':fourier_refinement,'connection_projection_error':projection_A,'geometric_metric_projection_error':projection_phi,'minimum_local_metric_eigenvalue':minmetric,'connection_commutator_norm':noncommuting,'zero_texture_physical_error':abs(zero_correct-free),'zero_texture_shortcut_defect':abs(zero_shortcut-free),'minimum_scored_signal':float(selected.min()),'maximum_scored_signal':float(selected.max())},'identifiability':{'calibration_curves_strictly_decrease_with_inertia':True,'maximum_grid_difference':monotonic,'profiles':profiles},'calibration_seed':META['calibration_seed'],'noise_seed':META['noise_seed'],'seconds':time.perf_counter()-start,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'environment/data/calibration.json',TASK/'tests/metadata.json',ROOT/'scripts/geometric_rotor_baseline.py',Path(__file__)]}}
    for p in [ROOT/'jobs/geometric-rotor-r2-validation/summary.json',ROOT/'results/geometric-rotor-r2-validation.json']:
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
