"""Validate local spin absorption against an independent finite Fock space."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigh
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/paired-spin-noise'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

oracle=load('oracle',TASK/'solution/model.py')
shortcut=load('shortcut',ROOT/'scripts/paired_spin_noise_baseline.py')
reference=load('reference',TASK/'tests/reference.py')
META=json.loads((TASK/'tests/metadata.json').read_text())


def reduced_rate(e):
    # A particle-hole transformation on only the down-spin orbitals gives
    # six canonical fermionic variables and a number-conserving Hamiltonian.
    B=oracle.energy_matrix(e)
    pick=[0,2,4,7,9,11]
    energy,V=eigh(B[np.ix_(pick,pick)])
    f=expit(-energy/e['temperature'])
    vertex=np.zeros((6,6));vertex[e['site'],e['site']]=.5
    vertex[e['site']+3,e['site']+3]=.5
    vertex=V.conj().T@vertex@V
    gap=energy[:,None]-energy[None,:]
    band=np.where(gap>0.,gap**2*np.exp(-.5*((gap-e['center'])/e['width'])**2),0.)
    return float(np.sum(abs(vertex)**2*(1-f[:,None])*f[None,:]*band))


def spectral_checks(e):
    B=oracle.energy_matrix(e);energy,V=eigh(B)
    kernel=np.einsum('ar,br,r->abr',V,V.conj(),expit(energy/e['temperature']))
    site=e['site'];a=2*site;b=a+1
    covariance=kernel.sum(axis=2);n=covariance[a+6,a+6].real
    equal_time=.5*(n*(1-n)-abs(covariance[a,b+6])**2)
    _,weights=oracle.spectrum(e)
    total=np.zeros((12,12),complex)
    signs=np.tile([.5,-.5],3)
    for i in range(6):
        for j in range(6):
            total+=signs[i]*signs[j]*(np.outer(kernel[i+6,j+6],kernel[i,j])-np.outer(kernel[i+6,j],kernel[i,j+6]))
    freq=energy[:,None]+energy[None,:]
    finite=abs(freq)>1e-8
    conservation=abs(np.sum(total[finite]))
    errors={};negative={}
    for label,module in [('oracle',oracle),('shortcut',shortcut)]:
        f,w=module.spectrum(e);bins={}
        for x,y in zip(f.ravel(),w.ravel()):
            key=round(float(x),9);bins[key]=bins.get(key,0.)+float(y)
        errors[label]=max((abs(bins.get(round(-x,9),0.)-np.exp(-x/e['temperature'])*z) for x,z in bins.items() if x>0),default=0.)
        negative[label]=min(bins.values())
    return abs(weights.sum()-equal_time),conservation,errors,negative


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args();start=time.perf_counter()
    cal=reference.calibration_inputs();truth=reference.predict(cal);sigma=META['sigma']
    if args.generate:
        rng=np.random.default_rng(META['calibration_seed'])
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,truth+rng.normal(0,sigma,len(cal)))]
        for rel in ['environment/data/calibration.json','tests/data/calibration.json']:(TASK/rel).write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    assert len(records)==144 and all(r['sigma']==sigma for r in records)
    x=oracle.predict_at(cal,1.);xb=shortcut.predict_at(cal,1.)
    calibration_error=float(np.max(abs(x-xb)));assert calibration_error<1e-14
    refcal_error=float(np.max(abs(reference.predict(cal,1.)-x)));assert refcal_error<1e-12
    groups=reference.hidden_inputs();targets={k:reference.predict(v) for k,v in groups.items()}
    actual={k:oracle.predict_at(v,reference.TRUE_PARAMETER) for k,v in groups.items()}
    scales={k:float(np.sqrt(np.mean(y*y))) for k,y in targets.items()}
    hidden_error=max(float(np.max(abs(actual[k]-targets[k]))) for k in groups);assert hidden_error<1e-12
    controls={}
    for label,module in [('oracle',oracle),('shortcut',shortcut)]:
        model=module.Model().fit(records)
        residual=(model.predict(cal)-np.array([r['value'] for r in records]))/sigma
        errors={k:float(np.sqrt(np.mean((model.predict(v)-targets[k])**2))/scales[k]) for k,v in groups.items()}
        controls[label]={'gain':model.gain,'parameter_relative_error_max':abs(model.gain/reference.TRUE_PARAMETER-1),'calibration_chi2':float(residual@residual/(len(cal)-1)),'hidden':errors}
        assert controls[label]['calibration_chi2']<1.5 and controls[label]['parameter_relative_error_max']<.03
        assert all((v<META['prediction_limit'])==(label=='oracle' or k=='normal_anchors') for k,v in errors.items())
    rng=np.random.default_rng(META['noise_seed']);y=truth[None,:]+rng.normal(0,sigma,(256,len(cal)))
    gains=y@x/(x@x);residual=(y-gains[:,None]*x[None,:])/sigma
    chi2=np.sum(residual**2,axis=1)/(len(cal)-1)
    rel=abs(gains/reference.TRUE_PARAMETER-1)
    assert np.max(chi2)<1.5 and np.max(rel)<.03
    extrema={}
    for label,module in [('oracle',oracle),('shortcut',shortcut)]:
        extrema[label]={}
        for k,v in groups.items():
            unit=module.predict_at(v,1.)
            errors=np.sqrt(np.mean((gains[:,None]*unit-targets[k])**2,axis=1))/scales[k]
            extrema[label][k]={'min':float(errors.min()),'max':float(errors.max())}
            assert np.all((errors<META['prediction_limit'])==(label=='oracle' or k=='normal_anchors'))
    domain_error=0.;reduced_error=0.;equal_time_error=0.;global_spin_error=0.;kms={'oracle':0.,'shortcut':0.};minline={'oracle':1.,'shortcut':1.}
    corner_count=0
    for h,g,T,i,c,w,a,o in itertools.product([.45,1.],[0.,1.4],[.2,.8],[0,1,2],[.8,4.],[.25,.75],[-1.5,1.5],[-.25,.25]):
        e=reference.experiment(hopping=h,pairing=g,temperature=T,site=i,center=c,width=w,phase=a,offset=o)
        exact=oracle.response(e);ref=reference.rate(e)
        domain_error=max(domain_error,abs(exact-ref));reduced_error=max(reduced_error,abs(exact-reduced_rate(e)))
        assert exact>=-1e-13 and shortcut.response(e)>=-1e-13
        corner_count+=1
    for experiments in groups.values():
        for e in experiments:
            z,q,k,p=spectral_checks(e);equal_time_error=max(equal_time_error,z);global_spin_error=max(global_spin_error,q)
            for label in kms:kms[label]=max(kms[label],k[label]);minline[label]=min(minline[label],p[label])
    assert domain_error<1e-11 and reduced_error<1e-11 and equal_time_error<1e-12 and global_spin_error<1e-12
    assert max(kms.values())<1e-9 and min(minline.values())>-1e-12
    # The physical global-spin cancellation is not inherited by the approximation.
    e=reference.experiment();B=oracle.energy_matrix(e);energies,V=eigh(B);K=np.einsum('ar,br,r->abr',V,V.conj(),expit(energies/e['temperature']));coeff=np.zeros((12,12),complex)
    for a in range(6):
        for b in range(6):coeff+=(-1)**(a+b)*.25*np.outer(K[a+6,b+6],K[a,b])
    f=energies[:,None]+energies[None,:];band=np.where(f>0,f*f*np.exp(-.5*((f-e['center'])/e['width'])**2),0.)
    false_global=float(np.sum(coeff.real*band));assert false_global>.05
    nonzero=np.concatenate([v for k,v in targets.items() if k!='normal_anchors'])
    assert np.min(nonzero)>.015
    report={'task':'paired-spin-noise','revision':1,'noise_trials':256,'calibration_seed':META['calibration_seed'],'noise_seed':META['noise_seed'],'controls':controls,'noise':{'oracle_passes':256,'shortcut_rejections':256,'all_calibration_pass':True,'max_calibration_chi2':float(chi2.max()),'max_gain_relative_error':float(rel.max())},'noise_extrema':extrema,'physical_checks':{'independent_hidden_absolute_error':hidden_error,'normal_calibration_closure_equality':calibration_error,'normal_calibration_fock_error':refcal_error,'public_domain_corners':corner_count,'max_corner_fock_error':domain_error,'max_reduced_canonical_basis_error':reduced_error,'equal_time_spin_sum_rule_error':equal_time_error,'total_spin_finite_frequency_weight_error':global_spin_error,'shortcut_spurious_global_spin_rate_diagnostic':false_global,'kms_error':kms,'minimum_grouped_spectral_line':minline,'minimum_scored_paired_rate':float(nonzero.min()),'maximum_scored_paired_rate':float(nonzero.max()),'gain_information':float(x@x/sigma**2)},'seconds':time.perf_counter()-start,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'environment/data/calibration.json',TASK/'tests/metadata.json',ROOT/'scripts/paired_spin_noise_baseline.py',Path(__file__)]}}
    for p in [ROOT/'jobs/paired-spin-noise-validation/summary.json',ROOT/'results/paired-spin-noise-validation.json']:
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
