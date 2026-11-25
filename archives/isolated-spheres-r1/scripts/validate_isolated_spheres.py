"""Validate conserved-charge thermal forces and the independent bispherical reference."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import gammaln

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/isolated-spheres'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

oracle=load('sphere_oracle',TASK/'solution/model.py')
source=load('sphere_source',ROOT/'scripts/isolated_spheres_baseline.py')
ref=load('sphere_reference',TASK/'tests/reference.py')
meta=json.loads((TASK/'tests/metadata.json').read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--generate',action='store_true');args=ap.parse_args()
    start=time.perf_counter();cal=ref.calibration_inputs();truth=ref.predict(cal);sigma=meta['sigma']
    if args.generate:
        noise=np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(cal))
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,truth+noise)]
        for rel in ['environment/data/calibration.json','tests/data/calibration.json']:
            (TASK/rel).write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    assert len(records)==240 and all(r['sigma']==sigma for r in records)
    assert all(r['input']['kind']=='dipole' for r in records)
    assert np.max(abs(oracle.predict_at(cal,ref.TRUE_PARAMETER)-source.predict_at(cal,ref.TRUE_PARAMETER)))==0
    groups=ref.hidden_inputs();targets={k:ref.predict(es) for k,es in groups.items()}
    def errors(module,radius):
        return {k:float(np.linalg.norm(module.predict_at(es,radius)-targets[k])/np.linalg.norm(targets[k])) for k,es in groups.items()}
    controls={}
    for label,module in [('oracle',oracle),('shortcut',source)]:
        fitted=module.Model().fit(records)
        residual=(fitted.predict(cal)-np.array([r['value'] for r in records]))/sigma
        controls[label]=dict(radius=fitted.radius,parameter_relative_error=abs(fitted.radius/ref.TRUE_PARAMETER-1),calibration_chi2=float(residual@residual/(len(cal)-1)),hidden=errors(module,fitted.radius))
        assert controls[label]['parameter_relative_error']<.03 and controls[label]['calibration_chi2']<1.5
        assert all((v<.04)==(label=='oracle' or k=='dipole_anchors') for k,v in controls[label]['hidden'].items())
    trials=[];extrema={label:{k:[] for k in groups} for label in ['oracle','shortcut']}
    rng=np.random.default_rng(meta['noise_seed'])
    for trial in range(256):
        values=truth+rng.normal(0,sigma,len(cal))
        noisy=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,values)]
        models={label:module.Model().fit(noisy) for label,module in [('oracle',oracle),('shortcut',source)]}
        assert models['oracle'].radius==models['shortcut'].radius
        radius=models['oracle'].radius;residual=(models['oracle'].predict(cal)-values)/sigma
        chi=float(residual@residual/(len(cal)-1));perror=abs(radius/ref.TRUE_PARAMETER-1)
        assert chi<1.5 and perror<.03
        for label,module in [('oracle',oracle),('shortcut',source)]:
            es=errors(module,radius)
            for k,v in es.items():
                extrema[label][k].append(v)
                assert (v<.04)==(label=='oracle' or k=='dipole_anchors')
        trials.append(dict(radius=radius,calibration_chi2=chi,parameter_relative_error=perror))
    summary={label:{k:dict(min=float(np.min(v)),max=float(np.max(v))) for k,v in by.items()} for label,by in extrema.items()}
    # Independent bispherical reference, multipole and series refinement, and separation work.
    rng=np.random.default_rng(710447)
    cases=list(itertools.product([.85,1.03,1.15],[3.,3.3,3.7,4.1,4.5]))
    cases.extend(zip(rng.uniform(.85,1.15,33),rng.uniform(3.,4.5,33)))
    reference_error=grounded_error=refinement=series_refinement=work_error=scaling=0.
    minimum_block_eigenvalue=1.;max_charge_variance=0.;min_force=np.inf;max_force=0.
    for a,d in cases:
        actual=oracle.pair_force(d,a);short=source.pair_force(d,a)
        energy,expected=ref.free_energy_force(d,a)
        reference_error=max(reference_error,abs(actual-expected))
        grounded_error=max(grounded_error,abs(short-ref.free_energy_force(d,a,False)[1]))
        refinement=max(refinement,abs(actual-oracle.pair_force(d,a,48)),abs(short-source.pair_force(d,a,48)))
        series_refinement=max(series_refinement,abs(expected-ref.free_energy_force(d,a,terms=192)[1]))
        h=2e-4
        fm2,fm1,fp1,fp2=[ref.free_energy_force(d+j*h,a)[0] for j in [-2,-1,1,2]]
        force_difference=-(fm2-8*fm1+8*fp1-fp2)/(12*h)
        work_error=max(work_error,abs(actual-force_difference))
        scaling=max(scaling,abs(actual-oracle.pair_force(d/a,1)/a))
        assert actual<0 and short<actual and d>2*a
        min_force=min(min_force,abs(actual));max_force=max(max_force,abs(actual))
        for m in [0,1,2]:
            ell=np.arange(m,37);left=ell[:,None];right=ell[None,:]
            logs=gammaln(left+right+1)-.5*(gammaln(left+m+1)+gammaln(left-m+1)+gammaln(right+m+1)+gammaln(right-m+1))
            coupling=np.exp(logs+(left+right+1)*np.log(a/d))
            margin=1-np.linalg.svd(coupling,compute_uv=False)[0]
            minimum_block_eigenvalue=min(minimum_block_eigenvalue,float(margin));assert margin>0
            if m==0:
                covariance=np.linalg.inv(np.eye(len(ell))-coupling@coupling)
                max_charge_variance=max(max_charge_variance,float(covariance[0,0]))
    assert reference_error<1e-10 and grounded_error<1e-10 and refinement<1e-10 and series_refinement<1e-13
    assert work_error<1e-9 and scaling<1e-13 and minimum_block_eigenvalue>.4
    far=[]
    for d in [20.,40.,80.]:
        far.append(dict(separation=d,grounded=source.pair_force(d,1,12)*d**3/(-1),isolated=oracle.pair_force(d,1,12)*d**7/(-18)))
    assert abs(far[-1]['grounded']-1)<.001 and abs(far[-1]['isolated']-1)<.002
    assert all(abs(far[i+1][k]-1)<abs(far[i][k]-1) for i in [0,1] for k in ['grounded','isolated'])
    recovery={};fields=np.array([e['field'] for e in cal])
    for a in np.linspace(.85,1.15,41):
        exact=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,ref.predict(cal,a))]
        recovery[str(a)]=abs(oracle.Model().fit(exact).radius/a-1)
    assert max(recovery.values())<1e-14
    # The least-squares objective is a positive quadratic in a^3; a>0 makes the parameter unique.
    information=9*ref.TRUE_PARAMETER**4*float(fields@fields)/sigma**2
    signal=np.concatenate([v for k,v in targets.items() if k!='dipole_anchors'])
    files=[TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'environment/data/calibration.json',TASK/'tests/metadata.json',ROOT/'scripts/isolated_spheres_baseline.py',Path(__file__)]
    report=dict(task='isolated-spheres',revision=1,noise_trials=256,controls=controls,
        noise=dict(oracle_passes=256,shortcut_rejections=256,all_calibration_pass=True,max_calibration_chi2=max(x['calibration_chi2'] for x in trials),max_parameter_relative_error=max(x['parameter_relative_error'] for x in trials)),noise_extrema=summary,
        physical_checks=dict(calibration_equivalence=0.,independent_cases=len(cases),isolated_force_reference_error=reference_error,grounded_force_reference_error=grounded_error,multipole36_to48_error=refinement,bispherical128_to192_error=series_refinement,free_energy_displacement_error=work_error,geometric_scaling_error=scaling,minimum_full_gaussian_block_eigenvalue=minimum_block_eigenvalue,source_nonzero_charge_variance_diagnostic=max_charge_variance,force_abs_range=[min_force,max_force],far_separation_checks=far,maximum_noiseless_radius_recovery_error=max(recovery.values()),calibration_information=information,minimum_allowed_surface_gap=3.-2*1.15,scored_force_abs_range=[float(min(abs(signal))),float(max(abs(signal)))]),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},seconds=time.perf_counter()-start)
    out=ROOT/'results/isolated-spheres-r1-validation.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
