"""Validate finite-prestrain surface waves under an ideal pressure actuator."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1];TASK=ROOT/'tasks/pressure-surface-waves'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
oracle=load('surface_oracle',TASK/'solution/model.py');source=load('surface_source',ROOT/'scripts/pressure_surface_waves_baseline.py');ref=load('surface_reference',TASK/'tests/reference.py');meta=json.loads((TASK/'tests/metadata.json').read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args();start=time.perf_counter()
    cal=ref.calibration_inputs();truth=ref.predict(cal);sigma=meta['sigma']
    if args.generate:
        values=truth+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(cal))
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,values)]
        for rel in ['environment/data/calibration.json','tests/data/calibration.json']:(TASK/rel).write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text());assert len(records)==240 and all(r['sigma']==sigma and r['input']['pressure']==0 for r in records)
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    cal_equal=float(np.max(abs(oracle.predict_at(cal,ref.TRUE_PARAMETER)-source.predict_at(cal,ref.TRUE_PARAMETER))))
    cal_reference=float(np.max(abs(oracle.predict_at(cal,ref.TRUE_PARAMETER)-truth)));assert cal_equal==0 and cal_reference<.0001
    groups=ref.hidden_inputs();targets={k:ref.predict(v) for k,v in groups.items()}
    controls={}
    for label,module in [('oracle',oracle),('shortcut',source)]:
        m=module.Model().fit(records);residual=(m.predict(cal)-np.array([r['value'] for r in records]))/sigma
        hidden={k:float(np.linalg.norm(m.predict(v)-targets[k])/np.linalg.norm(targets[k])) for k,v in groups.items()}
        controls[label]=dict(density=m.density,parameter_relative_error=abs(m.density/ref.TRUE_PARAMETER-1),calibration_chi2=float(residual@residual/(len(cal)-1)),hidden=hidden)
        assert controls[label]['parameter_relative_error']<.03 and controls[label]['calibration_chi2']<1.5
        assert all((x<.04)==(label=='oracle' or k=='unloaded_anchors') for k,x in hidden.items())
    # Exact density scaling allows the precomputed spatial solutions to be reused for all fits.
    basis={label:{k:module.predict_at(v,1000.) for k,v in groups.items()} for label,module in [('oracle',oracle),('shortcut',source)]}
    cal_basis=oracle.predict_at(cal,1000.)
    extrema={label:{k:[] for k in groups} for label in ['oracle','shortcut']};parameters=[];chis=[]
    rng=np.random.default_rng(meta['noise_seed'])
    for trial in range(256):
        values=truth+rng.normal(0,sigma,len(cal));noisy=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,values)]
        density=oracle.Model().fit(noisy).density;density2=source.Model().fit(noisy).density;assert density==density2
        scale=np.sqrt(1000/density);residual=(cal_basis*scale-values)/sigma;chi=float(residual@residual/(len(cal)-1))
        assert abs(density/ref.TRUE_PARAMETER-1)<.03 and chi<1.5
        parameters.append(density);chis.append(chi)
        for label in extrema:
            for key in groups:
                error=float(np.linalg.norm(basis[label][key]*scale-targets[key])/np.linalg.norm(targets[key]));extrema[label][key].append(error)
                assert (error<.04)==(label=='oracle' or key=='unloaded_anchors')
    noise={label:{key:dict(min=min(es),max=max(es)) for key,es in by.items()} for label,by in extrema.items()}
    # Stable decaying root and static normal balance throughout the allowed domain.
    balance=0.;residual=0.;minimum_v2=1.;minimum_decay=1.;minimum_root=1.;maximum_root=0.;branch_count=[]
    for s,p in itertools.product(np.linspace(.95,1.2,11),np.linspace(0,.8,17)):
        e=ref.experiment(float(s),float(1e6*p));_,t,_=oracle.geometry(e);J=s*t;c=1-2*np.log(J)
        balance=max(balance,abs((t*t-c)/J+p))
        for label,module in [('oracle',oracle),('shortcut',source)]:
            v2=module.mode(e);x=(v2-s*s+t*t)/(t*t);chi=t*t/(t*t+2+c)
            matrix=module.surface_matrix(x,s,t,p) if label=='oracle' else module.surface_matrix(x,s,t)
            residual=max(residual,abs(np.linalg.det(matrix))/(t**4));minimum_v2=min(minimum_v2,v2);minimum_decay=min(minimum_decay,np.sqrt(1-x),np.sqrt(1-chi*x));minimum_root=min(minimum_root,x);maximum_root=max(maximum_root,x)
            assert v2>0 and v2<s*s and 0<x<1
            xs=np.linspace(1e-4,1-1e-6,101)
            vals=[]
            for q in xs:
                z=module.surface_matrix(q,s,t,p) if label=='oracle' else module.surface_matrix(q,s,t)
                vals.append(np.linalg.det(z)/q)
            count=int(np.sum(np.array(vals[:-1])*np.array(vals[1:])<0));branch_count.append(count);assert count==1
    assert balance<1e-10 and residual<1e-10 and minimum_v2>.4 and minimum_decay>.3
    # Direct scalar-energy FEM checks both boundary laws; refined physical reference at off-grid points.
    independent_error=source_reference_error=mesh_change=depth_change=0.;checks=[]
    for s,p in [(.95,0.),(1.2,0.),(.95,.8),(1.2,.8),(1.037,.637),(1.153,.724)]:
        e=ref.experiment(s,1e6*p)
        actual=oracle.predict_at([e],1100.)[0];r0=ref.predict([e],1100.)[0];r1=ref.predict([e],1100.,cells=512)[0]
        rd=ref.predict([e],1100.,cells=512,depth=36.)[0]
        independent_error=max(independent_error,abs(actual-r1));mesh_change=max(mesh_change,abs(r1-r0));depth_change=max(depth_change,abs(rd-r1))
        source_exact=source.predict_at([e],1100.)[0]
        coarse=ref.finite_element(s,p,follower=False,n=384,depth=28.)
        fine=ref.finite_element(s,p,follower=False,n=768,depth=28.)
        source_ref=(4*fine-coarse)/3*np.sqrt(1e6/1100.)
        source_reference_error=max(source_reference_error,abs(source_exact-source_ref))
        checks.append(dict(stretch=s,pressure=1e6*p,oracle=actual,reference=r1,source=source_exact,source_reference=source_ref))
    assert independent_error<.0001 and mesh_change<.0001 and depth_change<.0001 and source_reference_error<.0001
    # Increment of actual-area normal pressure, derived directly by finite displacement.
    rng=np.random.default_rng(710459);traction_error=0.;null_lagrangian_error=0.
    for _ in range(24):
        s=rng.uniform(.95,1.2);p=rng.uniform(.1,.8);t=ref.stretch(s,p);F=np.diag([s,1.,t]);J=np.linalg.det(F);N=np.array([0.,0.,-1.]);L=rng.normal(size=(3,3));h=1e-6
        def applied(A):return -p*np.linalg.det(A)*np.linalg.solve(A.T,N)
        numeric=(applied((np.eye(3)+h*L)@F)-applied((np.eye(3)-h*L)@F))/(2*h)
        analytic=-p*J*(np.trace(L)*np.eye(3)-L.T)@np.linalg.solve(F.T,N)
        traction_error=max(traction_error,float(np.max(abs(numeric-analytic))))
        a=rng.normal(size=3);n=rng.normal(size=3);R=np.outer(a,n);h=.0003
        second=p*(np.linalg.det((np.eye(3)+h*R)@F)-2*J+np.linalg.det((np.eye(3)-h*R)@F))/h**2
        null_lagrangian_error=max(null_lagrangian_error,abs(second))
    assert traction_error<1e-8 and null_lagrangian_error<1e-7
    classical=brentq(lambda x:x**3-8*x*x+20*x-12,.1,.99)
    classical_error=abs(oracle.mode(ref.experiment(1.,0.))-classical);assert classical_error<1e-12
    # Hydrostatic-pressure independence in the incompressible limit, outside the graded finite-lambda apparatus.
    incompressible=[]
    for p in [0.,.4,.8]:
        lam=1e7;t=brentq(lambda q:q*q-1+lam*np.log(q)+p*q,.9,1.1);c=1-lam*np.log(t);chi=t*t/(t*t+lam+c)
        x=brentq(lambda x:((2-x)**2-4*np.sqrt((1-x)*(1-chi*x)))/x,1e-5,1-1e-9)
        incompressible.append(np.sqrt(1-t*t+t*t*x))
    incompressible_span=float(np.ptp(incompressible));assert incompressible_span<1e-7
    recovery=[]
    for density in np.linspace(900,1300,41):
        exact=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,oracle.predict_at(cal,density))]
        recovery.append(abs(oracle.Model().fit(exact).density/density-1))
    assert max(recovery)<1e-14
    derivative=-truth/(2*ref.TRUE_PARAMETER);assert np.all(derivative<0)
    physical=dict(calibration_equivalence=cal_equal,calibration_reference_error_m_per_s=cal_reference,domain_cases=187,both_controls_unique_subsonic_branches=True,static_balance_error=balance,boundary_determinant_error=residual,minimum_dimensionless_squared_speed=minimum_v2,minimum_decay_constant=minimum_decay,root_range=[minimum_root,maximum_root],independent_speed_error_m_per_s=independent_error,source_independent_energy_error_m_per_s=source_reference_error,mesh_refinement_change_m_per_s=mesh_change,depth_refinement_change_m_per_s=depth_change,direct_pressure_traction_variation_error=traction_error,pressure_potential_bulk_null_lagrangian_error=null_lagrangian_error,unstressed_Rayleigh_cubic_error=classical_error,incompressible_hydrostatic_speed_span=incompressible_span,maximum_noiseless_density_recovery_error=max(recovery),density_information=float(derivative@derivative/sigma**2),reference_cases=checks)
    files=[TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'environment/data/calibration.json',TASK/'tests/metadata.json',ROOT/'scripts/pressure_surface_waves_baseline.py',Path(__file__)]
    report=dict(task='pressure-surface-waves',revision=1,noise_trials=256,controls=controls,noise=dict(oracle_passes=256,shortcut_rejections=256,all_calibration_pass=True,max_calibration_chi2=max(chis),max_parameter_relative_error=float(np.max(abs(np.array(parameters)/ref.TRUE_PARAMETER-1)))),noise_extrema=noise,physical_checks=physical,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},seconds=time.perf_counter()-start)
    (ROOT/'results/pressure-surface-waves-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
