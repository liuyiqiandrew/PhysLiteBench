"""Bounded pre-packaging science only; no task modifications or agent runs."""
from pathlib import Path
import hashlib
import importlib.util
import itertools
import json
import time
import numpy as np
from recirculating import finite_volume, hermite, gaussian_closure, constant_clock_anchor

HERE=Path(__file__).resolve().parent
STAGE=HERE.parents[2]


def root_anchor():
    path=STAGE/'results/prototypes/rotating-reservoir-r8-root/exact_constant_collision.py'
    spec=importlib.util.spec_from_file_location('independent_root_monomials',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path


def setting(ta=1.6,tb=.4,lam=2.,omega=0.,eta=0.,flip=1.2):
    return dict(temperature_a=ta,temperature_b=tb,relaxation_rate=lam,
                angular_speed=omega,flux_fraction=eta,flip_rate=flip)


def proposed_groups():
    return dict(
        constant_contact_energy_noise=[setting(lam=1.8),setting(lam=2.2,omega=.4),
            setting(lam=2.6,omega=-.6,flip=.8),setting(lam=3.,omega=.65,flip=2.)],
        partial_flux_contact_noise=[setting(ta=1.8,lam=2.,omega=.4,eta=.4,flip=.8),
            setting(ta=1.8,lam=2.4,omega=-.5,eta=.6),
            setting(ta=2.,lam=2.7,omega=.6,eta=.8,flip=1.6),
            setting(ta=2.,lam=3.,omega=-.65,eta=.7,flip=2.)],
        full_flux_contact_noise=[setting(ta=1.8,lam=1.8,eta=1.),
            setting(ta=2.,lam=2.2,omega=.4,eta=1.,flip=.8),
            setting(ta=1.8,lam=2.6,omega=-.5,eta=1.,flip=1.5),
            setting(ta=2.,lam=3.,omega=.6,eta=1.,flip=2.)])


def main():
    start=time.perf_counter();independent,path=root_anchor();groups=proposed_groups()
    corners=[]
    for gamma,ta,tb,lam,omega,flip,eta in itertools.product([.4,1.1],[.4,2.],[.4,1.4],[.2,3.],[-.65,.65],[.8,2.],[0.,1.]):
        e=setting(ta,tb,lam,omega,eta,flip)
        approx=gaussian_closure(e,gamma,noise=True)
        truth=hermite(e,gamma,degree=26,variance=2.,noise=True)
        corners.append(dict(input=e,gamma=gamma,gaussian=approx,hermite26=truth))
    assert min(r['gaussian']['minimum_covariance'] for r in corners)>0.
    assert min(r['gaussian']['noise'] for r in corners)>0.
    assert min(r['gaussian']['projected_nonconstant_decay'] for r in corners)>0.
    assert max(r['gaussian']['first_law_residual'] for r in corners)<1e-8
    assert max(r['gaussian']['projected_stationary_residual'] for r in corners)<1e-7
    assert min(r['hermite26']['noise'] for r in corners)>0.
    controls=[]
    for gamma,ta,tb,lam,omega,flip in itertools.product([.4,.67,1.1],[.4,2.],[.4,1.4],[.2,3.],[0.,.65],[.8,2.]):
        e=setting(ta,tb,lam,omega,0.,flip)
        root=independent.exact(ta,tb,gamma,lam,omega,flip)
        own=hermite(e,gamma,degree=4,variance=2.,total_degree=True,noise=True)
        assert abs(root['mean']-own['mean'])<2e-12 and abs(root['noise']-own['noise'])<2e-11
        controls.append(dict(input=e,gamma=gamma,root_exact=root,independent_hermite4=own))
        if omega==0.:
            closed=constant_clock_anchor(e,gamma)
            assert abs(closed['noise']-root['noise'])<2e-12
            approx=gaussian_closure(e,gamma,noise=True)
            assert abs(closed['gaussian_noise']-approx['noise'])<2e-11
    rows={};runtime=time.perf_counter()
    for group,cases in groups.items():
        values=[]
        for e in cases:
            a=gaussian_closure(e,noise=True)
            h26=hermite(e,degree=26,variance=2.,noise=True)
            h34=hermite(e,degree=34,variance=2.,noise=True)
            h42=hermite(e,degree=42,variance=2.,noise=True)
            coarse=finite_volume(e,points=81,extent=7.,details=True,noise=True)
            fine=finite_volume(e,points=162,extent=7.,details=True,noise=True)
            richer={key:(4*fine[key]-coarse[key])/3 for key in ['noise','mean']}
            values.append(dict(input=e,source=a,hermite26=h26,hermite34=h34,hermite42=h42,
                fv81=coarse,fv162=fine,fv_richardson=richer))
        expected=np.array([r['hermite42']['noise'] for r in values]);source=np.array([r['source']['noise'] for r in values]);fv=np.array([r['fv_richardson']['noise'] for r in values])
        rows[group]=dict(cases=values,source_relative_error=float(np.linalg.norm(source-expected)/np.linalg.norm(expected)),
            fv_relative_error=float(np.linalg.norm(fv-expected)/np.linalg.norm(expected)),
            hermite34_to42_maximum_relative=float(max(abs(r['hermite34']['noise']-r['hermite42']['noise'])/r['hermite42']['noise'] for r in values)))
        print(group,rows[group]['source_relative_error'],rows[group]['fv_relative_error'],flush=True)
    assert min(r['source_relative_error'] for r in rows.values())>.08
    assert max(r['fv_relative_error'] for r in rows.values())<.002
    assert max(r['hermite34_to42_maximum_relative'] for r in rows.values())<.001
    cold_inputs=sum(groups.values(),[])+[setting(ta=1.,tb=1.,lam=.2),setting(ta=1.,tb=1.,lam=3.)]
    cold_start=time.perf_counter()
    cold=[]
    for e in cold_inputs:
        a=finite_volume(e,points=81,noise=True);b=finite_volume(e,points=162,noise=True)
        cold.append((4*b['noise']-a['noise'])/3)
    cold_seconds=time.perf_counter()-cold_start
    assert cold_seconds<50
    isolated=[]
    for ta,tb,eta,lam in itertools.product([.4,2.],[.4,1.4],[0.,1.],[.2,3.]):
        e=setting(ta,tb,lam,0.,eta)
        # This separate boundary is checked by the continuous reference:
        # gas heat is minus bounded total energy when there is no bath A.
        a=hermite(e,0.,degree=26,variance=2.,noise=True)
        assert abs(a['noise'])<2e-10
        isolated.append(dict(input=e,reference=a))
    equilibrium=[]
    for t,lam in itertools.product([.4,1.,1.4],[.2,1.,3.]):
        e=setting(t,t,lam);a=constant_clock_anchor(e)
        conductance=.67*lam*3.6/(2*.67*lam+3.6*(.67+lam))
        assert abs(a['noise']-2*conductance*t*t)<2e-12
        equilibrium.append(dict(input=e,noise=a['noise'],closed_FDT=2*conductance*t*t))
    report=dict(status='PASS',revision='rotating-reservoir-r8-proposal',seconds=time.perf_counter()-start,
        domain=dict(gamma=[.4,1.1],temperature_a=[.4,2.],temperature_b=[.4,1.4],relaxation_rate=[.2,3.],
            angular_speed=[-.65,.65],flip_rate=[.8,2.],flux_fraction=[0.,1.],collision_rate=3.6),
        proposed_groups=rows,source_domain_corners=corners,independent_root_constant_clock_controls=controls,
        equilibrium_FDT=equilibrium,isolated_bath_energy_boundary=isolated,
        grading_richardson_pair=[81,162],cold_grading_cases=len(cold_inputs),cold_grading_seconds=cold_seconds,
        proposed_gate=.04,
        unchanged_frozen_task='rotating-reservoir-r7; this script modifies no task files',
        limitations='This is scientific proposal screening only. No calibration regenerated, no task packaged and no model runs. Full 256 direct noisy-fit separation, independent tail/quadrature refinement and final source/cold Docker controls remain required after approval.',
        hashes={str(p.relative_to(STAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'recirculating.py',HERE/'README.md',Path(__file__),path]})
    target=STAGE/'results/rotating-reservoir-r8-finite-contact-proposal-audit.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',report['seconds'],'cold',cold_seconds,'SHA',hashlib.sha256(target.read_bytes()).hexdigest(),flush=True)


if __name__=='__main__':
    main()
