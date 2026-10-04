"""Scientific controls for waves in a homogeneously prestrained solid."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/prestrained-solid'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


oracle = load('oracle', TASK/'solution/model.py')
shortcut = load('shortcut', ROOT/'scripts/prestrained_solid_baseline.py')
ref = load('reference', TASK/'tests/reference.py')
META = json.loads((TASK/'tests/metadata.json').read_text())


def stress(f):
    c = f.T@f; volume = np.linalg.det(f)
    return .6*f + .4*(np.trace(c)*f - f@c) + (2*np.log(volume)-1.4)*np.linalg.inv(f).T


def derivative(f, h):
    c = f.T@f; g = np.linalg.inv(f).T
    dc = h.T@f + f.T@h
    return (.6*h + .4*(np.trace(dc)*f + np.trace(c)*h - h@c - f@dc)
            + 2*np.trace(np.linalg.solve(f,h))*g
            - (2*np.log(np.linalg.det(f))-1.4)*g@h.T@g)


def scalar_closure(f, n):
    b=f@f.T; a=1.4-2*np.log(np.linalg.det(f)); w=b@n
    return a*np.eye(3)+(2+a)*np.outer(n,n)+.4*(np.outer(w,w)-(n@b@n)*b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--generate', action='store_true');args=p.parse_args();start=time.perf_counter()
    cal=ref.calibration_inputs();sigma=META['sigma'];truth=ref.predict(cal)
    if args.generate:
        rng=np.random.default_rng(META['calibration_seed'])
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,truth+rng.normal(0,sigma,len(cal)))]
        for r in ['environment/data/calibration.json','tests/data/calibration.json']:(TASK/r).write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    assert len(records)==216 and all(r['sigma']==sigma for r in records)
    unit=oracle.predict_at(cal,1.);base=shortcut.predict_at(cal,1.)
    cal_equal=float(np.max(abs(unit-base)));cal_ref=float(np.max(abs(unit-ref.predict(cal,1.))))
    assert cal_equal<1e-10 and cal_ref<2e-6
    groups=ref.hidden_inputs();target={k:ref.predict(v) for k,v in groups.items()};scale={k:np.linalg.norm(y) for k,y in target.items()}
    controls={}
    for label,module in [('oracle',oracle),('shortcut',shortcut)]:
        m=module.Model().fit(records);res=(m.predict(cal)-np.array([r['value'] for r in records]))/sigma
        errors={k:float(np.linalg.norm(m.predict(v)-target[k])/scale[k]) for k,v in groups.items()}
        controls[label]={'modulus':m.modulus,'parameter_relative_error_max':abs(m.modulus/ref.TRUE_PARAMETER-1),
                         'calibration_chi2':float(res@res/(len(cal)-1)),'hidden':errors}
        assert controls[label]['parameter_relative_error_max']<.03 and controls[label]['calibration_chi2']<1.5
        assert all((x<META['prediction_limit'])==(label=='oracle') for x in errors.values())
    rng=np.random.default_rng(META['noise_seed']);draws=truth[None,:]+rng.normal(0,sigma,(256,len(cal)))
    amplitude=draws@unit/(unit@unit);parameters=amplitude**2
    residual=(draws-amplitude[:,None]*unit[None,:])/sigma
    chi2=np.sum(residual**2,axis=1)/(len(cal)-1);perror=abs(parameters/ref.TRUE_PARAMETER-1)
    assert np.max(chi2)<1.5 and np.max(perror)<.03 and np.all((parameters>.8)&(parameters<1.6))
    extrema={}
    for label,module in [('oracle',oracle),('shortcut',shortcut)]:
        extrema[label]={}
        for k,v in groups.items():
            output=module.predict_at(v,1.)
            error=np.linalg.norm(amplitude[:,None]*output-target[k],axis=1)/scale[k]
            extrema[label][k]={'min':float(error.min()),'max':float(error.max())}
            assert np.all((error<META['prediction_limit'])==(label=='oracle'))
    # Broad admissible strains, orientation covariance and independent energy Hessians.
    cases=[]
    for sv in itertools.product(np.linspace(.85,1.45,7),repeat=3):
        if .95<=np.prod(sv)<=1.18:
            for direction in [[1.,0.,0.],[1.,-2.,.4]]:
                cases.append(ref.experiment(sv,direction,0,.37))
    for _ in range(100):
        while True:
            sv=rng.uniform(.85,1.45,3)
            if .95<=np.prod(sv)<=1.18:break
        e=ref.experiment(sv,rng.normal(size=3),0,rng.uniform(-3,3),rng.normal(size=3))
        # Isotropic material symmetry also permits arbitrary reference rotations.
        e['deformation']=(np.array(e['deformation'])@ref.rotation(rng.normal(size=3),rng.uniform(-3,3))).tolist()
        cases.append(e)
    cases.extend(e for values in groups.values() for e in values)
    min_exact=1e9;min_short=1e9;max_ref=0.;max_refine=0.;max_objective=0.;max_reverse=0.;max_source=0.;max_identity=0.;rotation_work=0.;rotation_energy=0.;max_speed_ref=0.
    for e in cases:
        f,n=oracle.geometry(e);volume=np.linalg.det(f)
        q,density=oracle.operator(e);b,_=shortcut.operator(e)
        # Operators are normalized by the common fitted modulus in Pa.
        min_exact=min(min_exact,float(np.linalg.eigvalsh(q)[0]));min_short=min(min_short,float(np.linalg.eigvalsh(b)[0]))
        energy=ref.energy_operator(e);fine=ref.energy_operator(e,.0006)
        max_ref=max(max_ref,float(np.max(abs(q*volume-energy))))
        max_refine=max(max_refine,float(np.max(abs(energy-fine))))
        max_speed_ref=max(max_speed_ref,float(np.max(abs(np.sqrt(1000*ref.TRUE_PARAMETER*np.linalg.eigvalsh(energy))-np.sqrt(1e6*ref.TRUE_PARAMETER*np.linalg.eigvalsh(q)/density)))))
        max_source=max(max_source,float(np.max(abs(b*volume-scalar_closure(f,n)))))
        sigma_tensor=stress(f)@f.T/volume
        max_identity=max(max_identity,float(np.max(abs(q-b-(n@sigma_tensor@n)*np.eye(3)))))
        r=ref.rotation([.3,-.7,1.],.81);er=dict(e,deformation=(r@f).tolist(),direction=(r@n).tolist())
        for module in [oracle,shortcut]:
            before=module.operator(e)[0];after=module.operator(er)[0]
            max_objective=max(max_objective,float(np.max(abs(after-r@before@r.T))))
            back=module.operator(dict(e,direction=(-3*n).tolist()))[0]
            max_reverse=max(max_reverse,float(np.max(abs(back-before))))
        omega=np.array([[0.,-.4,.7],[.4,0.,-.2],[-.7,.2,0.]])
        h=omega@f
        rotation_work=max(rotation_work,abs(float(np.sum(h*derivative(f,h))+np.sum(stress(f)*(omega@omega@f)))))
        rotation_energy=max(rotation_energy,abs(float(ref.energy(r@f)-ref.energy(f))))
    assert min_short>0 and min_exact>0 and max_ref<2e-7 and max_refine<4e-7 and max_speed_ref<3e-6
    assert max_source<1e-12 and max_identity<1e-12 and max_objective<1e-12 and max_reverse<1e-12
    assert rotation_work<1e-12 and rotation_energy<1e-12
    # Uniform lower bound for the shortcut over the stated singular-value/determinant domain.
    lower_bound=(1.4-2*np.log(1.18)-.4*1.18**2/.85**2)/1.18
    assert lower_bound>0
    anchors=[]
    for n in [[1.,0,0],[.3,-.7,.9]]:
        n=np.array(n);n=n/np.linalg.norm(n);e=dict(deformation=np.eye(3).tolist(),direction=n.tolist(),branch=0)
        expected=np.eye(3)+3.8*np.outer(n,n)
        anchors.append(float(np.max(abs(oracle.operator(e)[0]-expected))))
    assert max(anchors)<1e-12
    fit_endpoints={}
    for parameter in [.8,1.17,1.6]:
        noiseless=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,oracle.predict_at(cal,parameter))]
        fit=oracle.Model().fit(noiseless).modulus
        fit_endpoints[str(parameter)]=abs(fit-parameter);assert abs(fit-parameter)<1e-12
    # Explicit check that calibration's loaded axis carries stress, while wave normals do not.
    transverse_stress=[];loaded_stress=[]
    for e in cal:
        f,n=oracle.geometry(e);s=stress(f)@f.T/np.linalg.det(f)
        transverse_stress.append(abs(float(n@s@n)));loaded_stress.append(float(np.max(np.linalg.eigvalsh(s))))
    assert max(transverse_stress)<1e-13 and max(loaded_stress)>.8
    report={'task':'prestrained-solid','revision':1,'noise_trials':256,'calibration_seed':META['calibration_seed'],'noise_seed':META['noise_seed'],
        'controls':controls,'noise':{'oracle_passes':256,'shortcut_rejections':256,'all_calibration_pass':True,'max_calibration_chi2':float(chi2.max()),'max_parameter_relative_error':float(perror.max())},'noise_extrema':extrema,
        'physical_checks':{'calibration_equality_m_per_s':cal_equal,'calibration_energy_reference_error_m_per_s':cal_ref,'public_cases':len(cases),
        'minimum_exact_operator_eigenvalue':min_exact,'minimum_shortcut_operator_eigenvalue':min_short,'whole_domain_shortcut_lower_bound':float(lower_bound),
        'independent_energy_hessian_error':max_ref,'reference_step_refinement':max_refine,'independent_speed_error_m_per_s':max_speed_ref,
        'independent_shortcut_closed_matrix_error':max_source,'geometric_stress_identity_error':max_identity,'spatial_frame_covariance_error':max_objective,'direction_reversal_scale_error':max_reverse,
        'rigid_rotation_energy_error':rotation_energy,'rigid_rotation_second_variation_balance':rotation_work,'unstressed_lame_limit_error':max(anchors),
        'calibration_max_transverse_stress_per_modulus':max(transverse_stress),'calibration_max_loaded_stress_per_modulus':max(loaded_stress),
        'parameter_endpoint_recovery':fit_endpoints,'sqrt_parameter_information':float(unit@unit/sigma**2),
        'minimum_hidden_speed_m_per_s':float(min(v.min() for v in target.values())),'maximum_hidden_speed_m_per_s':float(max(v.max() for v in target.values()))},
        'seconds':time.perf_counter()-start,'source_sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'environment/data/calibration.json',TASK/'tests/metadata.json',ROOT/'scripts/prestrained_solid_baseline.py',Path(__file__)]}}
    output=ROOT/'results/prestrained-solid-r1-validation.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
