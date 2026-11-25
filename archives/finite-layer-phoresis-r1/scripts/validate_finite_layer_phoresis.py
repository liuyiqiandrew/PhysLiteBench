"""Scientific validation of finite-layer solute-driven motion; no model runs."""
from pathlib import Path
import argparse, hashlib, importlib.util, itertools, json, time
import numpy as np
from scipy.integrate import quad,solve_bvp

BASE=Path(__file__).resolve().parents[1]
TASK=BASE/'tasks/finite-layer-phoresis'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def normalized(actual,truth):
    return float(np.sqrt(np.mean((actual-truth)**2)/np.mean(truth**2)))


def force_balance(radius,width,strength,prototype):
    c,derivative=prototype.fields(radius,width,strength)
    end=radius+width
    def rhs(r,y):
        return np.array([y[1],y[2],y[3],-c(r)*derivative(r)+4*y[2]/r**2-8*y[1]/r**3+8*y[0]/r**4])
    def bc(left,right):
        return np.array([left[0],left[1],right[2]-2*right[0]/end**2,
                         right[3]-2*right[1]/end**2+4*right[0]/end**3])
    grid=np.linspace(radius,end,161)
    solution=solve_bvp(rhs,bc,grid,np.zeros((4,len(grid))),tol=1e-9,max_nodes=10000)
    assert solution.success
    y=solution.sol(radius)
    fluid=4*np.pi*radius**2/3*(-y[3]+2*y[2]/radius)
    reaction=4*np.pi/3*quad(lambda r:r*r*c(r)*derivative(r),radius,end,epsabs=1e-10,epsrel=1e-10)[0]
    return abs(fluid+reaction)/max(1.,abs(fluid),abs(reaction)),float(np.max(abs(bc(solution.sol(radius),solution.sol(end)))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    start=time.monotonic()
    oracle=load(TASK/'solution/model.py','oracle')
    shortcut=load(BASE/'scripts/finite_layer_phoresis_baseline.py','shortcut')
    reference=load(TASK/'tests/reference.py','reference')
    prototype=load(BASE/'prototypes/prototype.py','prototype')
    metadata=json.loads((TASK/'tests/metadata.json').read_text());sigma=metadata['sigma'];true=reference.TRUE_PARAMETER
    inputs=reference.calibration_inputs();noiseless=reference.predict(inputs)
    if args.generate:
        rng=np.random.default_rng(metadata['calibration_seed'])
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,noiseless+rng.normal(0,sigma,len(inputs)))]
        text=json.dumps(records,indent=2)+'\n'
        for p in [TASK/'environment/data/calibration.json',TASK/'tests/data/calibration.json']:p.write_text(text)
    records=json.loads((TASK/'tests/data/calibration.json').read_text())
    assert records==json.loads((TASK/'environment/data/calibration.json').read_text())
    assert all(set(r)=={'input','value','sigma'} and r['sigma']==sigma for r in records)
    hidden=reference.hidden_inputs();truth={key:reference.predict(es) for key,es in hidden.items()}
    nominal={}
    for name,module in [('oracle',oracle),('shortcut',shortcut)]:
        m=module.Model().fit(records);y=m.predict(inputs)
        nominal[name]={'viscosity':m.viscosity,'parameter_relative_error':abs(m.viscosity/true-1),
                       'calibration_chi2':float(np.sum(((y-np.array([r['value'] for r in records]))/sigma)**2)/(len(inputs)-1)),
                       'hidden':{key:normalized(m.predict(es),truth[key]) for key,es in hidden.items()}}
    assert max(nominal['oracle']['hidden'].values())<.04
    assert min(v for k,v in nominal['shortcut']['hidden'].items() if k!='wall_anchors')>.04
    rng=np.random.default_rng(metadata['noise_validation_seed']);noise=[]
    unit=oracle.predict_at(inputs,1.)
    hunit={name:{key:module.predict_at(es,1.) for key,es in hidden.items()} for name,module in [('oracle',oracle),('shortcut',shortcut)]}
    for _ in range(256):
        y=noiseless+rng.normal(0,sigma,len(inputs))
        inverse=np.sum(unit*y/sigma**2)/np.sum(unit**2/sigma**2);eta=1/inverse
        chi=float(np.sum(((unit/eta-y)/sigma)**2)/(len(inputs)-1))
        errors={name:{key:normalized(hunit[name][key]/eta,target) for key,target in truth.items()} for name in hunit}
        assert .8<eta<1.6 and abs(eta/true-1)<.03 and chi<1.5
        assert max(errors['oracle'].values())<.04
        assert min(v for k,v in errors['shortcut'].items() if k!='wall_anchors')>.04
        noise.append(dict(viscosity=eta,chi2=chi,errors=errors))
    corners=[reference.experiment(radius=a,width=w,strength=s) for a,w,s in itertools.product([.7,1.3],[.05,.5,1.7,3.5],[-2.,-.5,.5,2.])]
    corners+= [reference.experiment(radius=.91,width=2.37,strength=1.13),reference.experiment(radius=1.14,width=2.62,strength=-1.37)]
    ref=reference.predict(corners);actual=oracle.predict_at(corners,true)
    error=float(np.max(abs(actual-ref)))
    refined=reference.predict(corners,cells=384)
    refinement=float(np.max(abs(refined-ref)))
    assert error<2e-7 and refinement<2e-7
    prototype_errors=[];shortcut_errors=[];balance=[];boundary=[]
    for a,w,s in [(.7,3.5,2.),(1.3,3.5,-2.),(.9,.8,.8),(1.1,2.3,-1.2)]:
        result=prototype.velocities(a,w,s)
        e=reference.experiment(radius=a,width=w,strength=s)
        prototype_errors.append(abs(oracle.predict_at([e],1.)[0]-result['direct_stokes_velocity']))
        shortcut_errors.append(abs(shortcut.predict_at([e],1.)[0]-result['planar_kernel_velocity']))
        b,r=force_balance(a,w,s,prototype);balance.append(b);boundary.append(r)
    assert max(prototype_errors)<2e-8 and max(shortcut_errors)<2e-8
    assert max(balance)<2e-8 and max(boundary)<2e-8
    zero=[reference.experiment(kind=kind,width=.7,strength=0.) for kind in ['wall','sphere']]
    assert np.max(abs(oracle.predict_at(zero,true)))==0
    assert np.max(abs(shortcut.predict_at(zero,true)))==0
    assert np.max(abs(reference.predict(zero)))==0
    thin=prototype.velocities(1.,.003,1.2)
    assert abs(thin['particle_to_opposite_flat_mobility_ratio']-1)<.0003
    scaling=[]
    for module in [oracle,shortcut]:
        e=reference.experiment(radius=.7,width=1.4,strength=-1.1)
        larger=reference.experiment(radius=1.05,width=2.1,strength=-1.1)
        scaling.append(abs(module.predict_at([larger],1.)[0]/module.predict_at([e],1.)[0]-2.25))
    assert max(scaling)<1e-8
    exact_recovery=[]
    for eta in [.8,1.,1.2,1.6]:
        rr=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,oracle.predict_at(inputs,eta))]
        exact_recovery.append(abs(oracle.Model().fit(rr).viscosity/eta-1))
    assert max(exact_recovery)<1e-12
    report={'task':'finite-layer-phoresis','revision':1,'noise_trials':256,'controls':nominal,
      'noise':{'oracle_passes':256,'shortcut_rejections':256,'all_calibration_pass':True,'max_calibration_chi2':max(x['chi2'] for x in noise),'max_parameter_relative_error':max(abs(x['viscosity']/true-1) for x in noise),'max_oracle_hidden_error':max(v for x in noise for v in x['errors']['oracle'].values()),'min_shortcut_sphere_error':min(v for x in noise for k,v in x['errors']['shortcut'].items() if k!='wall_anchors')},
      'physical_checks':{'independent_reference_cases':len(corners),'oracle_reference_error_max':error,'reference_refinement_change_max':refinement,'independent_stokes_prototype_error_max':max(prototype_errors),'shortcut_planar_prototype_error_max':max(shortcut_errors),'fluid_traction_plus_direct_reaction_relative_error_max':max(balance),'no_slip_and_no_Stokeslet_boundary_residual_max':max(boundary),'geometric_scaling_error_max':max(scaling),'thin_layer':thin,'zero_interaction_exact':True,'noiseless_parameter_recovery_error_max':max(exact_recovery),'calibration_equivalence_max':float(np.max(abs(oracle.predict_at(inputs,1.)-shortcut.predict_at(inputs,1.)))),'minimum_hidden_sphere_signal':min(float(np.min(abs(v))) for k,v in truth.items() if k!='wall_anchors'),'maximum_hidden_sphere_signal':max(float(np.max(abs(v))) for k,v in truth.items() if k!='wall_anchors')},
      'data_integrity':{'fixed_instrument_sigma':sigma,'noiseless_sigma_side_channel':False,'public_private_identical':True,'seeds':{'calibration':metadata['calibration_seed'],'noise':metadata['noise_validation_seed']}},
      'runtime_seconds':time.monotonic()-start,'source_sha256':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(BASE.rglob('*')) if p.is_file() and p.suffix in ['.py','.md','.toml','.sh'] and '__pycache__' not in p.parts}}
    out=BASE/'results/finite-layer-phoresis-r1-validation.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
