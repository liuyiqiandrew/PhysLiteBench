"""Validate finite-stiffness planar ring equilibrium and the rigid-metric control."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/elastic-ring'


def module(path):
    spec=importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def finite_stiffness(experiment, stiffness=.9, spring_over_temperature=1e7, radial_order=8, angular_order=128):
    nodes, weights = hermgauss(radial_order)
    indices = np.array(np.meshgrid(*([np.arange(radial_order)]*4), indexing='ij')).reshape(4,-1).T
    lengths = 1+np.sqrt(2/spring_over_temperature)*nodes[indices]
    radial_weights = np.prod(weights[indices], axis=1)/np.pi**2
    l1,l2,l3,l4 = lengths.T
    angular_nodes, angular_weights = leggauss(angular_order)
    cutoff = experiment['cutoff']
    angles = cutoff+(angular_nodes+1)*(np.pi-2*cutoff)/2
    integral=[]
    rejected=[]
    for angle in angles:
        r1 = np.column_stack([l1, np.zeros_like(l1)])
        r3 = l4[:,None]*np.array([np.cos(angle), np.sin(angle)])
        diagonal = r3-r1
        distance = np.linalg.norm(diagonal,axis=1)
        e = diagonal/distance[:,None]
        x=(l2*l2-l3*l3+distance*distance)/(2*distance)
        height2=l2*l2-x*x
        height=np.sqrt(np.maximum(height2,1e-200))
        perpendicular=np.column_stack([-e[:,1],e[:,0]])
        r2=r1+x[:,None]*e-height[:,None]*perpendicular
        polygon=np.stack([np.zeros_like(r1),r1,r2,r3],axis=1)
        edges=np.roll(polygon,-1,axis=1)-polygon
        next_edges=np.roll(edges,-1,axis=1)
        cross=edges[:,:,0]*next_edges[:,:,1]-edges[:,:,1]*next_edges[:,:,0]
        valid=(np.min(lengths,axis=1)>0)&(height2>0)&np.all(cross>0,axis=1)
        jacobian=np.zeros(len(lengths))
        jacobian[valid]=np.prod(lengths[valid],axis=1)/(distance[valid]*height[valid])
        integral.append(radial_weights@jacobian)
        rejected.append(radial_weights@(~valid))
    potential=stiffness*(1-np.cos(angles-experiment['preferred']))
    weight=angular_weights*np.array(integral)*np.exp(-potential/experiment['temperature'])
    moment=np.sin(angles) if experiment['readout']=='sine' else np.cos(2*angles)
    return float(weight@moment/weight.sum()),float(max(rejected))

def physical_checks(good,bad,ref):
    experiments=sum(ref.hidden_inputs().values(),[])
    agreement=max(abs(good.predict_at(experiments,.9)-ref.predict(experiments,.9)))
    refined=max(abs(ref.predict(experiments,.9,160)-ref.predict(experiments,.9)))
    original=good.predict_at(experiments,.9)
    good.NODES,good.WEIGHTS=leggauss(192)
    refinement=max(abs(good.predict_at(experiments,.9)-original))
    good.NODES,good.WEIGHTS=leggauss(128)
    corners=[dict(readout=readout,temperature=t,cutoff=a,preferred=p) for readout in ['sine','cosine2']
             for t in [.5,2.] for a in [.2,.6] for p in [.5,np.pi-.5]]
    corner_error=max(max(abs(good.predict_at(corners,k)-ref.predict(corners,k))) for k in [.5,1.5])
    metric_error=max(abs(bad.shape_measure(a)-1) for a in np.linspace(.2,np.pi-.2,101))
    normal_error=max(abs(ref.transverse_weight(a)*4*np.sin(a)-1) for a in np.linspace(.2,np.pi-.2,101))
    high_temperature_errors=[]
    for a in [.2,.4,.6]:
        cases=[dict(readout=o,temperature=1.,cutoff=a,preferred=1.1) for o in ['sine','cosine2']]
        partition=2*np.log(1/np.tan(a/2))
        expected=np.array([(np.pi-2*a)/partition,1-4*np.cos(a)/partition])
        high_temperature_errors.append(float(max(abs(good.predict_at(cases,0.)-expected))))
    mirror_errors=[]
    for e in corners:
        mirror_errors.append(abs(good.predict_at([e],.9)[0]-good.predict_at([dict(e,preferred=np.pi-e['preferred'])],.9)[0]))
    cal=ref.calibration_inputs()
    cal_equivalence=max(abs(good.predict_at(cal,.9)-bad.predict_at(cal,.9)))
    # Differentiate the Cartesian coordinate map independently of the normal Hessian.
    def coordinates(q):
        tx,ty,orientation,angle,l1,l2,l3,l4=q
        origin=np.array([tx,ty]);r1=l1*np.array([np.cos(orientation),np.sin(orientation)])
        r3=l4*np.array([np.cos(orientation+angle),np.sin(orientation+angle)])
        delta=r3-r1;distance=np.linalg.norm(delta);axis=delta/distance
        x=(l2*l2-l3*l3+distance*distance)/(2*distance)
        height=np.sqrt(l2*l2-x*x)
        r2=r1+x*axis-height*np.array([-axis[1],axis[0]])
        return (origin+np.array([np.zeros(2),r1,r2,r3])).ravel(),l1*l2*l3*l4/(distance*height)
    jacobian_error=0.
    for angle in [.3,1.1,2.7]:
        q=np.array([.13,-.27,.4,angle,1.003,.995,1.006,.999])
        _,expected=coordinates(q)
        columns=[]
        for j in range(8):
            step=np.zeros(8);step[j]=1e-5
            columns.append((coordinates(q+step)[0]-coordinates(q-step)[0])/(2e-5))
        jacobian_error=max(jacobian_error,abs(abs(np.linalg.det(np.array(columns).T))/expected-1))
    finite_results=[]
    for cutoff,preferred,temp in [(.2,.5,2.),(.2,np.pi/2,1.5),(.6,2.64,.5)]:
        e=dict(readout='cosine2',cutoff=cutoff,preferred=preferred,temperature=temp)
        truth=ref.predict([e],.9)[0]
        values=[finite_stiffness(e,.9,s)[0] for s in [1e6,1e7,1e8]]
        refined_value,rejected=finite_stiffness(e,.9,1e8,10,160)
        errors=[abs(v-truth) for v in values]
        assert errors[2]<errors[1]<errors[0] and errors[2]<1e-5
        assert abs(values[-1]-refined_value)<1e-9 and rejected==0.
        finite_results.append(dict(input=e,spring_over_temperature=[1e6,1e7,1e8],errors=errors,
                                   radial8_vs10_angular128_vs160=abs(values[-1]-refined_value),
                                   quadrature_weight_outside_convex_sector=rejected))
    assert agreement<1e-12 and refined<1e-12 and refinement<1e-12 and corner_error<1e-12
    assert metric_error<1e-12 and normal_error<1e-12 and max(high_temperature_errors)<1e-12
    assert max(mirror_errors)<1e-12 and cal_equivalence==0. and jacobian_error<1e-7
    return {'oracle_normal_hessian_reference_max':float(agreement),'reference96_vs160_max':float(refined),
            'oracle128_vs192_max':float(refinement),'parameter_and_apparatus_corner_reference_max':float(corner_error),
            'rigid_tangent_metric_constant_error':float(metric_error),'normal_hessian_determinant_identity_error':float(normal_error),
            'zero_bending_analytic_moment_error':max(high_temperature_errors),'mirror_symmetry_error':float(max(mirror_errors)),
            'calibration_control_difference':float(cal_equivalence),'cartesian_coordinate_jacobian_relative_error':float(jacobian_error),
            'finite_stiffness_cartesian_quadrature':finite_results,
            'ensemble_note':'Cartesian finite-K canonical equilibrium is taken before K tends to infinity; COM removal and free orientation contribute shape-independent factors. Hard angular walls stay fixed.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256)
    args=parser.parse_args();start=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/elastic_ring_baseline.py');ref=module(TASK/'tests/reference.py')
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
    def scores(model):return {name:float(np.sqrt(np.mean((model.predict(es)-truth[name])**2))) for name,es in hidden.items()}
    report={'revision':1,'calibration_seed':metadata['calibration_seed'],'noise_seed':metadata['noise_seed'],'noise_trials':args.noise_trials,'controls':{}}
    for label,source in [('oracle',good),('shortcut',bad)]:
        model=source.Model().fit(records);result={'parameter':model.stiffness,'parameter_relative_error':abs(model.stiffness/true-1),
            'calibration_chi2':chi(model,records),'hidden':scores(model)}
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        assert (max(result['hidden'].values())<limit if label=='oracle' else min(result['hidden'].values())>limit)
        report['controls'][label]=result
    rng=np.random.default_rng(metadata['noise_seed']);parameters=[];chi2s=[]
    for _ in range(args.noise_trials):
        sample=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))]
        a=good.Model().fit(sample);b=bad.Model().fit(sample);assert abs(a.stiffness-b.stiffness)<1e-10
        parameters.append(a.stiffness);chi2s.append(chi(a,sample))
    assert max(chi2s)<1.5 and max(abs(np.array(parameters)/true-1))<.03
    report['noise']={'parameter_min':min(parameters),'parameter_max':max(parameters),'parameter_relative_error_max':float(max(abs(np.array(parameters)/true-1))),
        'calibration_chi2_max':max(chi2s),'calibration_pass_fraction':1.}
    extrema={}
    for label,source in [('oracle',good),('shortcut',bad)]:
        errors=[]
        for parameter in [min(parameters),max(parameters)]:
            model=source.Model();model.stiffness=parameter;errors.extend(scores(model).values())
        extrema[label]={'min':min(errors),'max':max(errors)}
    assert extrema['oracle']['max']<limit and extrema['shortcut']['min']>limit
    report['hidden_parameter_extrema_check']=extrema;report['physical_checks']=physical_checks(good,bad,ref);report['seconds']=time.time()-start
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/metadata.json',TASK/'environment/data/calibration.json',ROOT/'scripts/elastic_ring_baseline.py',Path(__file__)]}
    output=ROOT/'jobs/elastic-ring-validation';output.mkdir(parents=True,exist_ok=True);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'results/elastic-ring-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('elastic-ring PASS',output/'summary.json',report['seconds'])
    print(json.dumps(report['controls'],indent=2));print(json.dumps(report['physical_checks'],indent=2))


if __name__=='__main__':main()
