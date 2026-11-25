"""Bounded independent Hamiltonian check of a changing-tilt protocol."""
import hashlib,json,time
from pathlib import Path
from itertools import product
import numpy as np
from scipy.optimize import brentq
import r1_geometry as old
import r1_trajectory as initial
import screen

HERE=Path(__file__).resolve().parent


def trajectory(j0,delta_final,slope,s_final,duration=512.,step=.04,
               action_order=24,phase_order=256,phase_offset=.5):
    start=time.perf_counter()
    delta_initial=screen.tilt(.05,delta_final,slope,s_final)
    qi,pi,weights,initial_energy,prep_error,closure=initial.initial_ensemble(
        j0,float(delta_initial),action_order,phase_order,phase_offset)
    q=qi.copy();p=pi.copy();work=np.zeros_like(q)
    count=int(np.ceil(duration/step));dt=duration/count;t=0.
    c=1/(2-2**(1/3));composition=(c,1-2*c,c)
    for _ in range(count):
        for coefficient in composition:
            h=dt*coefficient
            q+=.5*h*p;t+=.5*h
            r=float(np.clip(t/duration,0.,1.))
            s=.05+(s_final-.05)*r*r*(3-2*r)
            ds=(s_final-.05)*6*r*(1-r)/duration
            delta=delta_final+slope*np.log(s/s_final)
            work+=h*ds*(-q*q/2+(1.5*delta+slope)*np.sqrt(s)*q)
            p+=h*(-q*q*q+s*q-delta*s**1.5)
            q+=.5*h*p;t+=.5*h
    final=initial.hamiltonian(q,p,s_final,delta_final)
    saddle=np.sqrt(s_final)*old.geometry(delta_final)[0][1]
    result={'mean_energy':float(weights@final),'left_fraction':float(weights@(q<saddle)),
            'energy_work_max_abs':float(np.max(abs(final-initial_energy-work))),
            'initial_energy_max_abs':prep_error,'orbit_closure_max_abs':closure,
            'seconds':time.perf_counter()-start,'trajectory_count':len(q)}
    return result


def save_run(parameters,configuration,label):
    result=trajectory(**parameters,**configuration)
    theory=screen.predict(parameters['j0'],parameters['delta_final'],parameters['slope'],parameters['s_final'],32)
    z,w=old.quadrature(32)
    predicted_left=float(w@np.array([r.get('probabilities',[np.nan])[0] for r in theory['details']])/2)
    row={'label':label,'parameters':parameters,'configuration':configuration,'direct':result,
         'physical':theory['physical'],'instant_area':theory['instant_area'],'endpoint_area':theory['endpoint_area'],
         'direct_relative_error':abs(result['mean_energy']/theory['physical']-1),
         'source_relative_error':theory['instant_relative_gap'],
         'predicted_left_fraction':predicted_left,'capture_fraction_error':abs(result['left_fraction']-predicted_left)}
    with (HERE/'trajectory-runs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    return row


def main():
    start=time.perf_counter();runs=[]
    representative={'j0':.1437,'delta_final':.2,'slope':.10,'s_final':2.2}
    configs=[dict(duration=256.,step=.04,action_order=16,phase_order=128),
             dict(duration=512.,step=.04,action_order=32,phase_order=256),
             dict(duration=1024.,step=.04,action_order=48,phase_order=384),
             dict(duration=512.,step=.02,action_order=32,phase_order=256),
             dict(duration=512.,step=.04,action_order=32,phase_order=256,phase_offset=.17)]
    for config in configs:runs.append(save_run(representative,config,'representative_refinement'))
    for j0,delta_final,slope,s_final in [(.12,.15,.08,1.8),(.16,.25,.12,2.2),(.12,.25,.12,1.8),(.16,.15,.10,2.2),(.1437,.2,0.,2.2)]:
        runs.append(save_run(dict(j0=j0,delta_final=delta_final,slope=slope,s_final=s_final),
                             dict(duration=512.,step=.04,action_order=24,phase_order=256),'domain_or_constant_path'))
    # Same final Hamiltonian, distinct fully declared preparation paths.
    runs.append(save_run(dict(j0=.1437,delta_final=.2,slope=.12,s_final=2.2),
                         dict(duration=512.,step=.04,action_order=32,phase_order=256),'same_final_path_contrast'))
    growth_min=float('inf');derivative_error=0.;curvature_min=float('inf');initial_action_max=0.;final_action_min=float('inf')
    geometry_rows=[]
    for final,slope,endpoint in product(np.linspace(.15,.25,5),np.linspace(0,.12,7),[1.8,2.,2.2]):
        for s in np.geomspace(.05,endpoint,65):
            delta=screen.tilt(s,final,slope,endpoint)
            areas=old.geometry(float(delta))[2]
            analytic=screen.area_derivative(float(delta))
            eps=1e-5
            numerical=(old.geometry(float(delta+eps))[2]-old.geometry(float(delta-eps))[2])/(2*eps)
            derivative_error=max(derivative_error,float(np.max(abs(analytic-numerical))))
            growth_min=min(growth_min,float(screen.growth(s,final,slope,endpoint).min()))
            c=old.geometry(float(delta))[0][1]
            curvature_min=min(curvature_min,1-3*c*c)
        initial_action_max=max(initial_action_max,float(.05**1.5*old.geometry(float(screen.tilt(.05,final,slope,endpoint)))[2].sum()/(2*np.pi)))
        final_action_min=min(final_action_min,float(endpoint**1.5*old.geometry(float(final))[2].sum()/(2*np.pi)))
    gaps=[]
    for j0,final,slope,endpoint in product([.12,.1437,.16],[.15,.2,.25],[.06,.08,.10,.12],[1.8,2.2]):
        value=screen.predict(j0,final,slope,endpoint)
        gaps.append({'parameters':dict(j0=j0,delta_final=final,slope=slope,s_final=endpoint),
                     **{k:v for k,v in value.items() if k!='details'}})
    max_quad=0.
    for params in [representative,{'j0':.12,'delta_final':.15,'slope':.12,'s_final':1.8},
                   {'j0':.16,'delta_final':.25,'slope':.12,'s_final':2.2}]:
        small=screen.predict(params['j0'],params['delta_final'],params['slope'],params['s_final'],24)
        large=screen.predict(params['j0'],params['delta_final'],params['slope'],params['s_final'],48)
        max_quad=max(max_quad,abs(small['physical']-large['physical']),abs(small['instant_area']-large['instant_area']))
    fits=[]
    for truth in np.linspace(.12,.16,17):
        target=screen.predict(truth,.2,0.,1.8)['physical']
        lower=screen.predict(.12,.2,0.,1.8)['instant_area']
        upper=screen.predict(.16,.2,0.,1.8)['instant_area']
        if target<=lower: recovered=.12
        elif target>=upper: recovered=.16
        else: recovered=brentq(lambda j:screen.predict(j,.2,0.,1.8)['instant_area']-target,.12,.16,xtol=1e-12)
        fits.append({'true':float(truth),'fit':float(recovered)})
    right_bound=1.5*old.geometry(.25)[2][1]+.12*screen.area_derivative(.25)[1]
    report={'status':'bounded_feasibility_complete_unevaluated','seconds':time.perf_counter()-start,
            'trajectory_runs':runs,'static_domain_path_count':105,'points_per_path':65,
            'minimum_sampled_lobe_growth':growth_min,'minimum_sampled_saddle_curvature':curvature_min,
            'area_derivative_max_centered_difference_error':derivative_error,
            'maximum_initial_separatrix_action':initial_action_max,'minimum_final_separatrix_action':final_action_min,
            'right_growth_bracket_global_lower_bound':float(right_bound),
            'sampled_gap_cases':len(gaps),'minimum_gap':min(r['instant_relative_gap'] for r in gaps),
            'minimum_absolute_physical_energy':min(abs(r['physical']) for r in gaps),
            'max_action_quadrature_change':max_quad,'global_noiseless_fits':fits,
            'max_fit_error':max(abs(r['true']-r['fit']) for r in fits),
            'gap_rows':gaps,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'model_evaluations':0}
    (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['trajectory_runs','gap_rows','global_noiseless_fits']},indent=2))

if __name__=='__main__':main()
