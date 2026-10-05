"""Bounded direct-dynamics verification; not a task/noise/model evaluation."""
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import prototype as p
HERE=Path(__file__).resolve().parent
LOG=HERE/'validation-runs.jsonl'

def record(kind,row):
    entry={'kind':kind,**row}
    with LOG.open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
    print(json.dumps({'kind':kind,'parameters':row.get('parameters'),'relative_energy_disagreement':row.get('relative_energy_disagreement'),'seconds':row.get('seconds')},allow_nan=False),flush=True)
    return entry

def run(kind,**kwargs):
    row=p.trajectory(**kwargs)
    z=row['parameters'];target=p.predictions(z['j0'],z['delta'],z['s_final'],action_order=64)
    row.update(area_prediction=target,relative_energy_disagreement=abs(row['mean_energy']/target['physical']-1),capture_probability_error=abs(row['right_lobe_fraction']-target['right_capture_probability']*target['capture_fraction']))
    return record(kind,row)

def independent_integrator(rtol):
    j0=.14;delta=.2;si=.05;sf=2.;duration=128.
    q,p0,weights,initial,_,_=p.initial_ensemble(j0,delta,si,8,64,.5)
    n=len(q);t0=time.perf_counter()
    def rhs(t,y):
        x,v=y[:n],y[n:2*n]
        r,dr=p.ramp(t/duration,'cubic');s=si+(sf-si)*r;ds=(sf-si)*dr/duration
        return np.concatenate((v,-x*x*x+s*x-delta*s**1.5,ds*(-x*x/2+1.5*delta*np.sqrt(s)*x)))
    z=solve_ivp(rhs,[0.,duration],np.concatenate((q,p0,np.zeros(n))),method='DOP853',rtol=rtol,atol=rtol*.05)
    assert z.success
    x,v,work=z.y[:n,-1],z.y[n:2*n,-1],z.y[2*n:,-1]
    e=v*v/2+p.potential(x,sf,delta)
    return record('independent_DOP853',{'rtol':rtol,'mean_energy':float(weights@e),'max_energy_work_error':float(np.max(abs(e-initial-work))),'nfev':z.nfev,'seconds':time.perf_counter()-t0})

def main():
    start=time.perf_counter()
    representative=[]
    for duration in [256.,512.,1024.,2048.]:
        representative.append(run('representative_duration',duration=duration,step=.04,action_order=32,phase_order=256))
    for kwargs in [dict(action_order=64,phase_order=256),dict(action_order=32,phase_order=512),dict(action_order=64,phase_order=512),dict(phase_offset=.173),dict(shape='quintic'),dict(step=.02)]:
        config=dict(duration=1024.,step=.04,action_order=32,phase_order=256);config.update(kwargs)
        representative.append(run('representative_refinement',**config))
    dop=[independent_integrator(x) for x in [1e-9,1e-11]]
    sym=run('same_grid_symplectic',duration=128.,step=.04,action_order=8,phase_order=64)
    for r in dop:r['relative_to_symplectic']=abs(r['mean_energy']/sym['mean_energy']-1)
    assert abs(dop[0]['mean_energy']/dop[1]['mean_energy']-1)<1e-6
    assert max(x['relative_to_symplectic'] for x in dop)<1e-5
    assert max(x['relative_energy_disagreement'] for x in representative)<.01
    domain=[]
    for j0,delta,sf in itertools.product([.12,.16],[.15,.25],[1.8,2.2]):
        domain.append(run('hidden_domain_corner',j0=j0,delta=delta,s_final=sf,duration=1024.,step=.04,action_order=32,phase_order=256))
    for j0,delta,sf in [(.131,.18,1.93),(.153,.23,2.08)]:
        domain.append(run('hidden_domain_interior',j0=j0,delta=delta,s_final=sf,duration=1024.,step=.04,action_order=32,phase_order=256))
    worst=max(domain,key=lambda x:x['relative_energy_disagreement'])
    z=worst['parameters'];refined=run('worst_corner_refinement',j0=z['j0'],delta=z['delta'],s_final=z['s_final'],duration=2048.,step=.02,action_order=64,phase_order=256)
    assert max(x['relative_energy_disagreement'] for x in domain)<.01 and refined['relative_energy_disagreement']<.01
    limits=[]
    for j0,delta,sf in [(.12,.15,.1),(.16,.25,.2),(.14,0.,2.),(.14,.2,.45)]:
        limits.append(run('limit',j0=j0,delta=delta,s_final=sf,duration=1024.,step=.04,action_order=32,phase_order=256))
    # The .45 endpoint remains just outside capture for this entire band.
    # True partial capture is separately tested in partial_capture.py.
    partial=[p.predictions(.14,.2,.45,action_order=n) for n in [32,64,128]]
    assert abs(partial[-1]['physical']/partial[-2]['physical']-1)<.001
    assert max(x['relative_energy_disagreement'] for x in limits)<.01
    static=[]
    for j0,delta,sf in itertools.product([.12,.14,.16],[.15,.2,.25],[1.8,2.,2.2]):
        a=p.predictions(j0,delta,sf,action_order=32);b=p.predictions(j0,delta,sf,action_order=64)
        static.append({'j0':j0,'delta':delta,'s_final':sf,**a,'action_quadrature_relative_change':abs(a['physical']/b['physical']-1)})
    assert min(x['relative_gap'] for x in static)>.04
    calibration=[]
    for delta,sf in itertools.product([.15,.2,.25],[.1,.15,.2]):
        energies=[]
        for j0 in np.linspace(.12,.16,17):
            a=p.predictions(j0,delta,sf)
            assert a['physical']==a['source'] and a['capture_fraction']==0
            energies.append(a['physical'])
        slopes=np.diff(energies)/(.04/16)
        assert min(slopes)>0
        calibration.append({'delta':delta,'s_final':sf,'minimum_sampled_energy_derivative':float(min(slopes)),'energy_values':energies})
    recovery=[]
    for j0 in np.linspace(.12,.16,17):
        truth=p.predictions(j0,.2,.15)['physical']
        fit=brentq(lambda j:p.predictions(j,.2,.15)['source']-truth,.12,.16,xtol=1e-11)
        recovery.append({'true':float(j0),'fitted':fit,'absolute_error':abs(fit-j0)})
    report={'status':'bounded_hamiltonian_reference_feasibility_passed_no_task_or_model_run',
      'representative':representative,'independent_integrator':dop,'same_grid_symplectic':sym,
      'domain':domain,'worst_corner_refinement':refined,'limits':limits,'near_threshold_noncrossing_action_quadrature':partial,
      'static_grid':static,'calibration':calibration,'fit_recovery':recovery,
      'summary':{'representative_reference_error_max':max(x['relative_energy_disagreement'] for x in representative),
        'domain_reference_error_max':max(x['relative_energy_disagreement'] for x in domain),
        'refined_worst_domain_error':refined['relative_energy_disagreement'],
        'limit_reference_error_max':max(x['relative_energy_disagreement'] for x in limits),
        'minimum_static_energy_gap':min(x['relative_gap'] for x in static),
        'minimum_absolute_hidden_energy':min(abs(x['physical']) for x in static),
        'maximum_trajectory_energy_work_error':max(x['energy_work_balance_max_abs'] for x in representative+domain+limits),
        'max_action_area_refinement':max(x['action_quadrature_relative_change'] for x in static),
        'max_fit_absolute_error':max(x['absolute_error'] for x in recovery),
        'runtime_seconds':time.perf_counter()-start},
      'qualifications':['Convergence is empirical at a scale well below the ordinary .04 prediction gate, not a theorem or high-precision asymptotic extrapolation.','At fixed phase/action resolution, error versus duration need not be monotone. All exploratory runs are preserved.','The direct Hamiltonian reference uses no capture probability or final action formula; initial action inversion only specifies the preparation.','Negative mean energy uses the fixed zero of the given potential; no offset or cancellation is introduced.','No calibration dataset, 256-noise study, grader or model trial is present.'],
      'source_sha256':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['prototype.py','validate.py','exploration.jsonl','validation-runs.jsonl']}}
    (HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('SUMMARY',json.dumps(report['summary'],indent=2),flush=True)
if __name__=='__main__':main()
