"""Author prototype: no agent evaluations or task package."""
from pathlib import Path
import itertools
import json
import time
import numpy as np
from scipy.optimize import minimize_scalar


def embedded(size, right):
    P=np.zeros((size,size))
    if size>1:P[0,1]=1.
    for i in range(1,size):
        P[i,i-1]=1-right
        if i+1<size:P[i,i+1]=right
    return P


def visit_formula(size,right,contrast,slow_probability,scale):
    P=embedded(size,right)
    G=np.linalg.solve(np.eye(size)-P,np.eye(size))
    jumps=G@np.ones(size)
    jumps2=(2*G-np.eye(size))@jumps
    mu=(1-slow_probability+slow_probability*contrast)/scale
    mu2=(1-slow_probability+slow_probability*contrast**2)/scale**2
    repeats=2*np.dot(G[0],np.diag(G)-1)
    mean=mu*jumps[0]
    second_renewal=mu**2*jumps2[0]+(2*mu2-mu**2)*jumps[0]
    second_static=second_renewal+(mu2-mu**2)*repeats
    return {'mean':float(mean),'variance':float(second_static-mean**2),'renewal_variance':float(second_renewal-mean**2),'repeat_factor':float(repeats),'excess_variance':float((mu2-mu**2)*repeats)}


def renewal_generator(size,right,contrast,slow_probability,scale):
    """Exact continuous-time state model with a new rate class at every arrival."""
    P=embedded(size,right);pi=np.array([1-slow_probability,slow_probability]);rates=scale/np.array([1.,contrast])
    Q=np.zeros((2*size,2*size))
    for i in range(size):
        for c in range(2):
            row=2*i+c;Q[row,row]=-rates[c]
            for j in range(size):
                for d in range(2):Q[row,2*j+d]+=rates[c]*P[i,j]*pi[d]
    m1=np.linalg.solve(-Q,np.ones(2*size))
    m2=np.linalg.solve(-Q,2*m1)
    mean=pi@m1[:2];second=pi@m2[:2]
    return {'mean':float(mean),'variance':float(second-mean**2),'minimum_offdiagonal':float((Q-np.diag(np.diag(Q))).min()),'maximum_rowsum':float(Q.sum(axis=1).max())}


def specimen_enumeration(size,right,contrast,slow_probability,scale):
    """Independent exact average of first-passage moments in fixed landscapes."""
    mean=0.;second=0.;conditional_variance=0.;mean_square=0.;weight_sum=0.
    for state in itertools.product([0,1],repeat=size):
        state=np.asarray(state);slow=int(state.sum())
        weight=slow_probability**slow*(1-slow_probability)**(size-slow)
        residence=np.where(state==0,1.,contrast)/scale
        # Backward generator from explicit physical rates, not the visit Green matrix.
        A=np.diag(1/residence)
        if size>1:A[0,1]=-1/residence[0]
        for i in range(1,size):
            A[i,i-1]=-(1-right)/residence[i]
            if i+1<size:A[i,i+1]=-right/residence[i]
        first=np.linalg.solve(A,np.ones(size))
        raw_second=np.linalg.solve(A,2*first)
        mean+=weight*first[0];second+=weight*raw_second[0]
        conditional_variance+=weight*(raw_second[0]-first[0]**2)
        mean_square+=weight*first[0]**2;weight_sum+=weight
    return {'mean':float(mean),'variance':float(second-mean**2),'thermal_conditional_variance':float(conditional_variance),'specimen_mean_variance':float(mean_square-mean**2),'weights':float(weight_sum)}


def main():
    start=time.monotonic();cases=[]
    controls=[(n,p,c,s) for n,p in [(3,.5),(5,.55),(7,.65),(9,.75)] for c,s in [(4.,.5),(12.,.3),(25.,.15)]]
    for args in controls:
        d=visit_formula(*args,.95);ref=specimen_enumeration(*args,.95);src=renewal_generator(*args,.95)
        rel=max(abs(d[k]-ref[k])/ref[k] for k in ['mean','variance'])
        meanerr=abs(d['mean']-src['mean'])/d['mean']
        srcerr=abs(d['renewal_variance']-src['variance'])/src['variance']
        gap=(d['variance']-src['variance'])/d['variance']
        assert rel<1e-11 and meanerr<1e-12 and srcerr<1e-12
        assert d['variance']>0 and src['variance']>0 and gap>.04
        assert abs(ref['weights']-1)<1e-12
        assert abs(ref['variance']-ref['thermal_conditional_variance']-ref['specimen_mean_variance'])/ref['variance']<1e-12
        cases.append({'controls':args,'oracle':d,'reference':ref,'source':src,'relative_reference_error':rel,'relative_calibration_difference':meanerr,'source_formula_difference':srcerr,'relative_variance_gap':gap})
    limits=[]
    for n,p,c,s in itertools.product([2,5,8],[.45,.7,1.],[1.,8.],[0.,.35,1.]):
        a=visit_formula(n,p,c,s,1.1);b=specimen_enumeration(n,p,c,s,1.1);z=renewal_generator(n,p,c,s,1.1)
        for k in ['mean','variance']:assert abs(a[k]-b[k])/b[k]<1e-10
        assert a['variance']>0 and z['variance']>0
        if c==1 or p==1 or s in [0.,1.]:assert abs(a['variance']-z['variance'])/a['variance']<1e-11
        scaled=visit_formula(n,p,c,s,2.2)
        assert abs(scaled['mean']*2-a['mean'])<1e-8 and abs(scaled['variance']*4-a['variance'])<1e-7
        limits.append({'controls':[n,p,c,s],'relative_reference_error':max(abs(a[k]-b[k])/b[k] for k in ['mean','variance']),'source_equal_when_required':c==1 or p==1 or s in [0.,1.]})
    calibration=[(n,p,c,s) for n,p in [(3,.45),(5,.6),(7,.8)] for c,s in [(2.,.2),(6.,.5),(12.,.7)]]
    recovery=[]
    for scale in np.linspace(.6,1.4,33):
        y=np.array([visit_formula(*e,scale)['mean'] for e in calibration])
        # Fit the completed renewal generator, not the visit identity used by the oracle.
        pred=lambda q:np.array([renewal_generator(*e,q)['mean'] for e in calibration])
        result=minimize_scalar(lambda q:np.sum((pred(q)-y)**2),bounds=(.6,1.4),method='bounded',options={'xatol':1e-12})
        err=abs(result.x-scale);assert err<1e-6
        recovery.append({'true_scale':float(scale),'fitted_scale':float(result.x),'error':float(err)})
    result={'status':'prototype_checks_passed','model_evaluations':0,'cases':cases,'limit_checks':limits,'calibration_settings':calibration,'scale_recovery':recovery,'maximum_reference_error':max(c['relative_reference_error'] for c in cases),'maximum_source_formula_error':max(c['source_formula_difference'] for c in cases),'variance_gap_range':[min(c['relative_variance_gap'] for c in cases),max(c['relative_variance_gap'] for c in cases)],'maximum_mean_difference':max(c['relative_calibration_difference'] for c in cases),'max_scale_fit_error':max(c['error'] for c in recovery),'elapsed_seconds':time.monotonic()-start}
    p=Path(__file__).with_name('report.json');p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','limit_checks','calibration_settings','scale_recovery']},indent=2))

if __name__=='__main__':main()
