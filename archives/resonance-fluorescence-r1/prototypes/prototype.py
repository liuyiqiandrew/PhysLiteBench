"""Stationary fluorescence counts: exact record versus population-noise closure."""
from pathlib import Path
import itertools,json,time
import numpy as np
from scipy.integrate import solve_ivp,quad
from scipy.linalg import expm
from scipy.sparse import eye,kron,diags,csc_matrix
from scipy.sparse.linalg import expm_multiply

LOWER=np.array([[0.,1.],[0.,0.]])
NUMBER=LOWER.T@LOWER
TRACE=np.array([1.,0.,0.,1.])


def generator(rabi,detuning,decay):
    h=np.array([[0.,rabi/2],[rabi/2,detuning]])
    jump=decay*np.kron(LOWER,LOWER)
    liou=-1j*(np.kron(np.eye(2),h)-np.kron(h.T,np.eye(2)))
    liou+=jump-decay/2*(np.kron(np.eye(2),NUMBER)+np.kron(NUMBER.T,np.eye(2)))
    a=liou.copy();a[0]=TRACE
    rho=np.linalg.solve(a,np.array([1.,0.,0.,0.])).reshape((2,2),order='F')
    return liou,jump,rho


def integrated_covariance(liou,seed,observable,gate):
    block=np.zeros((6,6),complex);block[:4,:4]=liou
    block[4,:4]=observable;block[5,4]=1
    initial=np.r_[seed.reshape(4,order='F'),0.,0.]
    return float((expm(gate*block)@initial)[5].real)


def population_closure(rabi,detuning,decay,gate,efficiency):
    liou,jump,rho=generator(rabi,detuning,decay)
    p=float(np.trace(NUMBER@rho).real);mean=efficiency*decay*p*gate
    seed=(NUMBER@rho+rho@NUMBER)/2-p*rho
    integral=integrated_covariance(liou,seed,NUMBER.T.reshape(4,order='F'),gate)
    variance=mean+2*(efficiency*decay)**2*integral
    return mean,variance


def count_moments(rabi,detuning,decay,gate,efficiency):
    """Direct density-matrix factorial-moment ODE, no Liouville vectorization."""
    p=rabi*rabi/(decay*decay+2*rabi*rabi+4*detuning*detuning)
    c=1j*rabi*(2*p-1)/(2*(1j*detuning-decay/2))
    rho=np.array([[1-p,c],[c.conjugate(),p]],complex)
    h=np.array([[0.,rabi/2],[rabi/2,detuning]])
    def evolve(_time,state):
        rr=state.reshape((3,2,2));out=[]
        for k in range(3):
            x=rr[k]
            rhs=-1j*(h@x-x@h)+decay*(LOWER@x@LOWER.T-(NUMBER@x+x@NUMBER)/2)
            if k:rhs+=k*efficiency*decay*LOWER@rr[k-1]@LOWER.T
            out.append(rhs)
        return np.array(out).ravel()
    initial=np.zeros((3,2,2),complex);initial[0]=rho
    result=solve_ivp(evolve,[0,gate],initial.ravel(),method='DOP853',rtol=2e-11,atol=2e-13).y[:,-1].reshape(3,2,2)
    mean=float(np.trace(result[1]).real)
    variance=float(np.trace(result[2]).real)+mean-mean**2
    return mean,variance


def tilted_contour(rabi,detuning,decay,gate,efficiency,points=32,radius=.08):
    liou,jump,rho=generator(rabi,detuning,decay)
    theta=2*np.pi*np.arange(points)/points
    values=np.array([TRACE@expm(gate*(liou+np.expm1(radius*np.exp(1j*t))*efficiency*jump))@rho.reshape(4,order='F') for t in theta])
    mean=float((np.mean(values*np.exp(-1j*theta))/radius).real)
    second=float((2*np.mean(values*np.exp(-2j*theta))/radius**2).real)
    return mean,second-mean**2


def resolved_counts(rabi,detuning,decay,gate,efficiency,cutoff=64):
    liou,jump,rho=generator(rabi,detuning,decay)
    a=kron(eye(cutoff+1),csc_matrix(liou-efficiency*jump))+kron(diags(np.ones(cutoff),-1,shape=(cutoff+1,cutoff+1)),csc_matrix(efficiency*jump))
    initial=np.zeros(4*(cutoff+1),complex);initial[:4]=rho.reshape(4,order='F')
    final=expm_multiply(gate*a,initial).reshape(cutoff+1,4)
    prob=(final@TRACE).real;n=np.arange(cutoff+1)
    mean=np.dot(prob,n);variance=np.dot(prob,n*n)-mean**2
    return mean,variance,{'normalization_error':float(abs(prob.sum()-1)),'minimum_probability':float(prob.min()),'top_probability':float(prob[-1])}


if __name__=='__main__':
    start=time.perf_counter();cases=[]
    for rabi,detuning,decay,gate in itertools.product([.6,1.,1.8],[-.5,0.,.5],[.8,1.2],[1.,3.,6.]):
        args=(rabi,detuning,decay,gate,.73)
        mean,var=count_moments(*args);source_mean,source=population_closure(*args)
        check_mean,check=tilted_contour(*args)
        cases.append({'input':list(args),'mean':mean,'variance':var,'source_variance':source,
                      'relative_error':abs(source/var-1),'fano':var/mean,'source_fano':source/mean,
                      'mean_equivalence_error':abs(source_mean-mean),'independent_contour_error':max(abs(check_mean-mean),abs(check-var))})
    resolved=[]
    for args in [(1.,0.,1.,6.,.73),(.6,.5,.8,3.,.5),(1.8,-.5,1.2,6.,.9)]:
        oracle=count_moments(*args);m,v,checks=resolved_counts(*args)
        resolved.append({'input':list(args),'error':float(max(abs(np.array([m,v])-oracle))),**checks})
    # Resonant regression has a known closed form independently of the counting ODE.
    analytic=[]
    for rabi,gate in itertools.product([.6,1.,1.8],[1.,3.,6.]):
        decay=1.;eta=.73;p=rabi*rabi/(1+2*rabi*rabi);freq=np.sqrt(rabi*rabi-1/16)
        mean=eta*p*gate
        corr=lambda t:-p*p*np.exp(-3*t/4)*(np.cos(freq*t)+3/(4*freq)*np.sin(freq*t))
        variance=mean+2*eta**2*quad(lambda t:(gate-t)*corr(t),0,gate,epsabs=1e-12)[0]
        analytic.append(abs(variance-count_moments(rabi,0.,decay,gate,eta)[1]))
    report={'status':'prototype only; no task/model evaluation','case_count':len(cases),'cases':cases,'number_resolved':resolved,
            'maximum_contour_error':max(c['independent_contour_error'] for c in cases),
            'maximum_mean_equivalence_error':max(c['mean_equivalence_error'] for c in cases),
            'relative_gap_range':[min(c['relative_error'] for c in cases),max(c['relative_error'] for c in cases)],
            'variance_range':[min(c['variance'] for c in cases),max(c['variance'] for c in cases)],
            'source_excess_variance_min':min(c['source_variance']-c['mean'] for c in cases),
            'resonant_analytic_error_max':max(analytic),'elapsed_seconds':time.perf_counter()-start}
    assert report['maximum_contour_error']<1e-8
    assert max(c['error'] for c in resolved)<1e-8
    assert report['resonant_analytic_error_max']<1e-8
    assert report['source_excess_variance_min']>=0
    Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
