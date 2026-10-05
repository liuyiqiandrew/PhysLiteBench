"""Bounded stationary coherent-scattering feasibility; no task or model runs."""
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import minimize_scalar

N = 5
PINNING = .6
GAMMA = .4


def stiffness_matrix(coupling, n=N):
    derivative = np.eye(n-1, n, k=1)-np.eye(n-1, n)
    return PINNING*np.eye(n)+coupling*derivative.T@derivative


def modal_covariances(stiffness, temperature, coupling, delay, n=N):
    values, modes = np.linalg.eigh(stiffness_matrix(coupling, n))
    omega = np.sqrt(stiffness*values-GAMMA**2/4)
    variance = temperature/(stiffness*values)
    decay = np.exp(-GAMMA*delay/2)*(np.cos(omega*delay)
              +GAMMA*np.sin(omega*delay)/(2*omega))
    return (modes*variance)@modes.T, (modes*(variance*decay))@modes.T


def responses(stiffness, temperature, coupling, wavevector, delay, n=N):
    c0, ct = modal_covariances(stiffness, temperature, coupling, delay, n)
    diagonal = np.diag(c0)
    phase = np.exp(1j*wavevector*np.arange(n))
    weights = np.real(phase[:, None]*phase.conj()[None, :])/n
    pair_variance = diagonal[:, None]+diagonal[None, :]
    static = np.sum(weights*np.exp(-wavevector**2*(pair_variance-2*c0)/2))
    physical = np.sum(weights*np.exp(-wavevector**2*(pair_variance-2*ct)/2))
    plateau = abs(np.sum(phase*np.exp(-wavevector**2*diagonal/2)))**2/n
    self_value = np.mean(np.exp(-wavevector**2*(diagonal-np.diag(ct))))
    self_plateau = np.mean(np.exp(-wavevector**2*diagonal))
    source = plateau+(static-plateau)*(self_value-self_plateau)/(1-self_plateau)
    return {'physical':float(physical), 'source':float(source),
            'uncorrected_vineyard':float(static*self_value),
            'static':float(static), 'plateau':float(plateau),
            'self':float(self_value), 'self_plateau':float(self_plateau)}


@lru_cache(None)
def gaussian_rule(order):
    z, w = hermgauss(order)
    return np.sqrt(2)*z, w/np.sqrt(np.pi)


def reference(stiffness, temperature, coupling, wavevector, delay, order=32, n=N):
    """Physical phase-space OU propagation and quadrature of pair fields."""
    hessian = stiffness*stiffness_matrix(coupling, n)
    drift = np.block([[np.zeros((n,n)), np.eye(n)],
                      [-hessian, -GAMMA*np.eye(n)]])
    noise = np.zeros((2*n,2*n)); noise[n:,n:] = 2*GAMMA*temperature*np.eye(n)
    stationary = solve_continuous_lyapunov(drift, -noise)
    propagator = expm(delay*drift)
    cross = (propagator@stationary)[:n,:n]
    z, w = gaussian_rule(order)
    result = 0.
    for i,j in product(range(n), repeat=2):
        covariance = np.array([[stationary[i,i], cross[i,j]],
                               [cross[i,j], stationary[j,j]]])
        eig, basis = np.linalg.eigh(covariance)
        assert min(eig)>-1e-11
        transform = basis*np.sqrt(np.maximum(eig, 0.))
        first = transform[0,0]*z[:,None]+transform[0,1]*z[None,:]
        second = transform[1,0]*z[:,None]+transform[1,1]*z[None,:]
        field = np.cos(wavevector*(i-j+first-second))
        result += w@field@w/n
    residual = drift@stationary+stationary@drift.T+noise
    return float(result), float(np.max(abs(residual)))


def main():
    start = time.monotonic()
    cases=[]
    for k,t,c,q,delay in product([.8,1.2],[.04,.12],[0.,1.],[1.,2.5,4.,2*np.pi],[0.,.5,1.,2.,4.]):
        r=responses(k,t,c,q,delay)
        ref,residual=reference(k,t,c,q,delay)
        fine,_=reference(k,t,c,q,delay,48)
        cases.append({'stiffness':k,'temperature':t,'coupling':c,'wavevector':q,'delay':delay,
                      **r,'reference_error':abs(ref-r['physical']),
                      'reference_refinement':abs(fine-ref),'stationary_residual':residual})
    # Preserve the broad screen, including exact/near equivalences and small signals.
    scan=[]
    for q,delay in product(np.linspace(1.,2*np.pi,17),np.linspace(.25,4.,16)):
        rows=[responses(k,t,1.,q,delay) for k,t in product([.8,1.,1.2],[.04,.08,.12])]
        signals=[abs(r['physical']) for r in rows]
        scan.append({'wavevector':float(q),'delay':float(delay),'minimum_signal':min(signals),
                     'minimum_relative_gap':min(abs(r['source']-r['physical'])/max(abs(r['physical']),1e-14) for r in rows),
                     'maximum_relative_gap':max(abs(r['source']-r['physical'])/max(abs(r['physical']),1e-14) for r in rows)})
    calibration=[(temperature,c,delay) for temperature in [.04,.08,.12]
                 for c,delay in [(1.,0.),(0.,.3),(0.,.8)]]
    recovery=[]; max_shared=0.;min_static_slope=float('inf')
    for k in np.linspace(.8,1.2,25):
        target=np.array([responses(k,t,c,2*np.pi,d)['physical'] for t,c,d in calibration])
        source=np.array([responses(k,t,c,2*np.pi,d)['source'] for t,c,d in calibration])
        max_shared=max(max_shared,float(np.max(abs(target-source))))
        def loss(x):return np.sum((np.array([responses(x,t,c,2*np.pi,d)['source'] for t,c,d in calibration])-target)**2)
        fitted=minimize_scalar(loss,bounds=(.8,1.2),method='bounded',options={'xatol':1e-13})
        fit=min([.8,float(fitted.x),1.2],key=loss)
        recovery.append({'stiffness':float(k),'fit':fit,'error':abs(k-fit)})
        inverse=np.linalg.inv(stiffness_matrix(1.))
        difference=np.diag(inverse)[:,None]+np.diag(inverse)[None,:]-2*inverse
        for t in [.04,.08,.12]:
            a=(2*np.pi)**2*t*difference/2
            slope=np.sum(a*np.exp(-a/k))/(N*k*k)
            min_static_slope=min(min_static_slope,float(slope))
    n1=max(abs(responses(k,t,c,q,d,1)['physical']-responses(k,t,c,q,d,1)['source'])
           for k,t,c,q,d in product([.8,1.2],[.04,.12],[0.,1.],[1.,4.],[0.,1.,3.]))
    plateau=max(abs(responses(k,t,1.,q,200.)['physical']-responses(k,t,1.,q,200.)['plateau'])
                for k,t,q in product([.8,1.2],[.04,.12],[1.,4.,2*np.pi]))
    healthy=sorted([r for r in scan if r['minimum_signal']>.2],key=lambda r:r['minimum_relative_gap'],reverse=True)
    report={'status':'bounded_feasibility_only_no_task_or_model_runs','parameters':{'n':N,'pinning':PINNING,'gamma':GAMMA},
            'reference_cases':cases,'broad_scan':scan,'calibration_settings':calibration,'parameter_recovery':recovery,
            'limits':{'single_scatterer_error':n1,'long_time_plateau_error':plateau},
            'summary':{'reference_cases':len(cases),'max_reference_error':max(r['reference_error'] for r in cases),
                       'max_reference_refinement':max(r['reference_refinement'] for r in cases),
                       'max_stationary_residual':max(r['stationary_residual'] for r in cases),
                       'calibration_equivalence':max_shared,'max_fit_error':max(r['error'] for r in recovery),
                       'minimum_static_calibration_derivative':min_static_slope,
                       'ordinary_signal_candidates':healthy[:12],'seconds':time.monotonic()-start}}
    Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
