from pathlib import Path
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm

GAMMA=.06
BETA=.1

def modes(z,e,plasma):
    w=e['frequency'];d=e['thickness'];k=w*np.sin(e['angle']);gamma=GAMMA;beta=BETA
    epsilon=1-plasma**2/(w*(w+1j*gamma));sigma=plasma**2/(gamma-1j*w)
    qt=np.sqrt(epsilon*w*w-k*k+0j);kl2=(w*(w+1j*gamma)-plasma**2)/beta**2;ql=np.sqrt(kl2-k*k+0j)
    fields=np.zeros((len(np.atleast_1d(z)),6,4),complex)
    for j,(q,s,anchor) in enumerate([(qt,1,0),(qt,-1,d),(ql,1,0),(ql,-1,d)]):
        wave=np.exp(1j*s*q*(np.atleast_1d(z)-anchor))
        if j<2:
            ex=s*q/(w*epsilon);ez=-k/(w*epsilon)
            state=np.array([ex,ez,1,sigma*ex,sigma*ez,0])
        else:
            state=np.array([-1j*k,-1j*s*q,0,w*k,w*s*q,kl2])
        fields[:,:,j]=wave[:,None]*state
    return fields

def wave_solution(e,plasma):
    a,b=modes([0.,e['thickness']],e,plasma);c=np.cos(e['angle'])
    matrix=np.array([a[0]+c*a[2],b[0]-c*b[2],a[4],b[4]])
    coeff=np.linalg.solve(matrix,np.array([2*c,0,0,0],complex))
    return coeff,np.linalg.cond(matrix)

def wave_fields(z,e,plasma):
    coeff,_=wave_solution(e,plasma)
    return modes(z,e,plasma)@coeff

def heat(e,plasma,readout='drag'):
    x,wg=leggauss(100);lo,hi=np.array(e['window'])*e['thickness'];z=(lo+hi)/2+(hi-lo)*x/2
    f=wave_fields(z,e,plasma)
    c=np.cos(e['angle']);w=e['frequency']
    if readout=='drag':integrand=GAMMA/plasma**2*np.sum(abs(f[:,3:5])**2,axis=1)/c
    else:integrand=GAMMA*plasma**2/(w*w+GAMMA**2)*np.sum(abs(f[:,:2])**2,axis=1)/c
    return (hi-lo)/2*np.dot(wg,integrand)

def ode_solution(e,plasma):
    w=e['frequency'];k=w*np.sin(e['angle']);c=np.cos(e['angle']);gamma=GAMMA;beta=BETA
    def derivative(y):
        ex,h,jz,rho=y
        ez=(jz-1j*k*h)/(1j*w)
        jx=(plasma**2*ex-1j*beta**2*k*rho)/(gamma-1j*w)
        return np.array([1j*w*h+1j*k*ez,1j*w*ex-jx,1j*w*rho-1j*k*jx,(plasma**2*ez-(gamma-1j*w)*jz)/beta**2])
    matrix=np.column_stack([derivative(v) for v in np.eye(4)])
    transition=expm(matrix*e['thickness'])
    bc=np.array([[1,c,0,0],np.array([1,-c,0,0])@transition,[0,0,1,0],transition[2]])
    initial=np.linalg.solve(bc,[2*c,0,0,0])
    def state(z):return expm(matrix*z)@initial
    def flux(z):
        ex,h,jz,rho=state(z)
        return np.real(ex*np.conj(h)+beta**2/plasma**2*rho*np.conj(jz))/c
    lo,hi=np.array(e['window'])*e['thickness']
    return flux(lo)-flux(hi),state,matrix

def experiment(d,w,angle,window):return dict(thickness=d,frequency=w,angle=angle,window=list(window))

if __name__=='__main__':
    rows=[];maximum=0.;energy=0.;calibration=0.
    for d,w,angle,window in [(d,w,a,v) for d in [.25,.5,.8] for w in [.8,1.05,1.2,1.4] for a in [0.,.5,.9] for v in [[0,.3],[.3,.7],[.7,1.]]]:
        e=experiment(d,w,angle,window);truth=heat(e,1.);shortcut=heat(e,1.,'local');r,state,matrix=ode_solution(e,1.)
        maximum=max(maximum,abs(r-truth))
        endpoints=wave_fields([0,d],e,1.);reflection=endpoints[0,2]-1;transmission=endpoints[1,2]
        total=heat(dict(e,window=[0,1]),1.);energy=max(energy,abs(total-(1-abs(reflection)**2-abs(transmission)**2)))
        if angle==0:calibration=max(calibration,abs(truth-shortcut))
        rows.append(dict(input=e,truth=truth,shortcut=shortcut,relative_error=abs(shortcut-truth)/truth,reference=r,condition=wave_solution(e,1.)[1]))
    report=dict(known_damping=GAMMA,known_pressure_speed=BETA,maximum_independent_reference_error=maximum,maximum_global_energy_balance_error=energy,normal_incidence_calibration_equivalence=calibration,cases=rows)
    Path('results/hydrodynamic-heating-prototype.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
    print('largest',sorted(rows,key=lambda r:r['relative_error'])[-8:])
