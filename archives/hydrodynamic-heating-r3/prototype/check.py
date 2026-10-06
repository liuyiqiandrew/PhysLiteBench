"""Bounded Maxwell-contact calorimetry check; no task or model evaluation."""
import json, time
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import minimize_scalar

GAMMA, NU = .06, .04
X, W = leggauss(96)

def modes(z, w, d, p):
    roots = np.roots([NU, GAMMA-1j*w-NU*w*w,
                     -w*w*(GAMMA-1j*w)-1j*w*p*p])
    vertical = np.sqrt(roots+0j)
    z = np.atleast_1d(z)
    out = np.zeros((len(z),4,4),complex)
    for j in range(2):
        current = (roots[j]-w*w)/(1j*w)
        for side in range(2):
            derivative = 1j*(1 if side == 0 else -1)*vertical[j]
            anchor = 0 if side == 0 else d
            wave = np.exp(derivative*(z-anchor))
            out[:,:,2*j+side] = wave[:,None]*np.array([1,derivative,current,derivative*current])
    return out

def modal_fields(e,p):
    w,d,alpha,tau = (e[k] for k in ('frequency','thickness','friction','relaxation'))
    left,right = modes([0,d],w,d,p)
    impedance=alpha/(1-1j*w*tau)
    boundary=np.array([left[1]+1j*w*left[0],right[1]-1j*w*right[0],
                       NU*left[3]-impedance*left[2],NU*right[3]+impedance*right[2]])
    coeff=np.linalg.solve(boundary,[2j*w,0,0,0])
    return lambda z: (modes(z,w,d,p)@coeff).T

def response(e,p):
    w,d,alpha,tau = (e[k] for k in ('frequency','thickness','friction','relaxation'))
    field=modal_fields(e,p)
    electric,_,current,_=field([0,d])
    traction=alpha*current/(1-1j*w*tau)
    mean=abs(traction)**2/(alpha*p*p)
    heat2=traction**2/(alpha*p*p)
    work2=traction*current/(p*p)
    z=(X+1)*d/2
    _,_,j,jz=field(z)
    bulk=d/2*np.dot(W,GAMMA*abs(j)**2+NU*abs(jz)**2)/(p*p)
    absorption=1-abs(electric[0]-1)**2-abs(electric[1])**2
    return dict(mean=mean,physical=heat2,source=work2,bulk=bulk,
                absorption=absorption,energy_residual=float(abs(absorption-bulk-sum(mean))),
                current=current,traction=traction)

def bvp_fields(e,p,tol=1e-9):
    w,d,alpha,tau = (e[k] for k in ('frequency','thickness','friction','relaxation'))
    a=np.array([[0,1,0,0],[-w*w,0,-1j*w,0],[0,0,0,1],
                [-p*p/NU,0,(GAMMA-1j*w)/NU,0]],complex)
    aa=np.block([[a.real,-a.imag],[a.imag,a.real]])
    impedance=alpha/(1-1j*w*tau)
    left=np.array([[1j*w,1,0,0],[0,0,0,0],[0,0,-impedance,NU],[0,0,0,0]],complex)
    right=np.array([[0,0,0,0],[-1j*w,1,0,0],[0,0,0,0],[0,0,impedance,NU]],complex)
    la=np.block([[left.real,-left.imag],[left.imag,left.real]])
    rb=np.block([[right.real,-right.imag],[right.imag,right.real]])
    rhs=np.array([0,0,0,0,2*w,0,0,0])
    z=np.linspace(0,d,25)
    sol=solve_bvp(lambda z,y:aa@y,lambda yl,yr:la@yl+rb@yr-rhs,z,np.zeros((8,len(z))),
                  tol=tol,max_nodes=4000,fun_jac=lambda z,y:np.repeat(aa[:,:,None],len(z),axis=2),
                  bc_jac=lambda yl,yr:(la,rb))
    assert sol.success,sol.message
    return lambda z: sol.sol(z)[:4]+1j*sol.sol(z)[4:]

def contact_time_reference(j,w,alpha,tau,p,tol=2e-10):
    # Integrate actual series-contact ODE. Obtain periodic initial state by its affine period map.
    period=2*np.pi/w
    drive=lambda t:float(np.real(j*np.exp(-1j*w*t)))
    sol=solve_ivp(lambda t,y:[(alpha*drive(t)-y[0])/tau],(0,period),[0.],rtol=tol,atol=tol*.02)
    initial=sol.y[0,-1]/(-np.expm1(-period/tau))
    def rhs(t,y):
        r=y[0]; v=drive(t); rd=(alpha*v-r)/tau
        q=2*r*r/(alpha*p*p); work=2*r*v/(p*p)
        return [rd,q,q*np.cos(2*w*t),q*np.sin(2*w*t),work,
                work*np.cos(2*w*t),work*np.sin(2*w*t)]
    final=solve_ivp(rhs,(0,period),[initial,0,0,0,0,0,0],rtol=tol,atol=tol*.02).y[:,-1]
    return dict(mean=final[1]/period,physical=2*(final[2]+1j*final[3])/period,
                source=2*(final[5]+1j*final[6])/period,
                mean_work=final[4]/period,periodicity=abs(final[0]-initial))

def independent(e,p,tol=2e-10):
    f=bvp_fields(e,p,tol*5)
    j=f([0,e['thickness']])[2]
    return [contact_time_reference(v,e['frequency'],e['friction'],e['relaxation'],p,tol) for v in j]

def record(e,p,ref=True):
    v=response(e,p)
    out=dict(input=e,plasma_frequency=p,mean=v['mean'].tolist(),
             physical=[[z.real,z.imag] for z in v['physical']],
             source=[[z.real,z.imag] for z in v['source']],
             relative_complex_gap=(abs(v['source']-v['physical'])/abs(v['physical'])).tolist(),
             energy_residual=v['energy_residual'])
    if ref:
        rr=independent(e,p)
        out['reference_mean_error']=float(max(abs(rr[i]['mean']-v['mean'][i]) for i in range(2)))
        out['reference_harmonic_error']=float(max(abs(rr[i]['physical']-v['physical'][i]) for i in range(2)))
        out['reference_source_error']=float(max(abs(rr[i]['source']-v['source'][i]) for i in range(2)))
        out['reference_periodicity']=float(max(r['periodicity'] for r in rr))
    return out

def main():
    start=time.monotonic()
    rows=[]
    for p in [.85,1.15]:
        for w in [.7,1.5]:
            for d in [.3,1.2]:
                for alpha in [.08,.4]:
                    for tau in [.2,1.2]:
                        rows.append(record(dict(frequency=w,thickness=d,friction=alpha,relaxation=tau),p))
    cal=[dict(frequency=w,thickness=d,friction=.2,relaxation=tau)
         for w in [.75,1.1,1.45] for d,tau in [(.5,.35),(.9,.8)]]
    predict=lambda p:np.array([response(e,p)['mean'][0] for e in cal])
    grid=np.linspace(.85,1.15,81); values=np.array([predict(p) for p in grid])
    fits=[]
    for p in np.linspace(.85,1.15,25):
        truth=predict(p)
        fit=minimize_scalar(lambda x:np.sum((predict(x)-truth)**2),bounds=(.85,1.15),method='bounded',options={'xatol':1e-12})
        fits.append(dict(true=float(p),fitted=float(fit.x),error=float(abs(fit.x-p))))
    e=dict(frequency=1.2,thickness=.8,friction=.2,relaxation=.7)
    refinement=[]
    for tol in [2e-10,2e-11]:
        r=independent(e,1.03,tol)
        refinement.append([[v['mean'],v['physical'].real,v['physical'].imag] for v in r])
    limits={}
    for key,seq in [('relaxation',[.3,.03,.003,0]),('friction',[.01,.001,.0001,100,1000,10000])]:
        limits[key]=[record(dict(e,**{key:x}),1.03,False) for x in seq]
    summary=dict(cases=len(rows),max_reference_mean_error=max(r['reference_mean_error'] for r in rows),
                 max_reference_harmonic_error=max(r['reference_harmonic_error'] for r in rows),
                 max_reference_source_error=max(r['reference_source_error'] for r in rows),
                 max_energy_residual=max(r['energy_residual'] for r in rows),
                 minimum_mean=min(min(r['mean']) for r in rows),maximum_mean=max(max(r['mean']) for r in rows),
                 minimum_gap=min(min(r['relative_complex_gap']) for r in rows),
                 maximum_fit_error=max(r['error'] for r in fits),
                 minimum_calibration_step_derivative=np.min(np.diff(values,axis=0)/(grid[1]-grid[0]),axis=0).tolist(),
                 maximum_calibration_step_derivative=np.max(np.diff(values,axis=0)/(grid[1]-grid[0]),axis=0).tolist(),
                 refinement_difference=float(np.max(abs(np.array(refinement[0])-refinement[1]))),
                 seconds=time.monotonic()-start)
    report=dict(scope='Bounded author prototype only; no task/data or model calls.',summary=summary,rows=rows,
                calibration=cal,identifiability_grid=grid.tolist(),calibration_predictions=values.tolist(),
                noiseless_fits=fits,refinement=refinement,limits=limits,
                qualification='Calibration injectivity is sampled evidence, not a proof. Full complex coefficient is represented by its two ordinary phase quadratures. Negative instantaneous source work is physically possible stored-energy transfer, but is not thermostat heat. No difficulty or qualification claim.')
    Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    assert summary['max_reference_harmonic_error']<1e-7
    assert summary['max_energy_residual']<1e-9
    assert summary['maximum_fit_error']<1e-6

if __name__=='__main__':main()
