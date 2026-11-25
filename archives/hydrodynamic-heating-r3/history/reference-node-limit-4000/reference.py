"""Independent Maxwell/momentum BVP and time-domain contact calorimetry."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_bvp,solve_ivp
TRUE_PARAMETER=1.03
GAMMA,NU=.06,.04
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

@lru_cache(maxsize=2048)
def measured(frequency,thickness,friction,relaxation,plasma_frequency,tolerance=2e-10):
    e=dict(frequency=frequency,thickness=thickness,friction=friction,relaxation=relaxation)
    current=bvp_fields(e,plasma_frequency,tolerance*5)([0.])[2,0]
    return contact_time_reference(current,frequency,friction,relaxation,plasma_frequency,tolerance)


def predict(experiments,plasma_frequency=TRUE_PARAMETER,tolerance=2e-10):
    values=[]
    for e in experiments:
        controls={k:e[k] for k in ('frequency','thickness','friction','relaxation')}
        r=measured(**controls,plasma_frequency=plasma_frequency,tolerance=tolerance)
        values.append({'mean':r['mean'],'in_phase':r['physical'].real,'quadrature':r['physical'].imag}[e['readout']])
    return np.asarray(values)


def experiment(w,d,alpha,tau,readout='mean'):
    return dict(frequency=w,thickness=d,friction=alpha,relaxation=tau,readout=readout)


def calibration_inputs():
    return [experiment(w,d,.2,tau) for w in [.75,1.1,1.45] for d,tau in [(.5,.35),(.9,.8)]]


def hidden_inputs():
    controls={
      'frequency_response':[(.75,.7,.2,.65),(1.,.7,.2,.65),(1.25,.7,.2,.65),(1.5,.7,.2,.65)],
      'contact_response':[(1.1,.8,.08,.8),(1.1,.8,.16,.8),(1.1,.8,.28,.8),(1.1,.8,.4,.8)],
      'slab_response':[(.9,.35,.15,.5),(1.2,.6,.25,.6),(1.4,.9,.35,.7),(1.,1.2,.18,1.)]}
    groups={name:[experiment(*c,r) for c in settings for r in ['in_phase','quadrature']] for name,settings in controls.items()}
    groups['mean_anchor']=[experiment(.8,.4,.1,.3),experiment(1.,.7,.3,.6),experiment(1.3,1.,.2,.9),experiment(1.45,1.15,.35,1.1)]
    return groups
