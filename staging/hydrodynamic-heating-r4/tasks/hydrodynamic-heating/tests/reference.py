"""Independent collocation fields and settled real-contact first-law heat."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
TRUE_PARAMETER=1.03
GAMMA,NU=.06,.04
G=.5

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
                  tol=tol,max_nodes=12000,fun_jac=lambda z,y:np.repeat(aa[:,:,None],len(z),axis=2),
                  bc_jac=lambda yl,yr:(la,rb))
    assert sol.success,(sol.message,e,p,tol,sol.x.size)
    return lambda z: sol.sol(z)[:4]+1j*sol.sol(z)[4:]

def first_law_reference(e, p, g=G, field_tol=1e-9, time_tol=2e-9,
                        settle_periods=8, steps_per_period=64, periods=1):
    """Collocation fields, real extension ODE, and mechanical input minus Udot.

    Choose rho=1 as a harmless normalization: v=J/p, kappa=alpha,
    k0=alpha/tau. E0=1 so division by incident mean power multiplies by2.
    No complex heat correction is used in this calculation.
    """
    w, alpha, tau = (e[k] for k in ('frequency', 'friction', 'relaxation'))
    assert tau > 0 and w > 0
    fields = bvp_fields(e, p, field_tol)
    j = fields([0.0])[2, 0]
    period = 2*np.pi/w
    k0 = alpha/tau

    def slip(t):
        return float(np.real(j*np.exp(-1j*w*t)))/p

    def extension_rate(t, x):
        return slip(t) - x/tau

    options = dict(method='DOP853', rtol=time_tol, atol=time_tol*0.01,
                   max_step=period/steps_per_period)
    settled = solve_ivp(lambda t, y: [extension_rate(t, y[0])],
                        (0.0, settle_periods*period), [0.0], **options)
    assert settled.success, settled.message
    initial = settled.y[0, -1]

    def rhs(t, y):
        x = y[0]
        v = slip(t)
        xd = extension_rate(t, x)
        traction = k0*x
        work = 2*traction*v
        # U=(k-T*k')*e^2/2=(1-g)*k0*e^2/2 at the clamped T0.
        udot = 2*(1-g)*k0*x*xd
        heat = work-udot
        dashpot = 2*alpha*(x/tau)**2
        cosine, sine = np.cos(2*w*t), np.sin(2*w*t)
        return [xd, heat, heat*cosine, heat*sine,
                dashpot, dashpot*cosine, dashpot*sine, work, udot]

    end = periods*period
    measured = solve_ivp(rhs, (0.0, end), [initial, 0, 0, 0, 0, 0, 0, 0, 0],
                         dense_output=True, **options)
    assert measured.success, measured.message
    final = measured.y[:, -1]
    values = np.array([final[1], 2*final[2], 2*final[3]])/end
    dashpot = np.array([final[4], 2*final[5], 2*final[6]])/end
    ts = np.linspace(0, period, 257)
    x = measured.sol(ts)[0]
    v = np.real(j*np.exp(-1j*w*ts))/p
    xd = v-x/tau
    work = 2*k0*x*v
    udot = 2*(1-g)*k0*x*xd
    heat = work-udot
    entropy_heat = 2*alpha*(x/tau)**2+2*g*k0*x*xd
    integrated_udot = final[8]
    endpoint_energy = (1-g)*k0*(final[0]**2-initial**2)
    return dict(values=values.tolist(), dashpot=dashpot.tolist(),
                closure_error=float(np.max(abs(heat-entropy_heat))),
                energy_integral_error=float(abs(integrated_udot-endpoint_energy)),
                periodicity=float(abs(final[0]-initial)),
                mean_storage_rate=float(final[8]/end),
                heat_waveform_min=float(heat.min()), heat_waveform_max=float(heat.max()),
                minimum_entropy_production=float(np.min(alpha*(x/tau)**2)),
                mean_work=float(final[7]/end), settle_periods=settle_periods,
                measurement_periods=periods, time_tol=time_tol,
                field_tol=field_tol, max_step=period/steps_per_period,
                ode_evaluations=settled.nfev+measured.nfev)


@lru_cache(maxsize=2048)
def measured(frequency,thickness,friction,relaxation,plasma_frequency):
    e=experiment(frequency,thickness,friction,relaxation)
    return first_law_reference(e,plasma_frequency)['values']

def predict(experiments,plasma_frequency=TRUE_PARAMETER):
    result=[]
    for e in experiments:
        controls={k:e[k] for k in ('frequency','thickness','friction','relaxation')}
        values=measured(**controls,plasma_frequency=plasma_frequency)
        result.append(values[{'mean':0,'in_phase':1,'quadrature':2}[e['readout']]])
    return np.asarray(result)

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
