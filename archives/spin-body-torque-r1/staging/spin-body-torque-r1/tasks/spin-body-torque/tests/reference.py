"""Independent nonlinear laboratory-frame Gilbert propagation and spin balance."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp

TRUE_PARAMETER = 1.07


def experiment(bias=1.1, frequency=.95, amplitude=.025, phase=.3, component='x'):
    return dict(bias=bias,frequency=frequency,amplitude=amplitude,phase=phase,component=component)


def laboratory_derivative(time,m,bias,frequency,amplitude):
    bx,by = amplitude*np.cos(frequency*time),amplitude*np.sin(frequency*time)
    x,y,z = m
    first = np.array([y*bias-z*by,z*bx-x*bias,x*by-y*bx])
    second = np.array([y*first[2]-z*first[1],z*first[0]-x*first[2],x*first[1]-y*first[0]])
    return -(first+.18*second)/(1+.18**2)


@lru_cache(maxsize=512)
def periodic_solution(bias,frequency,amplitude,decays=28,rtol=2e-10):
    period = 2*np.pi/frequency
    end = (np.ceil(decays*(1+.18**2)/(.18*bias*period))+2)*period
    solution = solve_ivp(lambda t,m:laboratory_derivative(t,m,bias,frequency,amplitude),
                         [0.,end],[0.,0.,1.],method='DOP853',rtol=rtol,atol=rtol*.01,dense_output=True)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.sol,end,period


def torque(experiment,moment=TRUE_PARAMETER,decays=28,rtol=2e-10):
    bias,frequency,amplitude,phase = [experiment[k] for k in ['bias','frequency','amplitude','phase']]
    solution,end,period = periodic_solution(bias,frequency,amplitude,decays,rtol)
    time = end-period+(phase%(2*np.pi))/frequency
    m = solution(time)
    derivative = laboratory_derivative(time,m,bias,frequency,amplitude)
    field = np.array([amplitude*np.cos(frequency*time),amplitude*np.sin(frequency*time),bias])
    external = moment*np.array([m[1]*field[2]-m[2]*field[1],
                               m[2]*field[0]-m[0]*field[2],m[0]*field[1]-m[1]*field[0]])
    spin_rate = -moment*derivative
    return external-spin_rate


def predict(experiments,moment=TRUE_PARAMETER):
    indices = {'x':0,'y':1,'z':2}
    return np.array([torque(e,moment)[indices[e['component']]] for e in experiments])


def calibration_inputs():
    return [experiment(b,w,a,phase,'z') for _ in range(12)
            for b in [.9,1.1,1.3] for w in [.7,.95,1.2] for a,phase in [(.015,.2),(.03,-.6)]]


def hidden_inputs():
    return {
        'transverse_x':[experiment(.85,w,.025,.3,'x') for w in [.55,.75,.95,1.2]],
        'transverse_y':[experiment(1.1,w,.03,.7,'y') for w in [.65,.85,1.05,1.3]],
        'phase_scan':[experiment(1.25,1.05,.02,p,'x') for p in [-1.2,-.3,.4,.9]],
        'longitudinal_checks':[experiment(.82,.9,.038,-2.,'z'),experiment(1.37,.45,.011,2.,'z')],
    }
