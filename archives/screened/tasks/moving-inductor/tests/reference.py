import numpy as np

from scipy.integrate import solve_ivp

def predict(experiments, resistance):
    result = []
    for e in experiments:
        if e['time'] == 0:
            result.append(e['initial_current'])
            continue
        def derivative(t, y):
            l = .6+.4*(e['x0']+e['amplitude']*np.sin(e['omega']*t))
            ld = .4*e['amplitude']*e['omega']*np.cos(e['omega']*t)
            return [(e['voltage']-(resistance+ld)*y[0])/l]
        s = solve_ivp(derivative, (0, e['time']), [e['initial_current']],
                      method='DOP853', rtol=2e-11, atol=1e-12)
        result.append(s.y[0, -1])
    return np.array(result)


def calibration_inputs():
    return [dict(time=float(t), initial_current=i, voltage=v, x0=x,
                 amplitude=0., omega=0.) for x, i, v in [(0.1, .2, 3.), (.5, 1.2, 0.), (.9, -.5, 4.)]
                 for t in np.linspace(.02, 1.4, 32)]

def hidden_inputs():
    return {f'motion_{j}': [dict(time=float(t), initial_current=i, voltage=v,
            x0=.5, amplitude=a, omega=w) for t in np.linspace(.03, 1.5, 30)]
            for j, (i, v, a, w) in enumerate([(1.3, 0., .4, 9.), (.1, 3., .45, 12.), (-.8, -2., .35, 6.)])}

PARAMETER = 'resistance'
TRUE_PARAMETER = 2.3
BOUNDS = (0.2, 8.0)
