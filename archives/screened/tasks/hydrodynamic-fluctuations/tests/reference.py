"""Independent SI-units force/covariance ODE reference."""
import numpy as np
from scipy.integrate import solve_ivp

PARAMETER = 'viscosity'
TRUE_PARAMETER = .0011


def experiment(measurement, time, separation=5., factors=(1., 1.), shift=(.12, 0.), bead=0, pair=(0, 1)):
    return dict(measurement=measurement, time=float(time), separation=float(separation),
                factors=list(factors), shift=list(shift), bead=bead, pair=list(pair))


def calibration_inputs():
    return [experiment('mean', t, separation=d, factors=f, shift=s, bead=b)
            for d, f, s, b in [(5., (1., 1.), (.12, 0.), 0),
                               (8., (.7, 1.4), (.1, -.08), 1)]
            for t in np.linspace(.004, .12, 50)]


def hidden_inputs():
    return {name: [experiment('covariance', t, separation=d, factors=f, shift=(0., 0.))
                   for t in np.linspace(.002, .16, 32)]
            for name, d, f in [('equal_traps', 5., (1., 1.)),
                               ('unequal_traps', 6., (.6, 1.6)),
                               ('swapped_traps', 8., (1.5, .7))]}


def predict(experiments, viscosity):
    out = []
    for e in experiments:
        mu = 1/(6*np.pi*viscosity*.5e-6)
        cross = 1/(4*np.pi*viscosity*e['separation']*1e-6)
        mobility = np.array([[mu, cross], [cross, mu]])
        stiffness = np.diag(.3e-6*np.array(e['factors']))
        rate = mobility@stiffness
        # Represent covariance in um^2 and mean in um for balanced ODE scales.
        source = 2*1.380649e-23*300*mobility*1e12
        shift = np.array(e['shift'])
        def rhs(t, state):
            mean, covariance = state[:2], state[2:].reshape(2, 2)
            return np.r_[-rate@(mean-shift), (-rate@covariance-covariance@rate.T+source).ravel()]
        if e['time'] == 0:
            state = np.zeros(6)
        else:
            state = solve_ivp(rhs, (0., e['time']), np.zeros(6), method='DOP853', rtol=2e-11, atol=2e-13).y[:, -1]
        out.append(state[e['bead']] if e['measurement']=='mean' else state[2:].reshape(2, 2)[tuple(e['pair'])])
    return np.array(out)
