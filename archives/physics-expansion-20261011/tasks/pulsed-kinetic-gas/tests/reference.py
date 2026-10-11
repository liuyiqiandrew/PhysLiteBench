"""Direct Gaussian integration of Langevin trajectories; no Fourier-Bessel expansion."""
import numpy as np
from scipy.integrate import quad

TRUE_PARAMETER = .17


def predict(experiments, friction, nx=256, nz=2049):
    z = np.linspace(-10., 10., nz)
    weight = np.exp(-z*z/2)/np.sqrt(2*np.pi)
    x = (2*np.pi*np.arange(nx)/nx)[:, None]
    result = []
    for e in experiments:
        gamma = friction
        tau, u, speed = e['delay'], e['after'], e['thermal_speed']
        def a(t):
            return -np.expm1(-gamma*t)/gamma
        # Initial thermal velocity plus the integrated white-noise forcing.
        def variance(t):
            return speed**2*a(t)**2 + 2*gamma*speed**2*quad(lambda r: a(r)**2, 0., t)[0]
        v1, v2 = variance(tau), variance(u)
        cov = speed**2*a(tau)*np.exp(-gamma*tau)*a(u)
        cov += 2*gamma*speed**2*a(u)*quad(lambda r: a(r)*np.exp(-gamma*r), 0., tau)[0]
        first = np.sqrt(v1)*z
        conditional = cov/v1*first if v1 else np.zeros_like(z)
        remaining = max(0., v2-cov*cov/v1) if v1 else v2
        kick = e['a1']*np.sin(e['k1']*x+e['phase1'])
        at_second = x+first+a(tau)*kick
        final_mean = x+first+conditional+a(tau+u)*kick
        final_mean += a(u)*e['a2']*np.sin(e['k2']*at_second+e['phase2'])
        phase = e['mode']*final_mean
        readout = np.cos(phase) if e['component']=='cos' else np.sin(phase)
        readout *= np.exp(-.5*e['mode']**2*remaining)
        result.append(float(np.trapezoid(readout*weight, z, axis=1).mean()))
    return np.asarray(result)


def experiment(**changes):
    row = dict(thermal_speed=.8, a1=.6, a2=.7, k1=1, k2=2,
               phase1=.2, phase2=.7, delay=2., after=2., mode=1, component='cos')
    row.update(changes)
    return row


def hidden_inputs():
    return {
        'single_impulse': [experiment(a1=0., k2=k, mode=k, after=t, delay=1.3,
                                     phase2=.4, thermal_speed=.6)
                           for k in [1, 2] for t in [.6, 1.2, 1.8]],
        'matched_delays': [experiment(delay=t, after=t, a1=a)
                           for t in [1.6, 2., 2.5, 3.] for a in [.5, .9]],
        'phase_and_sign': [experiment(phase1=p, phase2=-.5, a2=a, component=c)
                           for p in [-.6, .8] for a in [-.8, .6] for c in ['cos', 'sin']],
        'different_modes': [experiment(k1=k1, k2=k2, mode=k2-k1, delay=t,
                                       after=t*k1/(k2-k1)+offset, thermal_speed=.6)
                            for k1,k2,t in [(1,3,2.), (2,3,1.3)] for offset in [-.25, .25]],
    }


def calibration_inputs():
    return [experiment(a1=0., a2=a, k2=1, mode=1, thermal_speed=s,
                       after=t, delay=1., phase2=p)
            for s in [.5, .8] for a in [.5, .9] for t in [.5, 1., 1.5, 2., 2.5]
            for p in [.2, .8]]
