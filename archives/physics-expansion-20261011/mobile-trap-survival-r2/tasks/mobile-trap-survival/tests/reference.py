"""Independent drift contraction and finite-time survival bounds."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf, log_ndtr, logsumexp

TRUE_PARAMETER = 1.08


def experiment(density=.5, force=1., temperature=0., trap_diffusivity=1.):
    return dict(density=density, force=force, temperature=temperature,
                trap_diffusivity=trap_diffusivity)


def calibration_inputs():
    return [experiment(r, f, 0., b) for r in [.15, .4, .7, 1.]
            for f in [-3., -.5, 1., 2.] for b in [.5, 1.5]]


def hidden_inputs():
    return {
        'cold_anchors': [experiment(r, f, 0., b) for r in [.1, .6, 1.]
                         for f in [-4., 0., 2.] for b in [.5, 1.5]],
        'confined_survivors': [experiment(r, f, t, b) for r in [.6, 1.]
                               for f in [-1., .5, 1.] for t in [1., 2.] for b in [.5, 1.5]],
        'moving_survivors': [experiment(r, f, t, b) for r in [.3, .6]
                            for f in [-4., 2., 4.] for t in [.5, 1.] for b in [.5, 1.5]],
        'force_crossover': [experiment(.6, f, 1.5, b) for f in [-3., -.9, .9, 1.8, 2.4, 3.]
                            for b in [.5, 1.5]],
    }


def contraction(e, mobility, tolerance=1e-11):
    rho, u, diffusivity = e['density'], mobility*abs(e['force']), mobility*e['temperature']
    if diffusivity == 0.:
        return rho*u, u
    if u == 0.:
        return 0., 0.
    # Gaussian empirical target velocity cost plus the Poisson interval cost.
    # Reflection rules out negative optima for nonnegative u.
    def cost(w):
        return (w-u)**2/(4*diffusivity)+rho*w
    result = minimize_scalar(cost, bounds=(0., u), method='bounded', options={'xatol': tolerance})
    w = min([0., result.x, u], key=cost)
    return cost(w), w


def expected_range(drift, diffusivity, time):
    z = abs(drift)*np.sqrt(time/(4*diffusivity))
    if z < 1e-7:
        return 4*np.sqrt(diffusivity*time/np.pi)*(1+z*z/3)
    return ((abs(drift)*time+2*diffusivity/abs(drift))*erf(z)
            +2*np.sqrt(diffusivity*time/np.pi)*np.exp(-z*z))


def bounds(e, mobility, time):
    rho, u, diffusivity = e['density'], mobility*abs(e['force']), mobility*e['temperature']
    trap_diffusivity = e['trap_diffusivity']
    if diffusivity == 0.:
        value = rho*expected_range(u, trap_diffusivity, time)/time
        return value, value
    # Survival is at most E exp(-rho*abs(X_t)). Evaluate the two truncated
    # Gaussian integrals in log space, without underflowing survival itself.
    scale = np.sqrt(time/(2*diffusivity))
    log_moment = logsumexp([
        (rho*rho*diffusivity-rho*u)*time+log_ndtr((u-2*rho*diffusivity)*scale),
        (rho*rho*diffusivity+rho*u)*time+log_ndtr(-(u+2*rho*diffusivity)*scale)])
    lower = -log_moment/time
    _, w = contraction(e, mobility)
    # Tube width balances the initial empty interval, a Girsanov endpoint
    # bound, and target confinement. The trap cost is an exact drifted range.
    a = 2*rho+abs(u-w)/(2*diffusivity)
    ell = (np.pi**2*diffusivity*time/(2*a))**(1/3)
    upper = (rho*expected_range(w, trap_diffusivity, time)/time
             +(u-w)**2/(4*diffusivity)+a*ell/time
             +np.pi**2*diffusivity/(4*ell*ell))
    return lower, upper


def predict(experiments, mobility, tolerance=1e-11):
    return np.asarray([contraction(e, mobility, tolerance)[0] for e in experiments])
