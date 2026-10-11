"""Independent no-entry quadrature and confinement bounds."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import erfc

TRUE_PARAMETER = 1.1


def experiment(density=.5, target_diffusivity=0.):
    return dict(density=density, target_diffusivity=target_diffusivity)


def calibration_inputs():
    return [experiment(density=float(r)) for r in np.linspace(.15, .95, 32)]


def hidden_inputs():
    return {
        'stationary_anchors': [experiment(r, 0.) for r in [.1, .3, .7, 1.]],
        'slow_targets': [experiment(r, d) for r in [.1, .35, .7, 1.] for d in [.5, .8]],
        'comparable_diffusion': [experiment(r, d) for r in [.15, .4, .8, 1.] for d in [1., 1.5]],
        'fast_targets': [experiment(r, d) for r in [.1, .3, .6, 1.] for d in [2., 4.]],
    }


def no_entry_coefficient(density, trap_diffusivity, tolerance=1e-10):
    # Reflection principle: integrate the probability an exterior trap hits
    # either boundary. Coordinate is distance/sqrt(4*D_B*t).
    integral = quad(erfc, 0., np.inf, epsabs=tolerance, epsrel=tolerance)[0]
    return 2*density*np.sqrt(4*trap_diffusivity)*integral


def bounds(e, trap_diffusivity, time, tolerance=1e-10):
    lower = no_entry_coefficient(e['density'], trap_diffusivity, tolerance)
    diffusivity, rho = e['target_diffusivity'], e['density']
    if diffusivity == 0.:
        return lower, lower
    # Empty interval plus target ground-state confinement, minimized in
    # scaled half-width u=ell/time**(1/3). Work with logarithms throughout.
    def cost(log_u):
        u = np.exp(log_u)
        return 2*rho*u+np.pi**2*diffusivity/(4*u*u)
    result = minimize_scalar(cost, bounds=(-20., 20.), method='bounded',
                             options={'xatol': tolerance})
    return lower, lower+result.fun*time**(-1/6)


def predict(experiments, trap_diffusivity, tolerance=1e-10):
    # Expected-range convexity gives the lower log-survival bound; the
    # optimized interval has a vanishing t**(-1/6) gap. Their common limit
    # is this quadrature, independently of the target diffusivity.
    return np.asarray([no_entry_coefficient(e['density'], trap_diffusivity, tolerance)
                       for e in experiments])
