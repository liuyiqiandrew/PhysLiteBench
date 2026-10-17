"""Independent Brownian crossing-indicator quadrature for the joint scaling limit."""
from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre

TRUE_PARAMETER = .95


@lru_cache(None)
def nodes(number):
    return roots_legendre(number)


def indicator_covariance(x, t, s, diffusivity, angular_nodes=96):
    if min(t, s) == 0:
        return np.zeros_like(x, dtype=float)
    u, w = nodes(angular_nodes)
    angle = np.arcsin(np.sqrt(min(t, s)/max(t, s)))
    theta = (u+1)*angle/2
    a = np.asarray(x)[:, None]/np.sqrt(2*diffusivity*t)
    b = np.asarray(x)[:, None]/np.sqrt(2*diffusivity*s)
    exponent = -(a-b)**2/(2*np.cos(theta)**2)-a*b/(1+np.sin(theta))
    return np.exp(exponent) @ (w*angle/(4*np.pi))


@lru_cache(maxsize=4096)
def covariance(diffusivity, density, t, s, spatial_nodes=512, angular_nodes=96):
    if min(t, s) == 0:
        return 0.0
    u, w = nodes(spatial_nodes)
    radius = 12*np.sqrt(2*diffusivity*max(t, s))
    values = indicator_covariance(radius*u, t, s, diffusivity, angular_nodes)
    # Independent paths at each fixed initial position contribute their covariance.
    # Rank conservation converts the current fluctuation to displacement / density.
    return float(radius*np.dot(w, values)/density)


def predict(experiments, diffusivity):
    return np.asarray([covariance(float(diffusivity),float(e['density']),
                                 float(e['time_a']),float(e['time_b']))
                       for e in experiments])


def calibration_inputs():
    return [dict(density=rho, time_a=t, time_b=t)
            for _ in range(24) for rho in [.7, 1., 1.5] for t in [.3, .7, 1.6, 3.8]]


def hidden_inputs():
    return {
        'ratio_two': [dict(density=rho, time_a=(2*s if j%2 else s),
                          time_b=(s if j%2 else 2*s))
                      for rho in [.7, 1., 1.5] for j,s in enumerate([.3,.55,.8,1.2,1.7])],
        'ratio_four': [dict(density=rho, time_a=(4*s if j%2 else s),
                           time_b=(s if j%2 else 4*s))
                       for rho in [.7, 1., 1.5] for j,s in enumerate([.25,.4,.6,.8,1.])],
        'ratio_eight': [dict(density=rho, time_a=(8*s if j%2 else s),
                            time_b=(s if j%2 else 8*s))
                        for rho in [.7, 1., 1.5] for j,s in enumerate([.25,.3,.4,.5])],
        'diagonal_anchors': [dict(density=rho, time_a=t, time_b=t)
                             for rho in [.8,1.2,1.4] for t in [.25,.6,2.2,4.]]
    }
