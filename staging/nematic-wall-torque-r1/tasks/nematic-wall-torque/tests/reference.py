"""Independent minimized-energy virtual-work reference."""
from functools import lru_cache
import itertools
import numpy as np
from numpy.polynomial.legendre import leggauss

TRUE_PARAMETER = 11.7


def minimized_energy(inner_angle, outer_angle, radius_ratio, modulus,
                     modes=32, saddle_ratio=0.5):
    """Independent Ritz minimization of bulk div/curl energy minus surface flux.

    The returned energy is per unit axial length. No radial differential equation
    or torque formula enters this reference.
    """
    length = np.log(radius_ratio)
    q, w = leggauss(6*modes+16)
    s, w = (q+1)*length/2, w*length/2
    frequencies = np.pi*np.arange(1, modes+1)/length
    basis = np.sin(np.outer(s, frequencies))
    derivative = np.cos(np.outer(s, frequencies))*frequencies
    base = inner_angle+(outer_angle-inner_angle)*s/length
    base_slope = (outer_angle-inner_angle)/length
    coefficients = np.zeros(modes)
    for iteration in range(16):
        theta = base+basis@coefficients
        theta_s = base_slope+derivative@coefficients
        sine, cosine = np.sin(theta), np.cos(theta)
        # At phi=0, the nonzero Cartesian derivatives are
        # dn_x/dx=cos(theta)*theta_s/r, dn_z/dx=-sin(theta)*theta_s/r,
        # dn_y/dy=sin(theta)/r. The r^2 Jacobian cancels those denominators.
        div_scaled = cosine*theta_s+sine
        curl_scaled = sine*theta_s
        energy = np.pi*modulus*np.dot(w, div_scaled**2+curl_scaled**2)
        energy -= 2*np.pi*saddle_ratio*modulus*(np.sin(outer_angle)**2-np.sin(inner_angle)**2)
        dtheta = 2*theta_s*np.cos(2*theta)+2*sine*cosine
        dslope = 2*theta_s+2*sine*cosine
        gradient = np.pi*modulus*(basis.T@(w*dtheta)+derivative.T@(w*dslope))
        h00 = -4*theta_s*np.sin(2*theta)+2*np.cos(2*theta)
        h01 = 2*np.cos(2*theta)
        hessian = np.pi*modulus*(basis.T@((w*h00)[:, None]*basis)
                    +basis.T@((w*h01)[:, None]*derivative)
                    +derivative.T@((w*h01)[:, None]*basis)
                    +2*derivative.T@(w[:, None]*derivative))
        step = np.linalg.solve(hessian, gradient)
        if np.max(np.abs(step)) < 2e-13:
            return float(energy), {
                "iterations": iteration+1,
                "gradient_max": float(np.max(np.abs(gradient))),
                "reduced_energy": float(np.pi*modulus*np.dot(w, theta_s**2+sine**2)),
                "smallest_hessian_eigenvalue": float(np.linalg.eigvalsh(hessian)[0]),
            }
        coefficients -= step
    raise RuntimeError("Ritz minimization did not converge")


def energy_torque(inner_radius, radius_ratio, inner_angle, outer_angle, modulus,
                  modes=32, saddle_ratio=0.5, step=2e-4):
    energies = [minimized_energy(inner_angle, outer_angle+j*step, radius_ratio,
                                modulus, modes, saddle_ratio)[0]
                for j in [-2, -1, 1, 2]]
    derivative = (energies[0]-8*energies[1]+8*energies[2]-energies[3])/(12*step)
    return derivative/(2*np.pi*inner_radius*radius_ratio)



def flat(thickness, inner, outer):
    return dict(geometry='flat', thickness=thickness, inner_angle=inner, outer_angle=outer)


def annulus(radius, ratio, inner, outer):
    return dict(geometry='annulus', inner_radius=radius, radius_ratio=ratio,
                inner_angle=inner, outer_angle=outer)


def calibration_inputs():
    return [flat(length, low, high) for length, low, high in
            itertools.product([.75, 1., 1.25], [.05, .15], [.3, .4, .55, .7])]*6


def hidden_inputs():
    return {
        'flat_anchors': [flat(.8, .1, .35), flat(1.3, .05, .65)],
        'narrow_cells': [annulus(.75, 2., .05, .35), annulus(1.2, 2.2, .15, .65),
                         annulus(.9, 2.1, .1, .5), annulus(1.1, 2.4, .08, .7)],
        'wide_cells': [annulus(.8, 4., .12, .4), annulus(1.3, 3.7, .05, .7),
                       annulus(1., 3.5, .15, .55), annulus(.7, 3.9, .07, .3)],
        'mixed_tilts': [annulus(.85, 2.8, .05, .7), annulus(1.15, 3.2, .15, .3),
                        annulus(.95, 2.5, .08, .48), annulus(1.25, 3.6, .13, .6)],
    }


@lru_cache(maxsize=2048)
def unit_response(radius, ratio, inner, outer, modes):
    return energy_torque(radius, ratio, inner, outer, 1.0, modes=modes)


def predict(experiments, elastic_constant=TRUE_PARAMETER, modes=32):
    values=[]
    for e in experiments:
        if e['geometry']=='flat':
            value=(e['outer_angle']-e['inner_angle'])/e['thickness']
        else:
            value=unit_response(e['inner_radius'], e['radius_ratio'], e['inner_angle'], e['outer_angle'], modes)
        values.append(elastic_constant*value)
    return np.array(values)
