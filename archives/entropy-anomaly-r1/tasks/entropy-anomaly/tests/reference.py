from functools import lru_cache
import numpy as np
from scipy.sparse import bmat, csc_matrix, diags
from scipy.sparse.linalg import spsolve

PARAMETER = 'friction'
TRUE_PARAMETER = 1.1


def calibration_inputs():
    return [dict(force=float(f),temperature=float(t),contrast=0.,wavenumber=1+i%3)
            for i,(f,t) in enumerate(zip(np.linspace(.2,1.4,100),np.linspace(.8,1.4,100)))]


def hidden_inputs():
    return {
        'thermal_contrast':[dict(force=.3,temperature=1.,contrast=float(a),wavenumber=1) for a in np.linspace(.25,.65,8)],
        'thermal_wavelength':[dict(force=float(f),temperature=1.2,contrast=.55,wavenumber=1+i%3) for i,f in enumerate(np.linspace(.4,.8,9))],
        'driven_bath':[dict(force=float(f),temperature=float(t),contrast=.6,wavenumber=2) for f,t in zip([-.9,-.7,-.5,-.3,.3,.5,.7,.9],np.linspace(.8,1.4,8))],
    }


@lru_cache(None)
def spatial_grid(points):
    x = np.arange(points)*2*np.pi/points
    modes = np.fft.fftfreq(points,1/points)
    derivative = np.fft.ifft(1j*modes[:,None]*np.fft.fft(np.eye(points),axis=0),axis=0).real
    return x, csc_matrix(derivative)


def kinetic_state(experiment, friction, mass, points=95, modes=48):
    # Stationary Kramers equation, before eliminating velocity. Expand the
    # density in Gaussian-weighted normalized Hermites of w=sqrt(m/T0)*v.
    x, derivative = spatial_grid(points)
    base = experiment['temperature']
    temperature = base*(1+experiment['contrast']*np.cos(experiment['wavenumber']*x))
    identity = diags(np.ones(points))
    blocks = [[None]*modes for _ in range(modes)]
    for n in range(modes):
        blocks[n][n] = -friction*n*identity
        if n:
            blocks[n][n-1] = (-np.sqrt(mass*base*n)*derivative
                             +np.sqrt(mass*n/base)*experiment['force']*identity)
        if n+1 < modes:
            blocks[n][n+1] = -np.sqrt(mass*base*(n+1))*derivative
        if n >= 2:
            blocks[n][n-2] = friction*np.sqrt(n*(n-1))*diags(temperature/base-1)
    generator = bmat(blocks,format='lil')
    # Replace the redundant probability-balance equation by normalization.
    generator[0,:] = 0
    generator[0,:points] = 2*np.pi/points
    rhs = np.zeros(points*modes); rhs[0] = 1
    coefficients = spsolve(generator.tocsc(),rhs).reshape(modes,points)
    return x, temperature, coefficients


def finite_mass_entropy(experiment, friction, mass, points=95, modes=48):
    _, temperature, coefficients = kinetic_state(experiment,friction,mass,points,modes)
    # Converting the stated Stratonovich bath heat to its mean at finite m:
    # gamma * integral [<v^2>_x/T(x) - rho(x)/m] dx.
    second = experiment['temperature']*(coefficients[0]+np.sqrt(2)*coefficients[2])
    return float(friction/mass*np.mean(second/temperature-coefficients[0])*2*np.pi)


def predict(experiments, friction, points=95, modes=48, epsilon=.004):
    values = []
    for e in experiments:
        if e['contrast'] == 0:
            # Uniform-temperature driven Langevin motion is exactly Gaussian
            # at every mass and has mean velocity force/friction.
            values.append(e['force']**2/(friction*e['temperature']))
            continue
        mass = epsilon*friction**2/(e['temperature']*e['wavenumber']**2)
        rates = [finite_mass_entropy(e,friction,mass/factor,points,modes) for factor in [1,2,4]]
        # Three positive masses, quadratic Richardson extrapolation to zero.
        values.append(rates[0]/3-2*rates[1]+8*rates[2]/3)
    return np.array(values)
