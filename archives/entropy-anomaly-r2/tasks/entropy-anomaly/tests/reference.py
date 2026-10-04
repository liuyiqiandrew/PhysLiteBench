"""Finite-mass two-velocity Kramers moments, then a controlled mass limit."""
from functools import lru_cache
import numpy as np
from scipy.sparse import bmat, csc_matrix, diags, eye
from scipy.sparse.linalg import spsolve

PARAMETER = 'friction'
TRUE_PARAMETER = 1.1


def calibration_inputs():
    return [dict(force_x=float(f),force_y=float(.25+.4*np.sin(i/19)**2),drag_ratio=1.,
                 temperature=float(t),contrast=float(a),wavenumber=1+i%3)
            for i,(f,t,a) in enumerate(zip(np.linspace(.15,.8,100),np.linspace(.8,1.4,100),np.linspace(.25,.6,100)))]


def hidden_inputs():
    return {
        'slow_transverse_drag':[dict(force_x=.15,force_y=.1,drag_ratio=float(r),temperature=1.,contrast=.6,wavenumber=2) for r in np.linspace(.25,.45,8)],
        'fast_transverse_drag':[dict(force_x=.3,force_y=.4,drag_ratio=float(r),temperature=1.2,contrast=.55,wavenumber=2) for r in np.linspace(2.,8.,9)],
        'mixed_drive':[dict(force_x=float(f),force_y=float(.2+.03*i),drag_ratio=float(r),temperature=float(t),contrast=.6,wavenumber=3)
                       for i,(f,r,t) in enumerate(zip([-.5,-.3,.2,.4,-.4,.3,.5,-.2],[.3,.4,3.,4.,5.,6.,7.,8.],np.linspace(.8,1.4,8)))],
    }


@lru_cache(None)
def spatial_grid(points):
    x = np.arange(points)*2*np.pi/points
    modes = np.fft.fftfreq(points,1/points)
    derivative = np.fft.ifft(1j*modes[:,None]*np.fft.fft(np.eye(points),axis=0),axis=0).real
    return x, csc_matrix(derivative)


def kinetic_state(experiment, friction, mass, points=79, modes=32):
    # The y position is uniform. In velocity variables wi=sqrt(m/T0)*vi,
    # the Hermite coefficient equations are triangular in y degree. Degrees
    # 0,1,2 suffice exactly for calorimetry; no higher y degree feeds them.
    x, derivative = spatial_grid(points)
    base = experiment['temperature']
    temperature = base*(1+experiment['contrast']*np.cos(experiment['wavenumber']*x))
    identity = diags(np.ones(points))
    blocks = [[None]*modes for _ in range(modes)]
    for n in range(modes):
        blocks[n][n] = -friction*n*identity
        if n:
            blocks[n][n-1] = (-np.sqrt(mass*base*n)*derivative
                             +np.sqrt(mass*n/base)*experiment['force_x']*identity)
        if n+1 < modes:
            blocks[n][n+1] = -np.sqrt(mass*base*(n+1))*derivative
        if n >= 2:
            blocks[n][n-2] = friction*np.sqrt(n*(n-1))*diags(temperature/base-1)
    generator = bmat(blocks,format='csc')
    normalized = generator.tolil()
    normalized[0,:] = 0
    normalized[0,:points] = 2*np.pi/points
    rhs = np.zeros(points*modes); rhs[0] = 1
    zeroth = spsolve(normalized.tocsc(),rhs)
    gamma_y = experiment['drag_ratio']*friction
    # Constant transverse force has conditional mean velocity Fy/gamma_y.
    # This is also the exact solution of (Lx-gamma_y)*c1=-Fy*sqrt(m/T0)*c0.
    first = experiment['force_y']*np.sqrt(mass/base)*zeroth/gamma_y
    source = (experiment['force_y']*np.sqrt(2*mass/base)*first
              +np.tile(np.sqrt(2)*gamma_y*(temperature/base-1),modes)*zeroth)
    second = spsolve(generator-2*gamma_y*eye(points*modes,format='csc'),-source)
    return x, temperature, [c.reshape(modes,points) for c in [zeroth,first,second]]


def finite_mass_entropy(experiment, friction, mass, points=79, modes=32):
    _, temperature, (c0,c1,c2) = kinetic_state(experiment,friction,mass,points,modes)
    base = experiment['temperature']
    gamma_y = friction*experiment['drag_ratio']
    # The local Stratonovich noise work contributes -gamma_i*rho/m for
    # each velocity component. Evaluate the underdamped bath heat directly.
    xheat = friction*np.mean(base*(c0[0]+np.sqrt(2)*c0[2])/temperature-c0[0])
    yheat = gamma_y*np.mean(base*(c0[0]+np.sqrt(2)*c2[0])/temperature-c0[0])
    return float((xheat+yheat)*2*np.pi/mass)


@lru_cache(None)
def limiting_rate(force_x,force_y,drag_ratio,temperature,contrast,wavenumber,friction,points,modes,epsilon):
    if contrast == 0:
        return (force_x**2+force_y**2/drag_ratio)/(friction*temperature)
    e = dict(force_x=force_x,force_y=force_y,drag_ratio=drag_ratio,temperature=temperature,
             contrast=contrast,wavenumber=wavenumber)
    mass = epsilon*(friction*min(1.,drag_ratio))**2/(temperature*wavenumber**2)
    rates = [finite_mass_entropy(e,friction,mass/factor,points,modes) for factor in [1,2,4]]
    return rates[0]/3-2*rates[1]+8*rates[2]/3


def predict(experiments, friction, points=79, modes=32, epsilon=.004):
    return np.array([limiting_rate(e['force_x'],e['force_y'],e['drag_ratio'],e['temperature'],
                    e['contrast'],e['wavenumber'],friction,points,modes,epsilon) for e in experiments])
