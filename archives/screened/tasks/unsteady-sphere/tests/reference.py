"""Exterior axisymmetric flow modes, no-slip solve, and surface traction integral."""
import numpy as np
from numpy.polynomial.legendre import leggauss

PARAMETER = 'viscosity'
TRUE_PARAMETER = .008
RADIUS = .001
FLUID_DENSITY = 1000.
PARTICLE_DENSITY = 2300.


def calibration_inputs():
    return [dict(force=float(f),frequency=0.,component='cosine') for f in np.linspace(-1e-9,1e-9,100)]


def hidden_inputs():
    return {
        'in_phase': [dict(force=.75e-9,frequency=float(w),component='cosine') for w in np.geomspace(5.,120.,32)],
        'quadrature': [dict(force=.75e-9,frequency=float(w),component='sine') for w in np.geomspace(5.,120.,32)],
        'mixed_response': [dict(force=(-1.)**j*1e-9,frequency=float(w),component='cosine' if j%2 else 'sine')
                           for j,w in enumerate(np.linspace(8.,115.,32))]}


def fluid_force_per_velocity(omega,viscosity):
    if omega==0:
        return -6*np.pi*viscosity*RADIUS
    # psi=V*a^2*h(r/a)*sin(theta)^2, h=C/q+D*exp[-s(q-1)]*(1+1/(s*q)).
    # The two retained modes are the decaying pressure-potential and vorticity modes.
    s = RADIUS*np.sqrt(-1j*omega*FLUID_DENSITY/viscosity)
    boundary = np.array([[1.,1+1/s],[-1.,-s-1-1/s]],dtype=complex)
    c,d = np.linalg.solve(boundary,np.array([.5,1.]))
    h,h_prime = .5,1.
    h_second = 2*c+d*(s*s+2)*(1+1/s)
    # At the surface, p=-i*omega*rho*V*a*C*cos(theta).
    radial_stress = 1j*omega*FLUID_DENSITY*RADIUS*c+4*viscosity/RADIUS*(h_prime-2*h)
    shear_stress = viscosity/RADIUS*(-2*h-h_second+2*h_prime)
    cosine,weights = leggauss(20)
    traction_z = radial_stress*cosine**2-shear_stress*(1-cosine**2)
    return 2*np.pi*RADIUS**2*np.dot(weights,traction_z)


def predict(experiments,viscosity):
    mass = PARTICLE_DENSITY*4*np.pi*RADIUS**3/3
    out = []
    for e in experiments:
        force = fluid_force_per_velocity(e['frequency'],viscosity)
        velocity = e['force']/(-1j*e['frequency']*mass-force)
        out.append(velocity.real if e['component']=='cosine' else velocity.imag)
    return np.array(out)
