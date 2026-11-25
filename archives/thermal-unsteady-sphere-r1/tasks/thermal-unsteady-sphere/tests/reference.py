"""Exterior fluid traction and local thermal-stress propagation."""
from functools import lru_cache
import numpy as np
from scipy.integrate import quad
KB = 1.380649e-23
RHO = 1000.0
RHO_PARTICLE = 2200.0
FORCE = 1e-13
TRUE_PARAMETER = 1.07

def exterior_field(s):
    """Solve the two decaying radial modes for unit boundary translation."""
    if s == 0:
        def field(x):
            return (1.5/x-0.5/x**3, -1.5/x**2+1.5/x**4,
                    3/x**3-6/x**5)
        return field, None
    a, b = np.linalg.solve(
        np.array([[1, s+1], [-3, -(s*s+3*s+3)]], complex), [1, 0])

    def field(x):
        tail = b*np.exp(-s*(x-1))
        f = (a+tail*(s*x+1))/x**3
        fp = (-3*a-tail*(s*s*x*x+3*s*x+3))/x**4
        fpp = (12*a+tail*(s**3*x**3+5*s*s*x*x+12*s*x+12))/x**5
        return f, fp, fpp
    return field, a


@lru_cache(maxsize=4096)
def fluid_reference(radius, omega, viscosity, temperature, tolerance=2e-10,
                    angular_order=16):
    """Integrate local stochastic-stress power and surface traction.

    This function does not call the closed impedance or noise formula.
    For unit sphere speed, f defines ur=f*cos(theta), utheta=-g*sin(theta),
    with g=f+x*f'/2. A white isotropic bulk stress has covariance
    2*kBT*eta*(delta_ik delta_jl+delta_il delta_jk) delta(x-x') delta(t-t').
    Contracting it with the auxiliary strain gives 4*kBT*eta*integral D:D.
    """
    s = radius*np.sqrt(-1j*omega*RHO/viscosity)
    field, a = exterior_field(s)
    z, wz = np.polynomial.legendre.leggauss(angular_order)
    f, fp, fpp = field(1.)
    # Pressure coefficient of cos(theta), obtained from the irrotational mode.
    pressure = 1.5*viscosity/radius if omega == 0 else (
        -0.5j*omega*RHO*radius*a)
    sigma_rr = (-pressure+2*viscosity*fp/radius)*z
    sigma_rtheta = -viscosity/radius*(fp+fpp/2)*np.sqrt(1-z*z)
    traction = 2*np.pi*radius**2*np.sum(
        wz*(sigma_rr*z-sigma_rtheta*np.sqrt(1-z*z)))
    impedance = -traction

    def strain_density(x):
        _, dp, ddp = field(x)
        drr = dp*z
        dtt = -dp*z/2
        drt = -(dp+x*ddp/2)*np.sqrt(1-z*z)/2
        angular = 2*np.pi*np.sum(wz*(abs(drr)**2+2*abs(dtt)**2+2*abs(drt)**2))
        return float(x*x*angular)

    integral, error = quad(strain_density, 1, np.inf, epsabs=tolerance,
                           epsrel=tolerance, limit=240)
    random_force_psd = 4*KB*temperature*viscosity*radius*integral
    mass = (4*np.pi/3)*RHO_PARTICLE*radius**3
    mu = 1/(impedance-1j*omega*mass)
    return {
        'spectrum': random_force_psd*abs(mu)**2,
        'impedance': impedance,
        'mobility': mu,
        'dissipation_integral': 2*viscosity*radius*integral,
        'boundary_error': float(max(abs(f-1), abs(fp))),
        'quadrature_relative_bound': float(error/integral),
    }



def predict(experiments, viscosity):
    values = []
    for e in experiments:
        response = fluid_reference(e['radius']*1e-6, e['angular_frequency']*1e6,
                                   viscosity*1e-3, e['temperature'])
        if e['observable'] == 'in_phase':
            values.append(FORCE*response['mobility'].real*1e6)
        elif e['observable'] == 'quadrature':
            values.append(FORCE*response['mobility'].imag*1e6)
        else:
            values.append(response['spectrum']*1e12)
    return np.array(values)


def experiment(radius, frequency, temperature, observable='velocity_psd'):
    return {'radius':radius, 'angular_frequency':frequency,
            'temperature':temperature, 'observable':observable}


def hidden_inputs():
    return {
        'frequency_scan':[experiment(1.,w,300.) for w in [.5,2.,8.]],
        'radius_scan':[experiment(a,3.,300.) for a in [.6,1.,1.4]],
        'temperature_controls':[experiment(.8,1.5,280.),experiment(1.2,4.,320.),
                                experiment(1.5,6.,305.)],
        'response_anchors':[experiment(.9,0.,290.),experiment(1.3,.7,310.,'in_phase'),
                            experiment(.65,3.5,285.,'quadrature')],
    }
