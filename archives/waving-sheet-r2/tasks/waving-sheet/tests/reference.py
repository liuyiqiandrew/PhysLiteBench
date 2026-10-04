"""Harmonic-vorticity and mean polymer-stress/momentum boundary equations."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_bvp

TRUE_PARAMETER = 1.05
BETA = .25
RELAXATION = 1.

def complex_viscosity(nu, frequency, beta=BETA, relaxation=RELAXATION):
    return nu*(beta+(1-beta)/(1-1j*frequency*relaxation))


def tensors(psi, k, frequency, relaxation):
    """Actual harmonic velocity, gradient and Hookean conformation tensors."""
    u, v = psi[1], -1j*k*psi[0]
    uy, vy = psi[2], -1j*k*psi[1]
    g = np.array([[1j*k*u, uy], [1j*k*v, vy]])
    gy = np.array([[1j*k*uy, psi[3]], [1j*k*vy, -1j*k*psi[2]]])
    gyy = np.array([[1j*k*psi[3], psi[4]],
                    [k*k*psi[2], -1j*k*psi[3]]])
    factor = relaxation/(1-1j*frequency*relaxation)
    c = factor*(g+g.swapaxes(0, 1))
    cy = factor*(gy+gy.swapaxes(0, 1))
    cyy = factor*(gyy+gyy.swapaxes(0, 1))
    return u, v, uy, vy, g, gy, c, cy, cyy

def product(a, b):
    return np.einsum('ij...,jk...->ik...', a, b)

def fluxes(psi, nu, k, frequency, beta=BETA, relaxation=RELAXATION):
    u, v, uy, vy, g, gy, c, cy, cyy = tensors(psi, k, frequency, relaxation)
    eta_p = (1-beta)*nu
    polymer = eta_p*.5*np.real(
        product(g, c.conj())+product(c, g.conj().swapaxes(0, 1))
        -u*np.conj(1j*k*c)-v*cy.conj())
    polymer_y = eta_p*.5*np.real(
        product(gy, c.conj())+product(g, cy.conj())
        +product(cy, g.conj().swapaxes(0, 1))
        +product(c, gy.conj().swapaxes(0, 1))
        -uy*np.conj(1j*k*c)-u*np.conj(1j*k*cy)
        -vy*cy.conj()-v*cyy.conj())
    reynolds = .5*np.real(u*v.conj())
    convection = .5*np.real(u*np.conj(1j*k*u)+v*uy.conj())
    return reynolds, polymer[0, 1], convection, polymer_y[0, 1]

@lru_cache(maxsize=1024)
def boundary_reference(nu, k, frequency, beta=BETA, relaxation=RELAXATION,
                       depth=24., tolerance=2e-8):
    """Independent harmonic-vorticity and forced mean-momentum BVPs.

    No exact harmonic modes or integrated-flux prediction are used. Polymer
    forcing is differentiated locally from the first-order conformation.
    """
    viscosity = complex_viscosity(nu, frequency, beta, relaxation)
    height = depth/min(k, np.sqrt(k*k-1j*frequency/viscosity).real)
    grid = np.linspace(0, height, 160)
    def fourth(z):
        return ((2*k*k-1j*frequency/viscosity)*z[2]
                -(k**4-1j*frequency*k*k/viscosity)*z[0])
    def linear_rhs(y, state):
        z = state[:4]+1j*state[4:]
        rhs = np.vstack((z[1:], fourth(z)))
        return np.vstack((rhs.real, rhs.imag))
    def linear_bc(left, right):
        return np.array([left[0]-frequency/k, left[1], right[0], right[1],
                         left[4], left[5], right[4], right[5]])
    flow = solve_bvp(linear_rhs, linear_bc, grid, np.zeros((8, len(grid))),
                     tol=tolerance, max_nodes=18000)
    assert flow.success, flow.message
    def local(y):
        z = flow.sol(y)
        z = z[:4]+1j*z[4:]
        return np.concatenate((z, np.asarray(fourth(z))[None, ...]), axis=0)
    def mean_rhs(y, state):
        z = local(y)
        u, v, uy, vy, g, gy, c, cy, cyy = tensors(z, k, frequency, relaxation)
        # Componentwise conformation transport, independently assembled.
        nc_y = np.zeros_like(u, dtype=complex)
        for j in range(2):
            nc_y += (gy[0, j]*c[1, j].conj()+g[0, j]*cy[1, j].conj()
                     +cy[0, j]*g[1, j].conj()+c[0, j]*gy[1, j].conj())
        nc_y -= (uy*np.conj(1j*k*c[0, 1])+u*np.conj(1j*k*cy[0, 1])
                 +vy*cy[0, 1].conj()+v*cyy[0, 1].conj())
        polymer_y = (1-beta)*nu*.5*nc_y.real
        convection = .5*np.real(u*np.conj(1j*k*u)+v*uy.conj())
        return np.vstack((state[1], (convection-polymer_y)/nu,
                          state[3], convection/nu))
    boundary = -.5*local(0.)[2].real
    def mean_bc(left, right):
        return np.array([left[0]-boundary, right[1], left[2]-boundary, right[3]])
    mean = solve_bvp(mean_rhs, mean_bc, grid, np.zeros((4, len(grid))),
                    tol=tolerance, max_nodes=18000)
    assert mean.success, mean.message
    def velocity(y):
        z = local(y)
        return np.array([z[1],-1j*k*z[0]])
    def nonlinear_stress(y):
        return fluxes(local(y),nu,k,frequency,beta,relaxation)[1]
    return dict(physical=float(mean.sol(height)[0]), source=float(mean.sol(height)[2]),
        velocity=velocity, linear_state=flow.sol, mean_state=mean.sol,
        nonlinear_stress=nonlinear_stress, height=height,
        linear_nodes=len(flow.x), mean_nodes=len(mean.x))

def predict(experiments,viscosity):
    output = []
    for e in experiments:
        result = boundary_reference(viscosity,e['wave_number'],e['frequency'])
        if e['observable']=='pumping':
            output.append(result['physical'])
        else:
            u,v = result['velocity'](e['height'])
            output.append({'u_real':u.real,'u_imag':u.imag,
                           'v_real':v.real,'v_imag':v.imag}[e['observable']])
    return np.array(output,dtype=float)


def experiment(wave_number,frequency,observable='pumping',height=None):
    result = dict(wave_number=wave_number,frequency=frequency,observable=observable)
    if height is not None:
        result['height'] = height
    return result


def hidden_inputs():
    return {
        'frequency_scan':[experiment(1.,w) for w in [4.,6.,8.]],
        'wavelength_scan':[experiment(k,6.) for k in [.75,1.,1.25]],
        'combined_controls':[experiment(.8,4.5),experiment(1.2,6.5),experiment(.9,7.5)],
        'linear_anchors':[experiment(.85,1.2,'u_real',.3),experiment(1.15,5.5,'u_imag',.8),
                          experiment(1.25,7.,'v_real',1.3),experiment(.75,2.5,'v_imag',.6)],
    }
