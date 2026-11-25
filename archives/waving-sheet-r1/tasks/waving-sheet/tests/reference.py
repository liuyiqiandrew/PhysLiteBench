"""Boundary-value reference for unsteady and time-averaged liquid momentum."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_bvp

TRUE_PARAMETER = 1.05

@lru_cache(maxsize=1024)
def boundary_reference(viscosity, wave_number, frequency, depth=20., tolerance=1e-8):
    """Solve linear vorticity and mean Navier-Stokes momentum as boundary ODEs."""
    k, w, nu = wave_number, frequency, viscosity
    height = depth/k
    grid = np.linspace(0, height, 160)
    def linear_rhs(y, state):
        psi = state[:4]+1j*state[4:]
        fourth = (2*k*k-1j*w/nu)*psi[2]-(k**4-1j*w*k*k/nu)*psi[0]
        derivative = np.vstack((psi[1:], fourth))
        return np.vstack((derivative.real, derivative.imag))
    def linear_bc(left, right):
        return np.array([left[0]-w/k, left[1], right[0], right[1],
                         left[4], left[5], right[4], right[5]])
    flow = solve_bvp(linear_rhs, linear_bc, grid, np.zeros((8, len(grid))),
                     tol=tolerance, max_nodes=12000)
    assert flow.success, flow.message
    def mean_rhs(y, state):
        z = flow.sol(y)
        psi = z[:4]+1j*z[4:]
        u, v = psi[1], -1j*k*psi[0]
        # Average u*d_x u + v*d_y u over a period, directly from the ODE field.
        convection = .5*np.real(u*np.conj(1j*k*u)+v*np.conj(psi[2]))
        return np.vstack((state[1], convection/nu))
    boundary = -.5*flow.sol(0)[2]
    def mean_bc(left, right):
        return np.array([left[0]-boundary, right[1]])
    mean = solve_bvp(mean_rhs, mean_bc, grid, np.zeros((2, len(grid))),
                    tol=tolerance, max_nodes=12000)
    assert mean.success, mean.message
    return {
        'source': float(boundary), 'physical': float(mean.sol(height)[0]),
        'linear_nodes': len(flow.x), 'mean_nodes': len(mean.x),
        'linear_state': flow.sol, 'mean_state': mean.sol,
        'velocity': lambda y: np.array([
            flow.sol(y)[1]+1j*flow.sol(y)[5],
            -1j*k*(flow.sol(y)[0]+1j*flow.sol(y)[4])]),
    }



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
        'frequency_scan':[experiment(1.,w) for w in [1.,3.,7.]],
        'wavelength_scan':[experiment(k,4.) for k in [.75,1.,1.25]],
        'combined_controls':[experiment(.8,2.),experiment(1.2,5.),experiment(.9,8.)],
        'linear_anchors':[experiment(.85,1.2,'u_real',.3),experiment(1.15,5.5,'u_imag',.8),
                          experiment(1.25,7.,'v_real',1.3),experiment(.75,2.5,'v_imag',.6)],
    }
