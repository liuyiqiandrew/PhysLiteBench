"""Small-amplitude pumping: analytic modes versus independent boundary ODEs."""
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parent


def modes(viscosity, wave_number, frequency):
    k, w = wave_number, frequency
    s = np.sqrt(k*k-1j*w/viscosity)
    a = w*s/(k*(s-k))
    b = -w/(s-k)
    return np.array([k, s]), np.array([a, b])


def velocity(viscosity, wave_number, frequency, height):
    p, c = modes(viscosity, wave_number, frequency)
    e = np.exp(-p*height)
    return np.array([np.sum(-p*c*e), np.sum(-1j*wave_number*c*e)])


def pumping(viscosity, wave_number, frequency):
    p, c = modes(viscosity, wave_number, frequency)
    uc, vc = -p*c, -1j*wave_number*c
    boundary = -.5*np.real(np.sum(p*p*c))
    flux_integral = .5*np.real(np.sum(
        uc[:, None]*vc.conj()[None, :]/(p[:, None]+p.conj()[None, :])))
    return float(boundary), float(boundary+flux_integral/viscosity)


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
        'velocity': lambda y: np.array([
            flow.sol(y)[1]+1j*flow.sol(y)[5],
            -1j*k*(flow.sol(y)[0]+1j*flow.sol(y)[4])]),
    }


def main():
    started = time.perf_counter()
    cases = []
    for nu, k, w in itertools.product([.7, 1.4], [.7, 1.3], [.5, 2., 8.]):
        source, physical = pumping(nu, k, w)
        ref = boundary_reference(nu, k, w)
        linear = velocity(nu, k, w, .7/k)
        cases.append(dict(viscosity=nu, wave_number=k, frequency=w,
            source=source, physical=physical, reference=ref['physical'],
            relative_error=abs(ref['physical']/physical-1),
            source_error=abs(source/physical-1),
            linear_error=float(np.linalg.norm(ref['velocity'](.7/k)-linear)/np.linalg.norm(linear))))
    refinement = []
    for nu, k, w in [(.7,.7,8.),(1.4,1.3,.5),(1.05,1.,4.)]:
        a = boundary_reference(nu,k,w)
        b = boundary_reference(nu,k,w,depth=26.,tolerance=1e-9)
        refinement.append(abs(a['physical']/b['physical']-1))
    settings = list(itertools.product([.7,1.,1.3],[.5,2.,8.],[.4,1.]))
    def calibration(nu):
        z = np.array([velocity(nu,k,w,h) for k,w,h in settings]).ravel()
        return np.r_[z.real,z.imag]
    recovery = []
    for nu in np.linspace(.7,1.4,25):
        truth = calibration(nu)
        objective = lambda x: float(np.sum((calibration(x)-truth)**2))
        fit = minimize_scalar(objective,bounds=(.7,1.4),method='bounded',
                              options={'xatol':1e-12})
        found = min([fit.x,.7,1.4],key=objective)
        recovery.append(dict(true=float(nu),fit=float(found),error=abs(found/nu-1)))
    limits = []
    for w in [1e-2,1e-3,1e-4]:
        source, physical = pumping(1.,1.,w)
        limits.append(dict(frequency=w,source_scaled=source/w,physical_scaled=physical/w))
    assert max(x['relative_error'] for x in cases) < 1e-6
    assert max(x['linear_error'] for x in cases) < 1e-7
    assert max(refinement) < 1e-6
    assert max(x['error'] for x in recovery) < 1e-6
    report = dict(status='prototype_only_no_task_or_model_runs',cases=cases,
        max_reference_error=max(x['relative_error'] for x in cases),
        max_linear_error=max(x['linear_error'] for x in cases),
        max_refinement_error=max(refinement),parameter_recovery=recovery,
        max_parameter_error=max(x['error'] for x in recovery),
        low_frequency_limit=limits,seconds=time.perf_counter()-started,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['cases','parameter_recovery']},indent=2))


if __name__ == '__main__':
    main()
