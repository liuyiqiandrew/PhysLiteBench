"""Oldroyd-B sheet: exact harmonic tensors versus local boundary equations.

Prototype only. Density is one; viscosity is kinematic in mm^2/s. The sheet is
y=a*cos(k*x-w*t), with material x fixed. Velocities are divided by a and mean
pumping by a^2. No task files or model evaluations are generated.
"""
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad_vec, solve_bvp
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parent
BETA = .25
RELAXATION = 1.


def complex_viscosity(nu, frequency, beta=BETA, relaxation=RELAXATION):
    return nu*(beta+(1-beta)/(1-1j*frequency*relaxation))


def derivatives(nu, k, frequency, y, beta=BETA, relaxation=RELAXATION):
    """Streamfunction and first four derivatives, stable at creeping flow."""
    viscosity = complex_viscosity(nu, frequency, beta, relaxation)
    s = np.sqrt(k*k-1j*frequency/viscosity)
    delta = -1j*frequency/(viscosity*(s+k))
    y = np.asarray(y)
    decay = np.exp(-k*y)
    small = np.abs(delta*y) < .25
    divided_exponential = np.empty_like(y, dtype=complex)
    divided_exponential[small] = decay[small]*np.expm1(-delta*y[small])/delta
    divided_exponential[~small] = (np.exp(-s*y[~small])-decay[~small])/delta
    values = []
    for n in range(5):
        divided_power = (-1)**n*sum(s**(n-1-j)*k**j for j in range(n))
        values.append(frequency/k*(decay*((-k)**n-k*divided_power)
                                   -k*(-s)**n*divided_exponential))
    return np.asarray(values)


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


def velocity(nu, k, frequency, height, beta=BETA, relaxation=RELAXATION):
    z = derivatives(nu, k, frequency, height, beta, relaxation)
    return np.array([z[1], -1j*k*z[0]])


def integrated_prediction(nu, k, frequency, beta=BETA, relaxation=RELAXATION,
                          tolerance=1e-10):
    def integrand(x):
        psi = derivatives(nu, k, frequency, x/k, beta, relaxation)
        reynolds, polymer, _, _ = fluxes(psi, nu, k, frequency, beta, relaxation)
        return np.array([reynolds, polymer])/k
    integral, error = quad_vec(integrand, 0., np.inf, epsabs=tolerance, epsrel=tolerance)
    boundary = -.5*float(derivatives(nu, k, frequency, 0., beta, relaxation)[2].real)
    source = boundary+integral[0]/nu
    physical = source-integral[1]/nu
    return dict(source=float(source), physical=float(physical),
                boundary=boundary, reynolds=float(integral[0]/nu),
                polymer=float(-integral[1]/nu), quadrature_error=float(error))


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
    y = np.linspace(0, height, 1300)
    r, n, _, _ = fluxes(local(y), nu, k, frequency, beta, relaxation)
    balance = nu*mean.sol(y)[1]+n-r
    exact = velocity(nu, k, frequency, .7/k, beta, relaxation)
    z = local(.7/k)
    observed = np.array([z[1], -1j*k*z[0]])
    return dict(physical=float(mean.sol(height)[0]), source=float(mean.sol(height)[2]),
                linear_relative_error=float(np.linalg.norm(observed-exact)/np.linalg.norm(exact)),
                momentum_balance_max=float(np.max(np.abs(balance))),
                linear_nodes=len(flow.x), mean_nodes=len(mean.x))


def main():
    start = time.perf_counter()
    rng = np.random.default_rng(284291)
    controls = list(itertools.product([.7, 1.4], [.7, 1.3], [.5, 2., 8.]))
    controls += [tuple(rng.uniform([.7, .7, .5], [1.4, 1.3, 8.])) for _ in range(12)]
    cases = []
    for nu, k, w in controls:
        answer = integrated_prediction(nu, k, w)
        reference = boundary_reference(nu, k, w)
        assert all(np.isfinite(value) for value in answer.values())
        assert all(np.isfinite(value) for value in reference.values())
        scale = max(abs(answer['physical']), 1e-14)
        cases.append(dict(viscosity=nu, wave_number=k, frequency=w, **answer,
            reference=reference, relative_reference_error=abs(reference['physical']-answer['physical'])/scale,
            relative_source_error=abs(answer['source']-answer['physical'])/scale,
            scaled_physical=answer['physical']/(k*w)))
    refinement = []
    for nu, k, w in [(.7, .7, 8.), (1.4, 1.3, .5), (1.05, 1., 2.)]:
        first = boundary_reference(nu, k, w)
        second = boundary_reference(nu, k, w, depth=30., tolerance=2e-9)
        quadrature = integrated_prediction(nu, k, w, tolerance=1e-12)
        refinement.append(dict(viscosity=nu, wave_number=k, frequency=w,
            bvp_change=abs(first['physical']/second['physical']-1),
            refined_reference_error=abs(second['physical']/quadrature['physical']-1)))
    limits = []
    for nu in [10., 100., 1000., 10000.]:
        z = integrated_prediction(nu, 1., 2.)
        limits.append(dict(viscosity=nu, source_scaled=z['source'], physical_scaled=z['physical'],
                           ratio=z['physical']/z['source']))
    newtonian = []
    for beta, relaxation in [(1., 1.), (.25, 0.)]:
        for nu, k, w in [(1.05, 1., 2.), (.7, .7, 8.)]:
            z = integrated_prediction(nu, k, w, beta, relaxation)
            newtonian.append(dict(beta=beta, relaxation=relaxation, viscosity=nu,
                                   wave_number=k, frequency=w, **z))
    window = []
    for nu, k, w in itertools.product([.7, 1.05, 1.4], [.7, 1., 1.3], [4., 6., 8.]):
        z = integrated_prediction(nu, k, w)
        window.append(dict(viscosity=nu, wave_number=k, frequency=w, **z,
                           relative_source_error=abs(z['source']/z['physical']-1)))
    extra_checks = []
    for nu, k, w in [(.7, .7, 8.), (1.4, 1.3, .5), (1.05, 1., 2.)]:
        z = integrated_prediction(nu, k, w)
        reverse = integrated_prediction(nu, k, -w)
        y, step = .6/k, 1e-5/k
        def values_at(h):
            return fluxes(derivatives(nu, k, w, h), nu, k, w)
        numerical_gradient = (values_at(y+step)[1]-values_at(y-step)[1])/(2*step)
        viscosity = complex_viscosity(nu, w)
        wall = derivatives(nu, k, w, 0.)
        pressure = (1j*w*wall[1]+viscosity*(wall[3]-k*k*wall[1]))/(1j*k)
        normal_stress = -pressure-2j*k*viscosity*wall[1]
        work = -.5*np.real((-1j*k*wall[0])*normal_stress.conjugate())
        def dissipation(x):
            g = tensors(derivatives(nu, k, w, x/k), k, w, RELAXATION)[4]
            return .25*viscosity.real*np.sum(np.abs(g+g.T)**2)/k
        loss, _ = quad_vec(dissipation, 0., np.inf, epsabs=1e-10, epsrel=1e-10)
        extra_checks.append(dict(viscosity=nu, wave_number=k, frequency=w,
            wave_reversal_error=abs(reverse['physical']/z['physical']+1),
            polymer_gradient_error=float(abs(numerical_gradient-values_at(y)[3])),
            harmonic_wall_power=float(work), volume_loss=float(loss),
            relative_energy_error=float(abs(work/loss-1))))
    settings = list(itertools.product([.7, 1.3], [.5, 2., 8.], [.5, .6]))
    def calibration(nu):
        z = np.array([velocity(nu, k, w, h) for k, w, h in settings]).ravel()
        return np.r_[z.real, z.imag]
    def fit(data):
        objective = lambda x: float(np.sum((calibration(x)-data)**2))
        optimum = minimize_scalar(objective, bounds=(.7, 1.4), method='bounded',
                                  options={'xatol':1e-12})
        return min([.7, optimum.x, 1.4], key=objective)
    recovery = []
    information = []
    for nu in np.linspace(.7, 1.4, 29):
        found = fit(calibration(nu))
        recovery.append(dict(true=float(nu), fit=float(found), relative_error=abs(found/nu-1)))
        derivative = (calibration(nu+1e-5)-calibration(nu-1e-5))/2e-5
        information.append(float(np.linalg.norm(derivative)/.002))
    noisy_fits = [fit(calibration(1.05)+rng.normal(0, .002, 48)) for _ in range(64)]
    identity_errors = []
    for nu, k, w in controls:
        s = np.sqrt(k*k-1j*w/complex_viscosity(nu, w))
        for h in [.5, .6]:
            u, v = velocity(nu, k, w, h)
            # Sheet phase fixes U+iV=omega*exp(-s*h).
            identity_errors.append(abs((u+1j*v)/(w*np.exp(-s*h))-1))
    report = dict(status='prototype_only_no_task_or_model_evaluation',
        structured_cases=12, random_cases=12, beta=BETA, relaxation=RELAXATION,
        cases=cases, refinement=refinement, creeping_limit=limits,
        creeping_expected_ratio=.4, newtonian_limits=newtonian,
        candidate_frequency_window=window, extra_checks=extra_checks,
        parameter_recovery=recovery, noise_sigma=.002, noise_draws=64,
        noisy_fits=noisy_fits,
        max_relative_reference_error=max(x['relative_reference_error'] for x in cases),
        min_relative_source_error=min(x['relative_source_error'] for x in cases),
        min_absolute_pumping=min(abs(x['physical']) for x in cases),
        min_scaled_absolute_pumping=min(abs(x['scaled_physical']) for x in cases),
        min_window_relative_source_error=min(x['relative_source_error'] for x in window),
        min_window_absolute_pumping=min(abs(x['physical']) for x in window),
        max_refinement=max(x['bvp_change'] for x in refinement),
        max_parameter_error=max(x['relative_error'] for x in recovery),
        worst_local_standard_error=1/min(information),
        max_noisy_relative_error=max(abs(x/1.05-1) for x in noisy_fits),
        max_velocity_identity_error=float(max(identity_errors)),
        phase_bound_radians=.1*np.sqrt(1.3**2+8/(.25*.7)),
        seconds=time.perf_counter()-start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in
                     ['cases', 'parameter_recovery', 'noisy_fits', 'newtonian_limits',
                      'candidate_frequency_window']}, indent=2))
    assert report['max_relative_reference_error'] < 2e-5
    assert report['max_refinement'] < 2e-5
    assert report['max_parameter_error'] < 1e-6
    assert report['max_velocity_identity_error'] < 1e-12
    assert abs(limits[-1]['ratio']/.4-1) < .001
    assert max(x['relative_energy_error'] for x in extra_checks) < 1e-8
    assert max(x['wave_reversal_error'] for x in extra_checks) < 1e-12
    assert max(x['polymer_gradient_error'] for x in extra_checks) < 1e-6
    assert max(abs(x['polymer']) for x in newtonian) == 0


if __name__ == '__main__':
    main()
