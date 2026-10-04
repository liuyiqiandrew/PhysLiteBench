"""Translating-sphere thermal spectrum: closed impedance versus fluid stress."""
from pathlib import Path
import hashlib
import itertools
import json
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

KB = 1.380649e-23
RHO = 1000.0
RHO_PARTICLE = 2200.0
FORCE = 1e-13
ETA_BOUNDS = (0.0007, 0.0014)


def mobility(radius, omega, viscosity):
    mass = (4*np.pi/3)*RHO_PARTICLE*radius**3
    s = radius*np.sqrt(-1j*omega*RHO/viscosity)
    impedance = 6*np.pi*viscosity*radius*(1+s+s*s/9)
    return 1/(impedance-1j*omega*mass)


def spectrum(radius, omega, viscosity, temperature, white=False):
    mu = mobility(radius, omega, viscosity)
    dissipation = 6*np.pi*viscosity*radius
    if not white:
        dissipation *= 1+radius*np.sqrt(abs(omega)*RHO/(2*viscosity))
    return 2*KB*temperature*dissipation*abs(mu)**2


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


def calibration(viscosity):
    values = []
    for radius, omega in itertools.product([0.75e-6, 1.25e-6],
                                            [0., 0.2e6, 0.8e6, 2e6]):
        velocity = FORCE*mobility(radius, omega, viscosity)*1e6
        values.append(velocity.real)
        if omega:
            values.append(velocity.imag)
    return np.array(values)


def main():
    start = time.perf_counter()
    rng = np.random.default_rng(261013)
    cases = list(itertools.product([.5e-6, 1.5e-6], [0., .2e6, 8e6],
                                   ETA_BOUNDS, [280., 320.]))
    cases += [(rng.uniform(.5e-6, 1.5e-6), 10**rng.uniform(np.log10(.2e6),
              np.log10(8e6)), rng.uniform(*ETA_BOUNDS), rng.uniform(280, 320))
              for _ in range(32)]
    records = []
    for radius, omega, eta, temperature in cases:
        ref = fluid_reference(radius, omega, eta, temperature)
        correct = spectrum(radius, omega, eta, temperature)
        source = spectrum(radius, omega, eta, temperature, True)
        exact_mu = mobility(radius, omega, eta)
        mass = (4*np.pi/3)*RHO_PARTICLE*radius**3
        exact_z = 1/exact_mu+1j*omega*mass
        records.append({
            'radius_m': radius, 'omega_rad_per_s': omega,
            'viscosity_Pa_s': eta, 'temperature_K': temperature,
            'correct_psd_m2_per_s': correct,
            'white_source_psd_m2_per_s': source,
            'reference_relative_error': abs(correct-ref['spectrum'])/correct,
            'source_relative_error': abs(correct-source)/correct,
            'traction_relative_error': abs(ref['impedance']-exact_z)/abs(exact_z),
            'fluid_dissipation_identity_relative_error': abs(
                ref['dissipation_integral']-ref['impedance'].real)/ref['impedance'].real,
            'boundary_error': ref['boundary_error'],
            'quadrature_relative_bound': ref['quadrature_relative_bound'],
        })
    refinements = []
    for case in [(1e-6, 1e6, .00107, 300.), (1.5e-6,8e6,.0007,320.),
                 (.5e-6,.2e6,.0014,280.)]:
        coarse = fluid_reference(*case)
        fine = fluid_reference(*case, tolerance=1e-12, angular_order=32)
        refinements.append(abs(coarse['spectrum']-fine['spectrum'])/fine['spectrum'])

    true_eta = .00107
    exact_cal = calibration(true_eta)
    fixed_sigma = .002  # micrometres/second, an absolute velocity instrument error.
    eta_grid = np.linspace(*ETA_BOUNDS, 501)
    objectives = np.array([np.sum(((calibration(eta)-exact_cal)/fixed_sigma)**2)
                           for eta in eta_grid])
    noiseless = minimize_scalar(lambda eta: np.sum((calibration(eta)-exact_cal)**2),
        bounds=ETA_BOUNDS, method='bounded', options={'xatol':1e-15}).x
    fitted = []
    for _ in range(32):
        measured = np.tile(exact_cal, 8)+fixed_sigma*rng.normal(size=8*len(exact_cal))
        fit = minimize_scalar(lambda eta: np.sum(((np.tile(calibration(eta),8)-measured)
                    /fixed_sigma)**2), bounds=ETA_BOUNDS, method='bounded',
                    options={'xatol':1e-14})
        fitted.append(float(fit.x))
    endpoint_recovery = []
    for true in ETA_BOUNDS:
        y = calibration(true)
        result = minimize_scalar(lambda eta: np.sum((calibration(eta)-y)**2),
            bounds=ETA_BOUNDS, method='bounded', options={'xatol':1e-15})
        # A bounded minimum may lie at a physical endpoint.
        estimate = min([result.x, *ETA_BOUNDS], key=lambda eta: np.sum((calibration(eta)-y)**2))
        endpoint_recovery.append(abs(estimate/true-1))

    # Further checks of the finite-frequency effective continuum, not a microscopic limit.
    dc_error = max(abs(spectrum(a,0,eta,300)/spectrum(a,0,eta,300,True)-1)
                   for a,eta in itertools.product([.5e-6,1.5e-6],ETA_BOUNDS))
    even_error = max(abs(spectrum(a,-w,eta,t)/spectrum(a,w,eta,t)-1)
                     for a,w,eta,t in cases)
    conjugacy = max(abs(mobility(a,-w,eta)-mobility(a,w,eta).conjugate())
                   /abs(mobility(a,w,eta)) for a,w,eta,_ in cases)
    finite_cases = [r for r in records if r['omega_rad_per_s']>0]
    report = {
        'status':'prototype_only_no_task_or_model_evaluation',
        'case_count':len(records), 'cases':records,
        'max_reference_relative_error':max(r['reference_relative_error'] for r in records),
        'max_traction_relative_error':max(r['traction_relative_error'] for r in records),
        'max_dissipation_identity_relative_error':max(r['fluid_dissipation_identity_relative_error'] for r in records),
        'max_boundary_error':max(r['boundary_error'] for r in records),
        'max_refinement_relative_change':max(refinements),
        'finite_frequency_source_relative_error_range':[min(r['source_relative_error'] for r in finite_cases),max(r['source_relative_error'] for r in finite_cases)],
        'finite_frequency_psd_range_m2_per_s':[min(r['correct_psd_m2_per_s'] for r in finite_cases),max(r['correct_psd_m2_per_s'] for r in finite_cases)],
        'calibration':{'unique_scalar_readings':len(exact_cal),'repetitions':8,
            'fixed_sigma_um_per_s':fixed_sigma, 'noiseless_recovery_relative_error':abs(noiseless/true_eta-1),
            'grid_local_minima':int(np.sum((objectives[1:-1]<objectives[:-2])&(objectives[1:-1]<objectives[2:]))),
            'endpoint_recovery_relative_error_max':max(endpoint_recovery),
            'noise_draws':len(fitted),'noise_max_parameter_relative_error':max(abs(np.array(fitted)/true_eta-1)),
            'min_nonzero_abs_response_um_per_s':float(min(abs(exact_cal[np.nonzero(exact_cal)]))),
            'both_controls_deterministic_predictions_identical':True},
        'dc_noise_equivalence_error':dc_error,'spectral_evenness_error':even_error,
        'mobility_frequency_conjugacy_error':conjugacy,
        'physical_regime':{
            'max_omega_a_over_sound_speed_using_1500m_s':8e6*1.5e-6/1500,
            'max_force_Reynolds_bound':RHO*FORCE/(6*np.pi*min(ETA_BOUNDS)**2),
            'thermal_Reynolds_bound_using_bare_mass':max(RHO*a*np.sqrt(KB*320/((4*np.pi/3)*RHO_PARTICLE*a**3))/min(ETA_BOUNDS) for a in [.5e-6,1.5e-6]),
            'max_hbar_omega_over_kBT':1.054571817e-34*8e6/(KB*280),
            'sum_rule_scope':'No microscopic infinite-frequency equipartition claim; the readout is a finite-frequency PSD in the specified linear incompressible continuum.'},
        'seconds':time.perf_counter()-start,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    assert report['max_reference_relative_error']<1e-8
    assert report['max_traction_relative_error']<1e-9
    assert report['max_dissipation_identity_relative_error']<1e-8
    assert report['max_refinement_relative_change']<1e-9
    assert report['finite_frequency_source_relative_error_range'][0]>.1
    assert report['calibration']['grid_local_minima']==1
    assert report['calibration']['noise_max_parameter_relative_error']<.003
    path = Path(__file__).with_name('report.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__ == '__main__':
    main()
