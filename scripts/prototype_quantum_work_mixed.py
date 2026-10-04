"""Outside-canonical fourth-cumulant prototype; does not generate task data."""
import json
from math import factorial
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

D = np.diag([0., 1., 2.35])
X = np.array([[0., 1., .25], [1., 0., .8], [.25, .8, 0.]])
Y = np.array([[0., -1j, .45j], [1j, 0., -.7j], [-.45j, .7j, 0.]])
CALIBRATION = [(.4,.55,.6,.3,.4), (.65,.5,-.7,.45,.35),
               (.5,-.4,1.1,.6,.3), (.7,.6,1.5,.35,.55),
               (-.4,.7,-1.2,.5,.45), (.6,.35,.3,.7,.2)]
HIDDEN = [( .8,-.9,2.9,1.05,1.04), (.99,-.81,-3.1,.98,1.03),
          (-.65,-.87,-1.45,.69,1.25), (-1.05,.46,1.14,1.24,1.2)]


def to_cumulants(m):
    a,b,c,d = m
    return np.array([a,b-a*a,c-3*a*b+2*a**3,
                     d-4*a*c-3*b*b+12*a*a*b-6*a**4])


def statistics(pulse, scale):
    a,b,phase,t1,t2 = pulse
    initial = scale*D
    unitary = expm(-1j*t2*(initial+b*(np.cos(phase)*X+np.sin(phase)*Y)))
    unitary = unitary@expm(-1j*t1*(initial+a*X))
    final = unitary.conj().T@initial@unitary
    work = final-initial
    energies = np.diag(initial)
    difference = energies[:,None]-energies[None,:]
    probability = abs(unitary)**2/3
    tpm = np.array([np.sum(probability*difference**n) for n in range(1,5)])
    operator = np.array([np.trace(np.linalg.matrix_power(work,n)).real/3
                         for n in range(1,5)])
    commutator = initial@final-final@initial
    gap = float(np.vdot(commutator,commutator).real/3)
    return to_cumulants(tpm),to_cumulants(operator),gap


def characteristic_reference(pulse, scale, radius=.18, nodes=64):
    """Independent Schrödinger integration and analytic moment-generating coefficient integral."""
    a,b,phase,t1,t2 = pulse
    energies = scale*np.array([0.,1.,2.35])
    first = np.diag(energies).astype(complex)
    second = first.copy()
    for i,j,cx,cy in [(0,1,1.,-1j),(0,2,.25,.45j),(1,2,.8,-.7j)]:
        first[i,j] = a*cx
        first[j,i] = first[i,j].conjugate()
        second[i,j] = b*(np.cos(phase)*cx+np.sin(phase)*cy)
        second[j,i] = second[i,j].conjugate()
    unitary = np.eye(3,dtype=complex)
    for h,duration in [(first,t1),(second,t2)]:
        if duration:
            sol = solve_ivp(lambda t,y:(-1j*h@y.reshape(3,3)).ravel(),
                            [0,duration],unitary.ravel(),method='DOP853',
                            rtol=2e-12,atol=2e-14)
            assert sol.success
            unitary = sol.y[:,-1].reshape(3,3)
    angles = 2*np.pi*np.arange(nodes)/nodes
    generating = []
    for z in radius*np.exp(1j*angles):
        initial = np.diag(np.exp(-z*energies))/3
        evolved = unitary@initial@unitary.conj().T
        generating.append(np.trace(np.diag(np.exp(z*energies))@evolved))
    coefficients = np.fft.fft(generating)/nodes
    moments = np.array([(factorial(n)*coefficients[n]/radius**n).real
                        for n in range(1,5)])
    return to_cumulants(moments)


def calibration(scale):
    return np.array([statistics(p,scale)[0][1:3] for p in CALIBRATION]).ravel()


def run():
    rng = np.random.default_rng(319207)
    random_pulses = [tuple(rng.uniform([-1.1,-1.1,-np.pi,0,0],
                                     [1.1,1.1,np.pi,1.3,1.3])) for _ in range(128)]
    identity_error = gap_error = positivity_error = 0.
    for pulse in CALIBRATION+HIDDEN+random_pulses:
        scale = rng.uniform(.8,1.2)
        exact,shortcut,gap = statistics(pulse,scale)
        identity_error = max(identity_error,float(max(abs(exact[:3]-shortcut[:3]))))
        gap_error = max(gap_error,float(abs(exact[3]-shortcut[3]-gap)))
        positivity_error = max(positivity_error,max(0.,-gap))
    reference_error = reference_refinement = 0.
    for i,pulse in enumerate(CALIBRATION+HIDDEN+random_pulses[:14]):
        scale = [.8,1.06,1.2][i%3]
        exact = statistics(pulse,scale)[0]
        ref = characteristic_reference(pulse,scale)
        refined = characteristic_reference(pulse,scale,.24,96)
        reference_error = max(reference_error,float(max(abs(exact-ref))))
        reference_refinement = max(reference_refinement,float(max(abs(ref-refined))))

    grid = np.linspace(.8,1.2,201)
    curve = np.array([calibration(s) for s in grid])
    derivative = np.array([(calibration(s+1e-5)-calibration(s-1e-5))/2e-5 for s in grid])
    # Each selected variance is strictly increasing; this alone identifies scale.
    min_variance_derivative = float(derivative[:,::2].min())
    assert min_variance_derivative>0
    third_min = float(abs(curve[:,1::2]).min())
    sigma = .0003
    repeat = 24
    worst_relative_parameter = worst_chi2 = max_noise_prediction_error = 0.
    for i in range(256):
        true = np.linspace(.8,1.2,256)[i]
        observations = calibration(true)[None,:]+sigma*rng.normal(size=(repeat,12))
        means = observations.mean(axis=0)
        loss = lambda s:float(np.sum((calibration(s)-means)**2))
        fit = minimize_scalar(loss,bounds=(.8,1.2),method='bounded',options={'xatol':1e-12})
        candidates = [(float(fit.x),float(fit.fun)),(.8,loss(.8)),(1.2,loss(1.2))]
        fitted = min(candidates,key=lambda x:x[1])[0]
        worst_relative_parameter = max(worst_relative_parameter,abs(fitted/true-1))
        chi2 = float(np.sum(((observations-calibration(fitted))/sigma)**2)/(repeat*12-1))
        worst_chi2 = max(worst_chi2,chi2)
        actual = np.array([statistics(p,true)[0][3] for p in HIDDEN])
        inferred = np.array([statistics(p,fitted)[0][3] for p in HIDDEN])
        max_noise_prediction_error = max(max_noise_prediction_error,float(max(abs((inferred-actual)/actual))))

    hidden = []
    for pulse in HIDDEN:
        values = np.array([[statistics(pulse,s)[j][3] for j in [0,1]] for s in grid])
        assert np.max(values)<0
        hidden.append({'pulse':pulse,'tpm_range':values[:,0].tolist()[::100],
                       'operator_range':values[:,1].tolist()[::100],
                       'minimum_absolute_tpm_kappa4':float(abs(values[:,0]).min()),
                       'minimum_relative_error':float((abs(values[:,1]-values[:,0])/abs(values[:,0])).min())})
    report = {
        'scope':'Prototype only; no task, calibration, grading, or evaluation edits.',
        'preparation':'An exact uniform classical mixture of the three H0 eigenstates, independent on every repetition; no Gibbs-temperature parameter.',
        'measurement':'Two projective energy records at the beginning and end of the isolated cyclic drive; their difference is the measured work.',
        'identity':'For rho=I/3, the first three TPM and W=U†H0U−H0 cumulants coincide. The fourth difference equals Tr([H0,U†H0U]†[H0,U†H0U])/3.',
        'classification':'Physical measurement error: the Hermitian energy-change operator defines a valid different single-measurement distribution. Its exact state and unitary do not make it the two-energy-record distribution.',
        'calibration_pulses':CALIBRATION,'calibration_cumulants':[2,3],
        'calibration_sigma':sigma,'calibration_repeats':repeat,'calibration_records':repeat*12,
        'identity_first_three_max_absolute_error':identity_error,
        'fourth_commutator_identity_max_absolute_error':gap_error,
        'commutator_nonnegativity_violation':positivity_error,
        'independent_characteristic_reference_max_absolute_error':reference_error,
        'characteristic_radius_nodes_refinement_max_change':reference_refinement,
        'minimum_calibration_variance_derivative_on_201_scale_grid':min_variance_derivative,
        'minimum_absolute_nonzero_calibration_third_cumulant':third_min,
        'noise_fits':256,'true_scales':'.8 to1.2 inclusive, equally spaced',
        'maximum_noise_relative_parameter_error':worst_relative_parameter,
        'maximum_noise_reduced_chi2':worst_chi2,
        'maximum_noise_hidden_relative_error':max_noise_prediction_error,
        'hidden_controls':hidden,
        'limits':'Zero pulse and protocols whose pulse Hamiltonians commute with H0 give zero work. No mean-calibration information exists in a cyclic maximally mixed preparation. Variance and nonzero third cumulant must identify the scale.',
        'novelty':'A fourth-order extension of the same operational work-measurement family, not a new task family.',
    }
    destination = Path('results/quantum-work-mixed-r2-prototype.json')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    run()
