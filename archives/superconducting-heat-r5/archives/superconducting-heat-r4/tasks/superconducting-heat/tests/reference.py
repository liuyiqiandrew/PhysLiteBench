"""Independent spin-matrix transition rates, adaptive quadrature and kinetic balance."""
from functools import lru_cache
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import root, least_squares
from scipy.special import expit

TRUE_PARAMETER = 45.
KB = 1.380649e-23
CHARGE = 1.602176634e-19
DENSITY = 1e6
POLARIZATION_TIME = 1e-5
RATE = 2e-6*KB/CHARGE**2
HEAT = 2e6*(KB/CHARGE)**2
SINGLET = np.array([[0., 1.], [-1., 0.]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)
RAISE = np.array([[0., 1.], [0., 0.]], dtype=complex)
LOWER = RAISE.T


def material_gaps(left, right):
    return tuple(1.764*tc*np.tanh(1.74*np.sqrt(tc/t-1)) for t, tc in ((left, 1.2), (right, 1.9)))


def spin_matrices(amplitude, chirality=1):
    return {0: IDENTITY, 1: amplitude*(RAISE if chirality == 1 else LOWER),
            -1: amplitude*(LOWER if chirality == 1 else RAISE)}


def branch_probability(energy, other, dl, dr, phase, harmonic, amplitude,
                       creation=False, chirality=1):
    matrices = spin_matrices(amplitude, chirality)
    electron = matrices[harmonic]
    hole = SINGLET@matrices[-harmonic].conj()@SINGLET.conj().T
    probability = np.zeros((2, 2))
    xi_l, xi_r = np.sqrt(max(0., energy*energy-dl*dl)), np.sqrt(max(0., other*other-dr*dr))
    for bl in (-1., 1.):
        ul, vl = np.sqrt((1+bl*xi_l/energy)/2), np.sqrt((1-bl*xi_l/energy)/2)
        for br in (-1., 1.):
            ur, vr = np.sqrt((1+br*xi_r/other)/2), np.sqrt((1-br*xi_r/other)/2)
            if creation:
                transition = (ul*vr*electron+vl*ur*np.exp(1j*phase)*hole)@SINGLET
            else:
                transition = ul*ur*electron-vl*vr*np.exp(1j*phase)*hole
            probability += abs(transition)**2/4
    return probability


def relaxation_integral(temperature, delta, mu):
    """Integrated collision kernel, using a separate stable logarithm identity."""
    x = mu/temperature
    if abs(x) < 30:
        a = np.exp(-delta/temperature)
        value = 2*temperature*np.arctanh(a*np.sinh(x)/(1+a*np.cosh(x)))
    else:
        value = temperature*(np.logaddexp(0., (mu-delta)/temperature) -
                             np.logaddexp(0., (-mu-delta)/temperature))
    slope = expit((mu-delta)/temperature)+expit((-mu-delta)/temperature)
    return value, slope


def transitions(left, right, phase, amplitude, drive_energy, mu,
                tolerance=1e-10, normal=False, chirality=1):
    dl, dr = (0., 0.) if normal else material_gaps(left, right)
    spins = np.array([1., -1.])
    values = np.zeros(10)
    def occupations(energy, other, probability, creation, work):
        fl = expit((spins*mu[0]-energy)/left)
        fr = expit((spins*mu[1]-other)/right)
        dlmu, drmu = spins*fl*(1-fl)/left, spins*fr*(1-fr)/right
        result = np.zeros(10)
        for i, sl in enumerate(spins):
            for j, sr in enumerate(spins):
                if not probability[i, j]:
                    continue
                if creation:
                    net = 1-fl[i]-fr[j]
                    injection = np.array([sl, sr])
                    derivatives = np.array([-dlmu[i], -drmu[j]])
                    heat = np.array([-energy, -other, work])
                    affinity = (energy-sl*mu[0])/left+(other-sr*mu[1])/right
                else:
                    net = fl[i]-fr[j]
                    injection = np.array([-sl, sr])
                    derivatives = np.array([dlmu[i], -drmu[j]])
                    heat = np.array([energy, -other, work])
                    affinity = -(energy-sl*mu[0])/left+(other-sr*mu[1])/right
                result += probability[i, j]*np.r_[net*heat, net*injection,
                                                 np.outer(injection, derivatives).ravel(), net*affinity]
        return result
    upper = np.sqrt(100*max(left, right)+max(abs(np.array(mu)))+drive_energy)
    for n in (-1, 0, 1):
        if n and not amplitude:
            continue
        edge = max(dl, dr-n*drive_energy)
        def integrand(z):
            energy, other = edge+z*z, edge+z*z+n*drive_energy
            dos = energy*other/np.sqrt((energy-dl)*(energy+dl)*(other-dr)*(other+dr))
            probability = branch_probability(energy, other, dl, dr, phase, -n, amplitude,
                                             chirality=chirality)
            return 2*z*dos*occupations(energy, other, probability, False, n*drive_energy)
        values += quad_vec(integrand, 0., upper, epsabs=tolerance, epsrel=tolerance)[0]
    if amplitude and drive_energy > dl+dr:
        interval = drive_energy-dl-dr
        def integrand(angle):
            energy = dl+interval*np.sin(angle)**2
            other = drive_energy-energy
            dos = 2*energy*other/np.sqrt((energy+dl)*(other+dr))
            probability = branch_probability(energy, other, dl, dr, phase, 1, amplitude,
                                             creation=True, chirality=chirality)
            return dos*occupations(energy, other, probability, True, drive_energy)
        values += quad_vec(integrand, 0., np.pi/2, epsabs=tolerance, epsrel=tolerance)[0]
    return values


@lru_cache(maxsize=4096)
def energy_components(left, right, phase, amplitude=0., drive_energy=.3,
                      conductance=TRUE_PARAMETER, tolerance=1e-10, normal=False,
                      chirality=1, relaxation_time=POLARIZATION_TIME, initial=(0., 0.),
                      force_unpolarized=False):
    dl, dr = (0., 0.) if normal else material_gaps(left, right)
    ratio = DENSITY/(relaxation_time*RATE*conductance)
    @lru_cache(maxsize=128)
    def integrals(mu):
        return transitions(left, right, phase, amplitude, drive_energy, mu, tolerance, normal, chirality)
    def balance(mu):
        losses = [relaxation_integral(t, d, m)[0] for t, d, m in zip((left, right), (dl, dr), mu)]
        return integrals(tuple(mu))[3:5]-ratio*np.array(losses)
    def jacobian(mu):
        slopes = [relaxation_integral(t, d, m)[1] for t, d, m in zip((left, right), (dl, dr), mu)]
        return integrals(tuple(mu))[5:9].reshape(2, 2)-ratio*np.diag(slopes)
    if force_unpolarized or not amplitude:
        mu = np.zeros(2)
    else:
        answer = root(balance, initial, jac=jacobian, method='hybr', options={'xtol': 1e-10})
        mu = answer.x
        if max(abs(balance(mu))) > 2e-10:
            bound, guess = 20., np.zeros(2)
            for _ in range(8):
                answer = least_squares(balance, guess, jac=jacobian, bounds=(-bound, bound),
                                       x_scale='jac', ftol=1e-13, xtol=1e-13, gtol=1e-13)
                mu = answer.x
                if max(abs(balance(mu))) < 2e-10 and max(abs(mu)) < .8*bound:
                    break
                bound *= 2
                guess = mu
            else:
                raise RuntimeError('Independent spin balance failed')
        for _ in range(4):
            value = balance(mu)
            if max(abs(value)) < 2e-12:
                break
            mu -= np.linalg.solve(jacobian(mu), value)
        if max(abs(balance(mu))) > 2e-10:
            raise RuntimeError('Independent spin balance residual')
    value = integrals(tuple(mu))
    losses = [relaxation_integral(t, d, m)[0] for t, d, m in zip((left, right), (dl, dr), mu)]
    scale = RATE*conductance*KB
    return dict(heat=HEAT*conductance*value[0], right_heat=HEAT*conductance*value[1],
                work=HEAT*conductance*value[2], polarization=mu, spin=RATE*conductance*value[3:5],
                residual=RATE*conductance*balance(mu), jacobian=jacobian(mu),
                tunneling_entropy=scale*value[9],
                relaxation_entropy=scale*ratio*np.sum(mu*np.array(losses)/np.array([left, right])))


def predict(experiments, conductance):
    return np.array([energy_components(e['left_temperature'], e['right_temperature'], e['phase'],
                    e.get('amplitude', 0.), e.get('drive_energy', .3), conductance=conductance)['heat']
                    for e in experiments])

def experiment(left,right,phase,amplitude=None,drive_energy=None):
    result = dict(left_temperature=float(left),right_temperature=float(right),phase=float(phase))
    if amplitude is not None:
        result['amplitude'] = float(amplitude)
    if drive_energy is not None:
        result['drive_energy'] = float(drive_energy)
    return result


def calibration_inputs():
    return [experiment(a,b,p) for a in np.linspace(.35,1.,8) for b in np.linspace(.3,1.02,9) for p in (-np.pi/2,np.pi/2)]


def hidden_inputs():
    settings = ((.25,.25,.6,7.),(.45,.8,.8,8.),(.75,.5,1.,10.),(1.05,.25,.7,12.),(.6,.9,.9,9.))
    return {'aligned_spin_phase':[experiment(l,r,0.,a,w) for l,r,a,w in settings],
            'opposite_spin_phase':[experiment(l,r,np.pi,a,w) for l,r,a,w in settings],
            'oblique_spin_phase':[experiment(l,r,p,a,w) for (l,r,a,w),p in zip(settings,(-.7,.5,1.1,-.9,.8))]}
