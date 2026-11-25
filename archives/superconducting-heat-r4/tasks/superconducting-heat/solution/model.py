"""Spin-resolved driven contact with finite spin-population relaxation."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit

KB = 1.380649e-23
ELECTRON_CHARGE = 1.602176634e-19
NODES, WEIGHTS = leggauss(192)


def gap(temperature, critical):
    return 1.764 * critical * np.tanh(1.74 * np.sqrt(critical / temperature - 1))


@lru_cache(maxsize=2048)
def power_per_conductance(left, right, phase, amplitude=0., drive_energy=.3):
    dl, dr = gap(left, 1.2), gap(right, 1.9)
    cosine = np.cos(phase)
    transfer = 0.
    for n, weight in ((0, 1.), (-1, amplitude**2 / 2), (1, amplitude**2 / 2)):
        if not weight:
            continue
        shift = n * drive_energy
        edge = max(dl, dr - shift)
        upper = np.sqrt(60 * max(left, right))
        z = (NODES + 1) * upper / 2
        energy = edge + z*z
        other = energy + shift
        denominator = np.sqrt((energy-dl)*(energy+dl)*(other-dr)*(other+dr))
        coherence_sign = -1. if n == 0 else 1.
        spectral = (energy*other+coherence_sign*dl*dr*cosine) / denominator
        occupation = expit(-energy/left)-expit(-other/right)
        transfer += weight * upper/2 * (WEIGHTS @ (2*z*energy*spectral*occupation))
    creation = 0.
    if amplitude and drive_energy > dl + dr:
        angle = (NODES + 1) * np.pi / 4
        energy = dl + (drive_energy-dl-dr) * np.sin(angle)**2
        other = drive_energy-energy
        occupation = 1-expit(-energy/left)-expit(-other/right)
        regular = 2*energy*(energy*other-dl*dr*cosine) / np.sqrt((energy+dl)*(other+dr))
        creation = amplitude**2/2 * np.pi/4 * (WEIGHTS @ (regular*occupation))
    return 2e6*(KB/ELECTRON_CHARGE)**2*(transfer-creation)


from scipy.optimize import least_squares

# Positive-energy quasiparticle states per spin per kelvin, including both xi branches.
QUASIPARTICLE_DENSITY = 1e6
NORMAL_POLARIZATION_TIME = 1e-5
RATE_CONVERSION = 2e-6 * KB / ELECTRON_CHARGE**2
HEAT_CONVERSION = 2e6 * (KB / ELECTRON_CHARGE)**2


@lru_cache(maxsize=2048)
def prepare_channels(left, right, phase, amplitude, drive_energy, nodes=192,
                     normal=False, polarization_window=20.):
    dl, dr = (0., 0.) if normal else (gap(left, 1.2), gap(right, 1.9))
    x, weights = leggauss(nodes)
    upper = np.sqrt(80 * max(left, right) + polarization_window)
    channels = []
    for n in (-1, 0, 1):
        if n and not amplitude:
            continue
        edge = max(dl, dr - n*drive_energy)
        z = (x + 1)*upper/2
        energy, other = edge + z*z, edge + z*z + n*drive_energy
        denominator = np.sqrt((energy-dl)*(energy+dl)*(other-dr)*(other+dr))
        sign = -1 if n == 0 else 1
        spectral = (energy*other + sign*dl*dr*np.cos(phase))/denominator
        measure = weights*upper/2 * 2*z*spectral
        spin_channels = ((1, 1, .5), (-1, -1, .5)) if n == 0 else ((-n, n, amplitude**2/2),)
        for sl, sr, weight in spin_channels:
            channels.append((energy, other, measure*weight, sl, sr, False, n*drive_energy))
    if amplitude and drive_energy > dl + dr:
        angle = (x + 1)*np.pi/4
        energy = dl + (drive_energy-dl-dr)*np.sin(angle)**2
        other = drive_energy-energy
        measure = weights*np.pi/4 * amplitude**2/2 * 2*(energy*other-dl*dr*np.cos(phase)) / np.sqrt((energy+dl)*(other+dr))
        channels.append((energy, other, measure, 1, 1, True, drive_energy))
    return dl, dr, channels


def transition_integrals(left, right, channels, polarization):
    heat, spin, jacobian = np.zeros(3), np.zeros(2), np.zeros((2, 2))
    entropy = 0.
    for energy, other, measure, sl, sr, creation, work in channels:
        fl = expit((sl*polarization[0]-energy)/left)
        fr = expit((sr*polarization[1]-other)/right)
        dl = fl*(1-fl)/left
        dr = fr*(1-fr)/right
        if creation:
            occupation = 1-fl-fr
            injection = np.array([sl, sr])
            derivatives = np.array([-sl*dl, -sr*dr])
            loss_l, loss_r = -energy, -other
            affinity = (energy-sl*polarization[0])/left + (other-sr*polarization[1])/right
        else:
            occupation = fl-fr
            injection = np.array([-sl, sr])
            derivatives = np.array([sl*dl, -sr*dr])
            loss_l, loss_r = energy, -other
            affinity = -(energy-sl*polarization[0])/left + (other-sr*polarization[1])/right
        net = measure@occupation
        heat += [measure@(occupation*loss_l), measure@(occupation*loss_r), work*net]
        spin += injection*net
        jacobian += np.outer(injection, derivatives@measure)
        entropy += measure@(occupation*affinity)
    return heat, spin, jacobian, entropy


def spin_loss(temperature, delta, polarization):
    """DOS times equal-energy polarization decay, without the nu/tau prefactor."""
    magnitude = abs(polarization)
    if magnitude/temperature < 50:
        loss = np.sign(polarization)*temperature*np.log1p(
            expit((-magnitude-delta)/temperature)*np.expm1(2*magnitude/temperature))
    else:
        loss = temperature*(np.logaddexp(0., (polarization-delta)/temperature) -
                            np.logaddexp(0., (-polarization-delta)/temperature))
    derivative = expit((polarization-delta)/temperature) + expit((-polarization-delta)/temperature)
    return loss, derivative


@lru_cache(maxsize=8192)
def stationary_state(left, right, phase, amplitude, drive_energy, conductance,
                     nodes=192, normal=False, relaxation_time=NORMAL_POLARIZATION_TIME,
                     density=QUASIPARTICLE_DENSITY, initial=(0., 0.)):
    ratio = density/(relaxation_time*RATE_CONVERSION*conductance)
    bound, guess = 20., np.array(initial, dtype=float)
    for expansion in range(8):
        dl, dr, channels = prepare_channels(left, right, phase, amplitude, drive_energy,
                                           nodes, normal, bound)
        def residual(mu):
            injection = transition_integrals(left, right, channels, mu)[1]
            loss = np.array([spin_loss(t, d, m)[0] for t, d, m in zip((left, right), (dl, dr), mu)])
            return injection-ratio*loss
        def derivative(mu):
            jacobian = transition_integrals(left, right, channels, mu)[2]
            loss = [spin_loss(t, d, m)[1] for t, d, m in zip((left, right), (dl, dr), mu)]
            return jacobian-ratio*np.diag(loss)
        answer = least_squares(residual, guess, jac=derivative, bounds=(-bound, bound),
                               x_scale='jac', ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=200)
        mu = answer.x
        # Cold gaps can make the least-squares gradient tiny before balance is accurate.
        for _ in range(12):
            value = residual(mu)
            if max(abs(value)) < 2e-14:
                break
            correction = np.linalg.solve(derivative(mu), value)
            step = 1.
            while step > 1e-6:
                candidate = mu-step*correction
                if max(abs(candidate)) < bound and max(abs(residual(candidate))) < max(abs(value)):
                    mu = candidate
                    break
                step /= 2
            if step <= 1e-6:
                break
        if max(abs(residual(mu))) < 1e-10 and max(abs(mu)) < .8*bound:
            break
        bound *= 2
        guess = mu
    else:
        raise RuntimeError('Stationary spin balance did not converge')
    heat, spin, jacobian, entropy = transition_integrals(left, right, channels, mu)
    loss = np.array([spin_loss(t, d, m)[0] for t, d, m in zip((left, right), (dl, dr), mu)])
    scale = RATE_CONVERSION*conductance*KB
    return dict(polarization=mu, heat=HEAT_CONVERSION*conductance*heat,
                spin=RATE_CONVERSION*conductance*spin,
                residual=RATE_CONVERSION*conductance*residual(mu),
                jacobian=derivative(mu), tunneling_entropy=scale*entropy,
                relaxation_entropy=scale*ratio*np.sum(mu*loss/np.array([left, right])),
                search_bound=bound, search_expansions=expansion)


def predict_at(experiments, conductance):
    values = []
    for e in experiments:
        left, right, phase = e['left_temperature'], e['right_temperature'], e['phase']
        amplitude, drive = e.get('amplitude', 0.), e.get('drive_energy', .3)
        if not amplitude:
            values.append(conductance*power_per_conductance(left, right, phase, 0., drive))
        else:
            values.append(stationary_state(left, right, phase, amplitude, drive, conductance)['heat'][0])
    return np.array(values)


class Model:
    def __init__(self):
        self.conductance = None

    def fit(self, records):
        x = predict_at([r['input'] for r in records], 1.)
        y = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        self.conductance = float(np.clip(np.sum(x*y/sigma**2)/np.sum((x/sigma)**2), 20., 80.))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.conductance)
