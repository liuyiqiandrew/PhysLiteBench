"""Thermal quantum phase oscillator and stationary spin-resolved BCS heat."""
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


def r4_predict_at(experiments, conductance):
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
        """Fit the single unknown normal-state conductance by weighted least squares."""
        if not records:
            raise ValueError('At least one calibration record is required.')
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0):
            raise ValueError('Calibration values must be finite and uncertainties positive.')
        experiments = [record['input'] for record in records]

        def residual(parameters):
            return (predict_at(experiments, float(parameters[0])) - values) / sigmas

        answer = least_squares(residual, np.array([50.]), bounds=([20.], [80.]),
                               xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=100)
        self.conductance = float(answer.x[0])
        return self

    def predict(self, experiments):
        if self.conductance is None:
            raise ValueError('Call fit() first.')
        result = np.asarray(predict_at(experiments, float(self.conductance)), dtype=float)
        if result.shape != (len(experiments),) or not np.all(np.isfinite(result)):
            raise FloatingPointError('Prediction was not a finite one-dimensional array.')
        return result


from scipy.special import gammaln, logsumexp
from scipy.linalg import eigh_tridiagonal,expm
OSCILLATOR_ENERGY = 24.


def sidebands(rho, temperature, count=24, energy=OSCILLATOR_ENERGY):
    ell = np.arange(-count, count+1)
    if rho == 0:
        return ell, (ell == 0).astype(float)
    nb = 1/np.expm1(energy/temperature)
    emission, absorption = rho*(nb+1), rho*nb
    # Difference of two independent Poisson counts, summed in log space.
    # This also preserves the zero-temperature Poisson limit.
    positive = np.arange(count+1)[:,None]
    pairs = np.arange(64)[None,:]
    if absorption == 0:
        log_positive = -emission+positive[:,0]*np.log(emission)-gammaln(positive[:,0]+1)
    else:
        log_positive = (-emission-absorption+positive[:,0]*np.log(emission)
                        +logsumexp(pairs*np.log(emission*absorption)
                                   -gammaln(pairs+1)-gammaln(pairs+positive+1),axis=1))
    log_probabilities = log_positive[abs(ell)] + np.minimum(ell,0)*energy/temperature
    return ell, np.exp(log_probabilities)



@lru_cache(256)
def nonlinear_spectrum(rho,eta,temperature,count=64,states=192):
    x,v=eigh_tridiagonal(np.zeros(states),np.sqrt(np.arange(1,states)))
    unitary=(v*np.exp(1j*(np.sqrt(rho)*x+eta*x*x)))@v.T
    thermal=(1-np.exp(-24/temperature))*np.exp(-24/temperature*np.arange(states))
    p=[];c=[]
    for ell in range(-count,count+1):
        m=np.arange(max(0,-ell),min(states,states-ell));d=unitary[m+ell,m]
        p.append(thermal[m]@abs(d)**2);c.append(thermal[m]@(d*d))
    return np.arange(-count,count+1),np.array(p),np.array(c)

def operator_matrix(rho,eta,chi,states=192,commuting=False):
    x,v=eigh_tridiagonal(np.zeros(states),np.sqrt(np.arange(1,states)))
    if commuting:
        ann=np.diag(np.sqrt(np.arange(1,states)),1);X=ann+ann.T;P=1j*(ann.T-ann)
        return expm(1j*(np.sqrt(rho)*X+eta*X@X+chi*P))
    first=(v*np.exp(1j*(np.sqrt(rho)*x+eta*x*x)))@v.T
    # A phase rotation maps X to P with the convention P=i(b†-b).
    # The opposite phase would implement -P and reverse the ordered coupling.
    momentum=(-1j)**np.arange(states)[:,None]*v
    second=(momentum*np.exp(1j*chi*x))@momentum.conj().T
    return first@second

def spectrum(matrix,temperature,count=64,thermal_states=None):
    states=len(matrix);thermal_states=thermal_states or states
    data=[]
    for ell in range(-count,count+1):
        m=np.arange(max(0,-ell),min(thermal_states,states-ell));a=matrix[m+ell,m];b=matrix[m,m+ell]
        thermal=(1-np.exp(-24/temperature))*np.exp(-24/temperature*m)
        data.append((ell,float(thermal@abs(a)**2),float(thermal@abs(b)**2),complex(thermal@(a*b))))
    return data

@lru_cache(256)
def ordered_spectrum(rho,eta,chi,temperature,count=64,states=192):
    rows=spectrum(operator_matrix(rho,eta,chi,states),temperature,count)
    return np.array([r[0] for r in rows]),np.array([(r[1]+r[2])/2 for r in rows]),np.array([r[3] for r in rows])

@lru_cache(256)
def bath_spectrum(rho, secondary, temperature, count=64, quadratic_coupling=0., momentum_coupling=0.):
    if momentum_coupling:
        first,p,cross=ordered_spectrum(rho,quadratic_coupling,momentum_coupling,temperature,count)
    elif quadratic_coupling:
        first,p,cross=nonlinear_spectrum(rho,quadratic_coupling,temperature,count)
    else:
        first,p=sidebands(rho,temperature,count);cross=(-1.)**first*p
    if secondary==0:return first,p,cross
    second,q=sidebands(secondary,temperature,count,energy=2*OSCILLATOR_ENERGY)
    separated=np.zeros(4*count+1);separated[::2]=q
    signed=np.zeros(4*count+1);signed[::2]=(-1.)**second*q
    return (np.arange(-3*count,3*count+1),np.convolve(p,separated),
            np.convolve(cross,signed))


@lru_cache(maxsize=512)
def prepare(left, right, phase, amplitude, drive, rho, oscillator_temperature,
            nodes=192, count=64, window=40., normal=False, factorized=False, secondary_coupling=0., quadratic_coupling=0., momentum_coupling=0.):
    dl, dr = (0., 0.) if normal else (gap(left, 1.2), gap(right, 1.9))
    x, weights = leggauss(nodes)
    ells, probabilities, coherences = bath_spectrum(rho, secondary_coupling, oscillator_temperature, count, quadratic_coupling, momentum_coupling)
    nb = 1/np.expm1(OSCILLATOR_ENERGY/oscillator_temperature)
    nb_second = 1/np.expm1(2*OSCILLATOR_ENERGY/oscillator_temperature)
    sigma=2*nb+1
    denominator=1-4j*quadratic_coupling*sigma
    dephasing = np.exp(-2*rho*sigma/denominator-2*secondary_coupling*(2*nb_second+1))/np.sqrt(denominator)
    upper = np.sqrt(80*max(left, right)+window)
    z = (x+1)*upper/2
    angle = (x+1)*np.pi/4
    channels = []
    for n in (-1, 0, 1):
        if n and amplitude == 0:
            continue
        transfer_spins = ((1, 1, .5), (-1, -1, .5)) if n == 0 else ((-n, n, amplitude**2/2),)
        pair_spins = ((1, -1, .5), (-1, 1, .5)) if n == 0 else ((n, n, amplitude**2/2),)
        for ell, probability, reverse_probability, coherence in zip(ells, probabilities, probabilities[::-1], coherences):
            if max(probability,reverse_probability) < 1e-26:
                continue
            shift = n*drive-ell*OSCILLATOR_ENERGY
            parity = dephasing if factorized else complex(float(coherence.real)/float(probability), float(coherence.imag)/float(probability)) if probability>0 else 0.
            phase_weight=np.real(parity*np.exp(-1j*phase))
            sign = (-1 if n == 0 else 1)*phase_weight
            edge = max(dl, dr-shift)
            energy, other = edge+z*z, edge+z*z+shift
            denominator = np.sqrt((energy-dl)*(energy+dl)*(other-dr)*(other+dr))
            measure = weights*upper/2*2*z*(energy*other+sign*dl*dr)/denominator
            for sl, sr, weight in transfer_spins:
                channels.append((energy, other, measure*weight, sl, sr, False,
                                 n*drive, ell*OSCILLATOR_ENERGY, probability, reverse_probability))
            if shift > dl+dr:
                energy = dl+(shift-dl-dr)*np.sin(angle)**2
                other = shift-energy
                sign = (1 if n == 0 else -1)*phase_weight
                measure = weights*np.pi/4*2*(energy*other+sign*dl*dr)/np.sqrt((energy+dl)*(other+dr))
                for sl, sr, weight in pair_spins:
                    channels.append((energy, other, measure*weight, sl, sr, True,
                                     n*drive, ell*OSCILLATOR_ENERGY, probability, reverse_probability))
    fields=list(zip(*channels))
    energy,other,measure=[np.stack(fields[j]) for j in range(3)]
    sl,sr=np.array(fields[3])[:,None],np.array(fields[4])[:,None]
    creation=np.array(fields[5])[:,None]
    work,recoil=np.array(fields[6]),np.array(fields[7])
    forward,backward=np.array(fields[8])[:,None],np.array(fields[9])[:,None]
    injection=np.column_stack((np.where(creation[:,0],sl[:,0],-sl[:,0]),sr[:,0]))
    return dl,dr,(energy,other,measure,sl,sr,creation,work,recoil,forward,backward,injection)


def transition(left,right,oscillator_temperature,channels,mu):
    energy,other,measure,sl,sr,creation,work,recoil,pf,pb,injection=channels
    fl=expit((sl*mu[0]-energy)/left)
    fr=expit((sr*mu[1]-other)/right)
    dl,dr=fl*(1-fl)/left,fr*(1-fr)/right
    occupation=np.where(creation,pf*(1-fl)*(1-fr)-pb*fl*fr,
                        pf*fl*(1-fr)-pb*fr*(1-fl))
    dmu_l=np.where(creation,-sl*dl,sl*dl)*(pf*(1-fr)+pb*fr)
    dmu_r=-sr*dr*np.where(creation,pf*(1-fl)+pb*fl,pf*fl+pb*(1-fl))
    loss_l=np.where(creation,-energy,energy)
    weighted=measure*occupation
    net=weighted.sum(axis=1)
    heat=np.array([np.sum(weighted*loss_l),-np.sum(weighted*other),work@net,recoil@net])
    spin=injection.T@net
    jacobian=injection.T@np.column_stack(((measure*dmu_l).sum(axis=1),(measure*dmu_r).sum(axis=1)))
    affinity=np.where(creation,(energy-sl*mu[0])/left,-(energy-sl*mu[0])/left)+(other-sr*mu[1])/right+recoil[:,None]/oscillator_temperature
    entropy=np.sum(weighted*affinity)
    minimum_rate=float(min(np.min(measure),np.min(occupation*affinity)))
    return heat,spin,jacobian,entropy,minimum_rate


@lru_cache(maxsize=4096)
def stationary(left, right, phase, amplitude, drive, rho, oscillator_temperature,
               conductance, nodes=192, count=64, normal=False, factorized=False,
               initial=(0., 0.), secondary_coupling=0., quadratic_coupling=0., momentum_coupling=0.):
    ratio = QUASIPARTICLE_DENSITY/(NORMAL_POLARIZATION_TIME*RATE_CONVERSION*conductance)
    bound, guess = 40., np.array(initial, dtype=float)
    for expansion in range(8):
        dl, dr, channels = prepare(left, right, phase, amplitude, drive, rho, oscillator_temperature,
                                   nodes, count, bound, normal, factorized, secondary_coupling, quadratic_coupling, momentum_coupling)
        def residual(mu):
            spin = transition(left, right, oscillator_temperature, channels, mu)[1]
            losses = [spin_loss(t, d, m)[0] for t,d,m in zip((left,right),(dl,dr),mu)]
            return spin-ratio*np.array(losses)
        def derivative(mu):
            jac = transition(left, right, oscillator_temperature, channels, mu)[2]
            losses = [spin_loss(t,d,m)[1] for t,d,m in zip((left,right),(dl,dr),mu)]
            return jac-ratio*np.diag(losses)
        answer = least_squares(residual, guess, jac=derivative, bounds=(-bound,bound),
                               x_scale='jac',ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=200)
        mu = answer.x
        for _ in range(15):
            value=residual(mu)
            if max(abs(value)) < 1e-12: break
            correction=np.linalg.solve(derivative(mu),value)
            step=1.
            while step>1e-8:
                candidate=mu-step*correction
                if max(abs(candidate)) < bound and max(abs(residual(candidate))) < max(abs(value)):
                    mu=candidate
                    break
                step/=2
            if step<=1e-8: break
        if max(abs(residual(mu)))<1e-9 and max(abs(mu))<.8*bound: break
        bound*=2
        guess=mu
    else: raise RuntimeError('Spin balance failed')
    heat,spin,jac,entropy,minrate=transition(left,right,oscillator_temperature,channels,mu)
    losses=np.array([spin_loss(t,d,m)[0] for t,d,m in zip((left,right),(dl,dr),mu)])
    scale=RATE_CONVERSION*conductance*KB
    return dict(mu=mu,heat=HEAT_CONVERSION*conductance*heat,
                spin=RATE_CONVERSION*conductance*spin,
                residual=residual(mu),jacobian=derivative(mu),
                entropy=scale*entropy,flip_entropy=scale*ratio*np.sum(mu*losses/np.array([left,right])),
                minimum_rate=minrate,search_bound=bound,expansions=expansion)




def oscillator_arguments(e):
    return (e['left_temperature'],e['right_temperature'],e['phase'],
            e.get('amplitude',0.),e.get('drive_energy',.3),
            e.get('oscillator_coupling',0.),e.get('oscillator_temperature',12.))


def mean_heat(e,conductance):
    if e.get('oscillator_coupling',0.)==0 and e.get('secondary_coupling',0.)==0 and e.get('quadratic_coupling',0.)==0 and e.get('momentum_coupling',0.)==0:
        return float(r4_predict_at([e],conductance)[0])
    return float(stationary(*oscillator_arguments(e),conductance,factorized=FACTORIZED,secondary_coupling=e.get('secondary_coupling',0.),quadratic_coupling=e.get('quadratic_coupling',0.),momentum_coupling=e.get('momentum_coupling',0.))['heat'][0])


def predict_at(experiments,conductance):
    values=[]
    for e in experiments:
        value=mean_heat(e,conductance)
        if e.get('statistic','mean')=='phase_contrast':
            other=dict(e,phase=(e['phase']+2*np.pi)%(2*np.pi)-np.pi)
            value=(value-mean_heat(other,conductance))/2
        values.append(value)
    return np.array(values)


FACTORIZED = False


def _poly_mul(a, b):
    """Multiply truncated power series whose first axis is polynomial order."""
    out = np.zeros_like(a, dtype=float)
    for i in range(4):
        for j in range(4-i):
            out[i+j] += a[i] * b[j]
    return out


def _poly_sigmoid(spin, energy, temperature, mu_series, island):
    """Taylor series of f(E,mu) in an auxiliary scalar parameter."""
    spin = np.asarray(spin, dtype=float)
    energy = np.asarray(energy, dtype=float)
    z = np.empty((4, energy.size), dtype=float)
    z[0] = (spin*mu_series[0, island] - energy) / temperature
    for k in range(1, 4):
        z[k] = spin*mu_series[k, island] / temperature
    f0 = expit(z[0])
    fp = f0*(1-f0)
    fpp = fp*(1-2*f0)
    fppp = fp*(1-6*f0+6*f0*f0)
    result = np.empty_like(z)
    result[0] = f0
    result[1] = fp*z[1]
    result[2] = fp*z[2] + .5*fpp*z[1]**2
    result[3] = fp*z[3] + fpp*z[1]*z[2] + (fppp/6)*z[1]**3
    return result


def _rate_series(events, mu_series):
    """Rates and their spin-potential derivatives as Taylor series."""
    el, er, sl, sr, pair, base, direction, v, q = events[:9]
    left, right = events[9]
    tl, tr = left, right
    fl = _poly_sigmoid(sl, el, tl, mu_series, 0)
    fr = _poly_sigmoid(sr, er, tr, mu_series, 1)
    one = np.zeros_like(fl); one[0] = 1.
    ol, ore = one-fl, one-fr
    flmu = (sl/tl) * _poly_mul(fl, ol)
    frmu = (sr/tr) * _poly_mul(fr, ore)
    fwd = direction > 0

    occ = np.where(pair[None, :],
                   np.where(fwd[None, :], _poly_mul(ol, ore), _poly_mul(fl, fr)),
                   np.where(fwd[None, :], _poly_mul(fl, ore), _poly_mul(fr, ol)))
    dl = np.where(pair[None, :],
                  np.where(fwd[None, :], -_poly_mul(flmu, ore), _poly_mul(flmu, fr)),
                  np.where(fwd[None, :], _poly_mul(flmu, ore), -_poly_mul(fr, flmu)))
    dr = np.where(pair[None, :],
                  np.where(fwd[None, :], -_poly_mul(ol, frmu), _poly_mul(fl, frmu)),
                  np.where(fwd[None, :], -_poly_mul(fl, frmu), _poly_mul(frmu, ol)))
    rate = base[None, :] * occ
    derivative = np.stack((base[None, :]*dl, base[None, :]*dr), axis=2)

    # Equal-energy spin flips.  The DOS factor cancels 1/tau(E), leaving dE.
    flip_rate_parts, flip_derivative_parts, flip_v_parts, flip_q_parts = [], [], [], []
    flip_arrays = events[-1]
    for island, (temperature, energy, measure) in enumerate(flip_arrays):
        up = _poly_sigmoid(np.ones_like(energy), energy, temperature, mu_series, island)
        down = _poly_sigmoid(-np.ones_like(energy), energy, temperature, mu_series, island)
        unit = np.zeros_like(up); unit[0] = 1.
        mup = _poly_mul(up, unit-up)
        mdown = -_poly_mul(down, unit-down)
        updown = _poly_mul(up, unit-down)
        downup = _poly_mul(down, unit-up)
        # d[f_up(1-f_down)]/dmu and d[f_down(1-f_up)]/dmu.
        dup = _poly_mul(mup, unit-down) - _poly_mul(up, mdown)
        ddown = _poly_mul(mdown, unit-up) - _poly_mul(down, mup)
        rates = np.concatenate((measure[None, :]*updown,
                                measure[None, :]*downup), axis=1)
        deriv = np.zeros((4, 2*len(energy), 2), dtype=float)
        deriv[:, :len(energy), island] = measure[None, :]*dup
        deriv[:, len(energy):, island] = measure[None, :]*ddown
        flip_rate_parts.append(rates)
        flip_derivative_parts.append(deriv)
        vj = np.zeros((2*len(energy), 2), dtype=float)
        vj[:len(energy), island] = -2.
        vj[len(energy):, island] = 2.
        flip_v_parts.append(vj)
        flip_q_parts.append(np.zeros(2*len(energy)))
    if flip_rate_parts:
        rate = np.concatenate([rate] + flip_rate_parts, axis=1)
        derivative = np.concatenate([derivative] + flip_derivative_parts, axis=1)
        v = np.concatenate([v] + flip_v_parts, axis=0)
        q = np.concatenate([q] + flip_q_parts)
    return rate, derivative, v, q


def _event_arrays(args, state, conductance, nodes, count, kwargs):
    """Build the conditional Poisson transitions in units of rate_conversion*G."""
    left, right, phase, amplitude, drive, rho, oscillator_temperature = args
    dl, dr, channels = prepare(left, right, phase, amplitude, drive, rho,
                               oscillator_temperature, nodes, count,
                               state['search_bound'], False, FACTORIZED,
                               kwargs.get('secondary_coupling', 0.),
                               kwargs.get('quadratic_coupling', 0.),
                               kwargs.get('momentum_coupling', 0.))
    energy, other, measure, sl, sr, creation, work, recoil, pf, pb, injection = channels
    channel_count, node_count = measure.shape
    flat = lambda a: np.broadcast_to(a, (channel_count, node_count)).reshape(-1)
    en, en_r = energy.reshape(-1), other.reshape(-1)
    spin_l, spin_r = flat(sl), flat(sr)
    is_pair = flat(creation).astype(bool)
    base_forward = measure * np.asarray(pf).reshape(channel_count, 1)
    base_reverse = measure * np.asarray(pb).reshape(channel_count, 1)
    base = np.concatenate((base_forward.reshape(-1), base_reverse.reshape(-1)))
    direction = np.concatenate((np.ones(en.size), -np.ones(en.size)))
    pair = np.concatenate((is_pair, is_pair))
    sl = np.concatenate((spin_l, spin_l)); sr = np.concatenate((spin_r, spin_r))
    forward_v = np.broadcast_to(injection[:, None, :], (channel_count, node_count, 2)).reshape(-1, 2)
    v = np.concatenate((forward_v, -forward_v), axis=0)
    harmonic = np.broadcast_to(work[:, None], (channel_count, node_count)).reshape(-1)
    q = np.concatenate((harmonic, -harmonic))
    event_energy = np.concatenate((en, en)); event_other = np.concatenate((en_r, en_r))
    temperatures = (left, right)

    ratio = QUASIPARTICLE_DENSITY/(NORMAL_POLARIZATION_TIME*RATE_CONVERSION*conductance)
    flips = []
    x, weights = leggauss(160)
    for temperature, delta, polarization in zip(temperatures, (dl, dr), state['mu']):
        upper = delta + max(80*temperature, 6*abs(polarization))
        e = delta + (x+1)*(upper-delta)/2
        w = ratio/2 * weights*(upper-delta)/2
        flips.append((temperature, e, w))
    events = (event_energy, event_other, sl, sr, pair, base, direction, v, q,
              temperatures, flips)
    return events


def _tilted_series(events, mu_series, p_series):
    rates, derivatives, v, q = _rate_series(events, mu_series)
    argument = np.zeros_like(rates)
    argument[1] = q
    for k in range(4):
        argument[k] += v @ p_series[k]
    exponential = np.empty_like(argument)
    exponential[0] = np.exp(argument[0])
    for k in range(1, 4):
        exponential[k] = sum(j*argument[j]*exponential[k-j]
                             for j in range(1, k+1))/k
    excess = exponential.copy(); excess[0] -= 1.
    rate_times_exp = _poly_mul(rates, exponential)
    rate_times_excess = _poly_mul(rates, excess)
    fp = np.stack([np.sum(rate_times_exp[k, :, None]*v, axis=0)
                   for k in range(4)])
    fmu = np.empty((4, 2), dtype=float)
    for j in range(2):
        fmu[:, j] = np.sum(_poly_mul(derivatives[:, :, j], excess), axis=1)
    h = np.sum(rate_times_excess, axis=1)
    return np.concatenate((fp, fmu), axis=1), h, rates, derivatives, v, q


def gate_statistics(e,conductance,nodes=192,count=64):
    args=oscillator_arguments(e)
    kwargs=dict(secondary_coupling=e.get('secondary_coupling',0.),
                quadratic_coupling=e.get('quadratic_coupling',0.),
                momentum_coupling=e.get('momentum_coupling',0.))
    state=stationary(*args,conductance,nodes=nodes,count=count,**kwargs)
    events=_event_arrays(args,state,conductance,nodes,count,kwargs)
    mu_series=np.zeros((4,2));mu_series[0]=state['mu']
    p_series=np.zeros((4,2))
    _,_,rates,derivatives,v,q=_tilted_series(events,mu_series,p_series)
    r0,dr0=rates[0],derivatives[0]
    work_gradient=np.sum(q[:,None]*dr0,axis=0)
    # The corrector removes the long-time drift caused by spin-population noise.
    response=np.linalg.solve(state['jacobian'].T,work_gradient)
    effective=q-v@response
    variance=float(r0@effective**2)
    shot=float(r0@q**2)
    scale=RATE_CONVERSION*conductance*KB**2*1e24
    return dict(noise=variance*scale,shot=shot*scale,
                response=response.tolist(),mu=state['mu'].tolist(),
                heat_derivative=work_gradient.tolist(),
                spin_jacobian=state['jacobian'].tolist())


mean_predict=predict_at
def predict_at(experiments,conductance):
    return np.array([gate_statistics(e,conductance)['noise'] if e.get('statistic','mean')=='gate_noise'
                     else mean_predict([e],conductance)[0] for e in experiments])

def gate_third(e,conductance,nodes=192,count=64):
    args=oscillator_arguments(e)
    kw=dict(secondary_coupling=e.get('secondary_coupling',0.),
            quadratic_coupling=e.get('quadratic_coupling',0.),
            momentum_coupling=e.get('momentum_coupling',0.))
    state=stationary(*args,conductance,nodes=nodes,count=count,**kw)
    events=_event_arrays(args,state,conductance,nodes,count,kw)
    mu_series=np.zeros((4,2));mu_series[0]=state['mu']
    p_series=np.zeros((4,2))
    _,_,rates,_,v,q=_tilted_series(events,mu_series,p_series)
    covariance=np.einsum('n,ni,nj->ij',rates[0],v,v)
    jac=state['jacobian']
    saddle=np.block([[jac,covariance],[np.zeros((2,2)),jac.T]])

    # Solve the stationary tilted Hamilton equations order by order.  This
    # includes the nonlinear rate response and spin-flip statistics at third order.
    for order in range(1,4):
        equations,_,_,_,_,_=_tilted_series(events,mu_series,p_series)
        correction=np.linalg.solve(saddle,-equations[order])
        mu_series[order]=correction[:2]
        p_series[order]=correction[2:]
    _, cumulant_series,_,_,_,_=_tilted_series(events,mu_series,p_series)
    third=6*cumulant_series[3]
    scale=RATE_CONVERSION*conductance*KB**3*1e36
    return dict(third=float(third*scale),screened_poisson=float(third*scale),
                count_response=p_series[1].tolist())


noise_predict=predict_at
def predict_at(experiments,conductance):
    return np.array([gate_third(e,conductance)['third'] if e.get('statistic','mean')=='gate_third'
                     else noise_predict([e],conductance)[0] for e in experiments])
