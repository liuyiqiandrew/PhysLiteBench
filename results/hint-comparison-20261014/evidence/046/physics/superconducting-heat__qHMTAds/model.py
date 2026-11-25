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
        """Fit the (normal-state, two-spin) contact conductance.

        The supplied calibration is deliberately in the unmodulated limit.  In
        that limit the prediction is linear in the conductance, so doing the
        weighted linear fit directly is both quicker and more numerically
        stable than repeatedly solving the spin kinetic equation.
        """
        if not records:
            raise ValueError('at least one calibration record is required')

        coefficients = []
        values = []
        sigmas = []
        original = True
        for record in records:
            inp = record['input']
            if (inp.get('statistic', 'mean') != 'mean' or
                    inp.get('amplitude', 0.) != 0. or
                    inp.get('oscillator_coupling', 0.) != 0. or
                    inp.get('secondary_coupling', 0.) != 0. or
                    inp.get('quadratic_coupling', 0.) != 0. or
                    inp.get('momentum_coupling', 0.) != 0.):
                original = False
                break
            coefficients.append(power_per_conductance(
                inp['left_temperature'], inp['right_temperature'],
                inp['phase'], 0., inp.get('drive_energy', .3)))
            values.append(record['value'])
            sigmas.append(record['sigma'])

        if original:
            coefficients = np.asarray(coefficients, dtype=float)
            values = np.asarray(values, dtype=float)
            sigmas = np.asarray(sigmas, dtype=float)
            if np.any(sigmas <= 0) or not np.isfinite(sigmas).all():
                raise ValueError('calibration uncertainties must be positive and finite')
            weights = 1. / sigmas**2
            denominator = np.sum(weights * coefficients**2)
            if denominator <= 0 or not np.isfinite(denominator):
                raise ValueError('calibration records do not constrain conductance')
            fitted = np.sum(weights * coefficients * values) / denominator
        else:
            # This fallback keeps fit useful for a mixed, explicitly supplied
            # calibration set while respecting the documented bounds.
            def objective(g):
                prediction = predict_at([r['input'] for r in records], float(g))
                residual = (prediction - np.asarray([r['value'] for r in records])) / np.asarray(
                    [r['sigma'] for r in records])
                return float(residual @ residual)
            from scipy.optimize import minimize_scalar
            result = minimize_scalar(objective, bounds=(20., 80.), method='bounded',
                                     options={'xatol': 1e-8})
            fitted = result.x

        self.conductance = float(np.clip(fitted, 20., 80.))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.conductance)


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
    momentum=1j**np.arange(states)[:,None]*v
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
            # The contrast preparation is phi+pi, not a 2*pi wrapping of phi.
            other=dict(e,phase=(e['phase']+2.*np.pi)%(2.*np.pi)-np.pi)
            value=(value-mean_heat(other,conductance))/2
        values.append(value)
    return np.array(values)


FACTORIZED = False

@lru_cache(maxsize=1024)
def _spin_susceptibility(temperature, delta, polarization, nodes=192):
    """Return dS/dmu and d2S/dmu2 per island-size factor.

    The change of variables E=delta+z**2 removes the BCS edge singularity.
    """
    x, weights = leggauss(nodes)
    zmax = np.sqrt(max(80. * temperature, 6. * abs(polarization), 1e-12))
    z = (x + 1.) * zmax / 2.
    energy = delta + z*z
    edge_measure = np.zeros_like(energy)
    denominator = np.sqrt(energy + delta)
    np.divide(2. * energy, denominator, out=edge_measure, where=denominator > 0)
    measure = weights * zmax / 2. * edge_measure
    up = expit((polarization - energy) / temperature)
    down = expit((-polarization - energy) / temperature)
    up1 = up * (1. - up) / temperature
    down1 = down * (1. - down) / temperature
    up2 = up * (1. - up) * (1. - 2. * up) / temperature**2
    down2 = down * (1. - down) * (1. - 2. * down) / temperature**2
    return (QUASIPARTICLE_DENSITY * np.sum(measure * (up1 + down1)),
            QUASIPARTICLE_DENSITY * np.sum(measure * (up2 - down2)))


def _gate_events(e, conductance, nodes=192, count=64):
    """Build conditional event rates and their first two state derivatives.

    The state coordinate here is the spin excess S in one macroscopic-size
    copy of the apparatus.  Thus a tunnelling or flip event has a fixed state
    jump q, which is the convenient coordinate for the large-size tilted
    generator.
    """
    args = oscillator_arguments(e)
    kwargs = dict(secondary_coupling=e.get('secondary_coupling', 0.),
                  quadratic_coupling=e.get('quadratic_coupling', 0.),
                  momentum_coupling=e.get('momentum_coupling', 0.))
    state = stationary(*args, conductance, nodes=nodes, count=count, **kwargs)
    left, right, phase, amplitude, drive = args[:5]
    dl, dr, channels = prepare(*args, nodes=nodes, count=count,
                               window=state['search_bound'], **kwargs)
    energy, other, measure, sl, sr, pair, work, recoil, pf, pb, injection = channels
    pair = pair[:, 0].astype(bool)
    sl, sr = sl[:, 0], sr[:, 0]
    nchan, nquad = energy.shape
    mu = state['mu']
    f = expit((sl[:, None] * mu[0] - energy) / left)
    g = expit((sr[:, None] * mu[1] - other) / right)
    f1 = sl[:, None] * f * (1. - f) / left
    g1 = sr[:, None] * g * (1. - g) / right
    f2 = f * (1. - f) * (1. - 2. * f) / left**2
    g2 = g * (1. - g) * (1. - 2. * g) / right**2

    # Forward and reverse occupation factors and their mu derivatives.
    if np.any(pair):
        forward = np.where(pair[:, None], (1. - f) * (1. - g), f * (1. - g))
        backward = np.where(pair[:, None], f * g, g * (1. - f))
        forward_l = np.where(pair[:, None], -f1 * (1. - g), f1 * (1. - g))
        forward_r = np.where(pair[:, None], -(1. - f) * g1, -f * g1)
        backward_l = np.where(pair[:, None], f1 * g, -g * f1)
        backward_r = np.where(pair[:, None], f * g1, g1 * (1. - f))
        forward_ll = np.where(pair[:, None], -f2 * (1. - g), f2 * (1. - g))
        forward_rr = np.where(pair[:, None], -(1. - f) * g2, -f * g2)
        forward_lr = np.where(pair[:, None], f1 * g1, -f1 * g1)
        backward_ll = np.where(pair[:, None], f2 * g, -g * f2)
        backward_rr = np.where(pair[:, None], f * g2, -f * g2)
        backward_lr = np.where(pair[:, None], f1 * g1, -f1 * g1)
    # The expressions above are written for both channel families at once;
    # the pair mask selects the pair-creation or quasiparticle-transfer form.
    rf = measure * np.where(pair[:, None], pf * forward, pf * forward)
    rb = measure * np.where(pair[:, None], pb * backward, pb * backward)
    gf = np.stack((measure * pf * forward_l, measure * pf * forward_r), axis=-1)
    gb = np.stack((measure * pb * backward_l, measure * pb * backward_r), axis=-1)
    hf = np.zeros((nchan, nquad, 2, 2))
    hb = np.zeros_like(hf)
    hf[..., 0, 0] = measure * pf * forward_ll
    hf[..., 1, 1] = measure * pf * forward_rr
    hf[..., 0, 1] = hf[..., 1, 0] = measure * pf * forward_lr
    hb[..., 0, 0] = measure * pb * backward_ll
    hb[..., 1, 1] = measure * pb * backward_rr
    hb[..., 0, 1] = hb[..., 1, 0] = measure * pb * backward_lr

    def flatten_channel(values):
        return np.asarray(values).reshape(nchan * nquad, *np.asarray(values).shape[2:])

    rf, rb = rf.ravel(), rb.ravel()
    gf, gb = gf.reshape(-1, 2), gb.reshape(-1, 2)
    hf, hb = hf.reshape(-1, 2, 2), hb.reshape(-1, 2, 2)
    qchan = np.broadcast_to(injection[:, None, :], (nchan, nquad, 2)).reshape(-1, 2)
    wchan = np.broadcast_to(work[:, None], (nchan, nquad)).ravel()
    rates = np.concatenate((rf, rb))
    grad_mu = np.concatenate((gf, gb))
    hess_mu = np.concatenate((hf, hb))
    q = np.concatenate((qchan, -qchan))
    marks = np.concatenate((wchan, -wchan))

    # Equal-energy stochastic spin flips.  The BCS DOS cancels the
    # superconducting enhancement in 1/tau_j(E), leaving an integral over dE.
    ratio = QUASIPARTICLE_DENSITY / (NORMAL_POLARIZATION_TIME * RATE_CONVERSION * conductance)
    flip_rates, flip_grad, flip_hess, flip_q, flip_marks = [], [], [], [], []
    for j, (temp, delta, pol) in enumerate(((left, dl, mu[0]), (right, dr, mu[1]))):
        x, weights = leggauss(max(96, nodes // 2))
        upper = delta + max(80. * temp, 6. * abs(pol))
        ee = delta + (x + 1.) * (upper - delta) / 2.
        measure_flip = weights * (upper - delta) / 2. * ratio / 2.
        up, down = expit((pol - ee) / temp), expit((-pol - ee) / temp)
        up1 = up * (1. - up) / temp
        down1 = -down * (1. - down) / temp
        up2 = up * (1. - up) * (1. - 2. * up) / temp**2
        down2 = down * (1. - down) * (1. - 2. * down) / temp**2
        ap = measure_flip * up * (1. - down)
        am = measure_flip * down * (1. - up)
        ap1 = measure_flip * (up1 * (1. - down) - up * down1)
        am1 = measure_flip * (down1 * (1. - up) - down * up1)
        ap2 = measure_flip * (up2 * (1. - down) - 2. * up1 * down1 - up * down2)
        am2 = measure_flip * (down2 * (1. - up) - 2. * down1 * up1 - down * up2)
        for rate, first, second, jump in ((ap, ap1, ap2, -2.), (am, am1, am2, 2.)):
            rates = np.concatenate((rates, rate))
            gadd = np.zeros((len(rate), 2)); gadd[:, j] = first
            hadd = np.zeros((len(rate), 2, 2)); hadd[:, j, j] = second
            grad_mu = np.concatenate((grad_mu, gadd))
            hess_mu = np.concatenate((hess_mu, hadd))
            flip_q.append(np.column_stack((np.full(len(rate), jump) if j == 0 else np.zeros(len(rate)),
                                           np.full(len(rate), jump) if j == 1 else np.zeros(len(rate)))))
            flip_marks.append(np.zeros(len(rate)))
    if flip_q:
        q = np.concatenate((q, *flip_q))
        marks = np.concatenate((marks, *flip_marks))

    chi = np.empty(2); chi2 = np.empty(2)
    for j, (temp, delta, pol) in enumerate(((left, dl, mu[0]), (right, dr, mu[1]))):
        chi[j], chi2[j] = _spin_susceptibility(temp, delta, pol)
    inverse = 1. / chi
    grad = grad_mu * inverse[None, :]
    hess = hess_mu * inverse[None, :, None] * inverse[None, None, :]
    hess[:, 0, 0] -= grad_mu[:, 0] * chi2[0] * inverse[0]**3
    hess[:, 1, 1] -= grad_mu[:, 1] * chi2[1] * inverse[1]**3
    return state, rates, grad, hess, q, marks, mu


def _cumulants_from_events(rates, gradients, hessians, jumps, marks):
    """Second and third SCGF derivatives for a stationary jump process."""
    # Remove exponentially irrelevant roundoff-sized negative quadrature
    # contributions, while retaining all physical interference terms.
    rates = np.maximum(np.asarray(rates, dtype=float), 0.)
    gradients = np.asarray(gradients, dtype=float)
    hessians = np.asarray(hessians, dtype=float)
    jumps = np.asarray(jumps, dtype=float)
    marks = np.asarray(marks, dtype=float)
    jmat = np.einsum('ni,nj->ij', gradients, jumps)
    diffusion = np.einsum('n,ni,nj->ij', rates, jumps, jumps)
    mark_gradient = np.einsum('ni,n->i', gradients, marks)
    jump_mark = np.einsum('n,ni->i', rates * marks, jumps)
    hess_mark = np.einsum('n,nij->ij', marks, hessians)

    A = np.zeros((4, 4))
    A[:2, 2:] = jmat
    A[2:, :2] = jmat.T
    A[2:, 2:] = diffusion
    hz = np.concatenate((mark_gradient, jump_mark))
    try:
        z1 = -np.linalg.solve(A, hz)
    except np.linalg.LinAlgError:
        z1 = -np.linalg.pinv(A) @ hz

    # Third derivatives of H(x,p,k) = sum r(x)[exp(p.q+k.w)-1].
    tensor = np.zeros((5, 5, 5))
    hxxp = np.einsum('nij,nk->ijk', hessians, jumps)
    hxxk = np.einsum('nij,n->ij', hessians, marks)
    hxpp = np.einsum('ni,nj,nk->ijk', gradients, jumps, jumps)
    hxpk = np.einsum('ni,nj,n->ij', gradients, jumps, marks)
    hxkk = np.einsum('ni,n->i', gradients, marks**2)
    hppp = np.einsum('n,ni,nj,nk->ijk', rates, jumps, jumps, jumps)
    hppk = np.einsum('n,ni,nj,n->ij', rates, jumps, jumps, marks)
    hpkk = np.einsum('n,ni,n->i', rates, jumps, marks**2)
    tensor[:2, :2, 2:4] = hxxp
    tensor[:2, 2:4, :2] = hxxp.transpose(0, 2, 1)
    tensor[2:4, :2, :2] = hxxp.transpose(2, 0, 1)
    tensor[:2, :2, 4] = hxxk
    tensor[:2, 4, :2] = hxxk
    tensor[4, :2, :2] = hxxk
    tensor[:2, 2:4, 2:4] = hxpp
    tensor[2:4, :2, 2:4] = hxpp.transpose(1, 0, 2)
    tensor[2:4, 2:4, :2] = hxpp.transpose(1, 2, 0)
    tensor[:2, 2:4, 4] = hxpk
    tensor[:2, 4, 2:4] = hxpk
    tensor[2:4, :2, 4] = hxpk.T
    tensor[2:4, 4, :2] = hxpk.T
    tensor[4, :2, 2:4] = hxpk
    tensor[4, 2:4, :2] = hxpk.T
    tensor[:2, 4, 4] = hxkk
    tensor[4, :2, 4] = hxkk
    tensor[4, 4, :2] = hxkk
    tensor[2:4, 2:4, 2:4] = hppp
    tensor[2:4, 2:4, 4] = hppk
    tensor[2:4, 4, 2:4] = hppk
    tensor[4, 2:4, 2:4] = hppk
    tensor[2:4, 4, 4] = hpkk
    tensor[4, 2:4, 4] = hpkk
    tensor[4, 4, 2:4] = hpkk
    tensor[4, 4, 4] = np.sum(rates * marks**3)

    z2_rhs = (np.einsum('ijk,j,k->i', tensor[:4, :4, :4], z1, z1) +
              2. * np.einsum('ij,j->i', tensor[:4, :4, 4], z1) +
              tensor[:4, 4, 4])
    try:
        z2 = -np.linalg.solve(A, z2_rhs)
    except np.linalg.LinAlgError:
        z2 = -np.linalg.pinv(A) @ z2_rhs
    second = float(np.sum(rates * marks**2) + hz @ z1)
    third = float(tensor[4, 4, 4] +
                  2. * np.dot(tensor[4, 4, :4], z1) +
                  np.einsum('ij,i,j->', tensor[4, :4, :4], z1, z1) +
                  np.dot(hz, z2))
    return second, third, z1, z2


def gate_statistics(e, conductance, nodes=192, count=64):
    if e.get('amplitude', 0.) == 0.:
        return dict(noise=0., shot=0., response=[0., 0.], mu=[0., 0.])
    state, rates, gradients, hessians, jumps, marks, mu = _gate_events(
        e, conductance, nodes, count)
    second, _, z1, _ = _cumulants_from_events(rates, gradients, hessians, jumps, marks)
    scale = RATE_CONVERSION * conductance * KB**2 * 1e24
    return dict(noise=second * scale, shot=float(np.sum(rates * marks**2) * scale),
                response=(-z1[2:]).tolist(), mu=mu.tolist(),
                spin_jacobian=state['jacobian'].tolist())


def gate_third(e, conductance, nodes=192, count=64):
    if e.get('amplitude', 0.) == 0.:
        return dict(third=0., screened_poisson=0., count_response=[0., 0.])
    state, rates, gradients, hessians, jumps, marks, mu = _gate_events(
        e, conductance, nodes, count)
    _, third, z1, _ = _cumulants_from_events(rates, gradients, hessians, jumps, marks)
    scale = RATE_CONVERSION * conductance * KB**3 * 1e36
    return dict(third=third * scale, screened_poisson=0.,
                count_response=z1[2:].tolist(), mu=mu.tolist())


_mean_predict = predict_at


def predict_at(experiments, conductance):
    values = []
    for experiment in experiments:
        statistic = experiment.get('statistic', 'mean')
        if statistic == 'gate_noise':
            values.append(gate_statistics(experiment, conductance)['noise'])
        elif statistic == 'gate_third':
            values.append(gate_third(experiment, conductance)['third'])
        else:
            values.append(_mean_predict([experiment], conductance)[0])
    return np.asarray(values, dtype=float)
