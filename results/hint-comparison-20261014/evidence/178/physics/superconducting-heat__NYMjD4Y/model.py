"""Thermal quantum phase oscillator and stationary spin-resolved BCS heat."""
from functools import lru_cache
from math import comb
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


from scipy.optimize import least_squares, minimize_scalar

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
        """Fit the one unknown scale parameter from the calibration records.

        The supplied calibration is deliberately in the unmodulated limit, in
        which the prediction is linear in the normal conductance.  Keeping the
        small weighted least-squares calculation here (rather than fitting a
        hidden spin or oscillator parameter) also makes ``fit`` behave sensibly
        for a user-supplied record set containing the optional fields.
        """
        if not records:
            raise ValueError('At least one calibration record is required')

        def prediction(record, conductance):
            value = mean_heat(record['input'], conductance)
            if record['input'].get('statistic', 'mean') == 'phase_contrast':
                value = phase_contrast(record['input'], conductance)
            return value

        target = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if np.any(~np.isfinite(target)) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError('Calibration values and uncertainties must be finite')

        def objective(conductance):
            values = np.asarray([prediction(record, conductance) for record in records])
            return float(np.sum(((values - target) / sigma) ** 2))

        # The calibration records are normally exactly linear in conductance;
        # bounded minimisation also covers optional records with a stationary
        # spin state without making that assumption part of the API.
        result = minimize_scalar(objective, bounds=(20., 80.), method='bounded',
                                 options={'xatol': 1e-10, 'maxiter': 100})
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError('Conductance fit did not converge')
        self.conductance = float(np.clip(result.x, 20., 80.))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.conductance)


from scipy.special import gammaln, logsumexp
from scipy.linalg import eigh_tridiagonal,expm
OSCILLATOR_ENERGY = 24.
THIRD_SIZE_FACTOR = 1_000_000.


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


def phase_contrast(e, conductance):
    """Mean heat contrast, allowing the two preparations to equilibrate separately."""
    # Wrap phi+pi into [-pi, pi].  The extra 2*pi before the modulo is
    # intentional: modulo after adding only pi would wrap back to phi.
    other = dict(e, phase=(e['phase'] + 2 * np.pi) % (2 * np.pi) - np.pi)
    return (mean_heat(e, conductance) - mean_heat(other, conductance)) / 2.


FACTORIZED = False

def gate_statistics(e,conductance,nodes=192,count=64):
    work_noise=True
    args=oscillator_arguments(e);kwargs=dict(secondary_coupling=e.get('secondary_coupling',0.),quadratic_coupling=e.get('quadratic_coupling',0.),momentum_coupling=e.get('momentum_coupling',0.))
    state=stationary(*args,conductance,nodes=nodes,count=count,**kwargs);mu=state['mu'];left,right=args[:2]
    dl,dr,ch=prepare(*args,nodes=nodes,count=count,window=state['search_bound'],**kwargs)
    energy,other,measure,sl,sr,pair,work,recoil,pf,pb,injection=ch
    fl=expit((sl*mu[0]-energy)/left);fr=expit((sr*mu[1]-other)/right);df=fl*(1-fl)/left;dg=fr*(1-fr)/right
    forward=np.where(pair,pf*(1-fl)*(1-fr),pf*fl*(1-fr));backward=np.where(pair,pb*fl*fr,pb*fr*(1-fl));shot=measure*(forward+backward)
    dleft=np.where(pair,-sl*df,sl*df)*(pf*(1-fr)+pb*fr)
    dright=-sr*dg*np.where(pair,pf*(1-fl)+pb*fl,pf*fl+pb*(1-fl))
    loss=np.broadcast_to(work[:,None],energy.shape) if work_noise else np.where(pair,-energy,energy);hprime=np.array([np.sum(measure*loss*dleft),np.sum(measure*loss*dright)])
    response=hprime@np.linalg.inv(state['jacobian'])
    ratio=QUASIPARTICLE_DENSITY/(NORMAL_POLARIZATION_TIME*RATE_CONVERSION*conductance)
    flips=[]
    for temp,delta,pol in zip([left,right],[dl,dr],mu):
        current,slope=spin_loss(temp,delta,pol)
        intensity=current/np.tanh(pol/temp) if abs(pol/temp)>1e-7 else temp*slope
        flips.append(2*ratio*intensity)
    raw=float(np.sum(shot*loss**2));correct=float(np.sum(shot*(loss-(injection@response)[:,None])**2)+np.sum(np.array(flips)*response**2))
    scale=RATE_CONVERSION*conductance*KB**2*1e24
    return dict(noise=correct*scale,shot=raw*scale,response=response.tolist(),mu=mu.tolist(),
                heat_derivative=hprime.tolist(),spin_jacobian=state['jacobian'].tolist(),flip_noise=flips,
                work_spin_identity_error=float(np.max(abs(work-args[4]/2*injection.sum(axis=1)))))
def _third_rate_context(e, conductance, nodes, count):
    """Build the local density-dependent jump process used by gate_third.

    The state is y_j=S_j/nu.  Under the Lambda scaling in the README, a jump
    changes y by q/(Lambda*nu), while every event intensity is Lambda times
    the intensity returned here.  Grouping events by (spin jump, gate work)
    keeps the subsequent polynomial calculation small without discarding any
    conditional-rate dependence.
    """
    args = oscillator_arguments(e)
    kw = dict(secondary_coupling=e.get('secondary_coupling', 0.),
              quadratic_coupling=e.get('quadratic_coupling', 0.),
              momentum_coupling=e.get('momentum_coupling', 0.))
    state = stationary(*args, conductance, nodes=nodes, count=count, **kw)
    dl, dr, channels = prepare(*args, nodes=nodes, count=count,
                               window=state['search_bound'], **kw)
    energy, other, measure, sl, sr, pair, work, recoil, pf, pb, injection = channels
    temperature = args[:2]

    # Two quadratures are enough for the smooth local derivatives.  The upper
    # limit follows the same exponentially safe cutoff used by gate_statistics.
    flip_quadrature = []
    for temp, delta, pol in zip(temperature, (dl, dr), state['mu']):
        x, weights = leggauss(128)
        upper = delta + max(80 * temp, 6 * abs(pol))
        ee = delta + (x + 1) * (upper - delta) / 2
        prefactor = weights * (upper - delta) / 2 * QUASIPARTICLE_DENSITY / (2 * NORMAL_POLARIZATION_TIME)
        flip_quadrature.append((ee, prefactor))

    # All tunnel groups which can occur, including the two directions.  The
    # keys use integer spin changes; the conversion to y-space is applied by
    # the generator below.
    tunnel_keys = np.concatenate((
        np.column_stack((injection.astype(int), work)),
        np.column_stack((-injection.astype(int), -work))))
    tunnel_keys = np.unique(tunnel_keys, axis=0)
    keys = [tuple(row) for row in tunnel_keys]
    for j in range(2):
        for sign in (-2, 2):
            key = [0., 0., 0.]
            key[j] = sign
            key = tuple(key)
            if key not in keys:
                keys.append(key)
    key_index = {key: i for i, key in enumerate(keys)}

    return dict(args=args, kw=kw, state=state, dl=dl, dr=dr,
                channels=channels, keys=keys, key_index=key_index,
                flip_quadrature=flip_quadrature,
                y0=np.array([spin_loss(t, d, m)[0]
                             for t, d, m in zip(temperature, (dl, dr), state['mu'])]))


def _third_rates(context, mu, conductance):
    """Return physical (per-second) grouped rates at a pair of spin potentials."""
    args = context['args']; tl, tr = args[:2]
    energy, other, measure, sl, sr, pair, work, recoil, pf, pb, injection = context['channels']
    f = expit((sl * mu[0] - energy) / tl)
    g = expit((sr * mu[1] - other) / tr)
    forward = measure * np.where(pair, pf * (1 - f) * (1 - g), pf * f * (1 - g))
    backward = measure * np.where(pair, pb * f * g, pb * g * (1 - f))
    rates = np.zeros(len(context['keys']))
    factor = RATE_CONVERSION * conductance

    # There are only a few distinct keys, so this vectorised grouping avoids
    # a Python loop over the oscillator and energy quadrature points.
    raw = np.concatenate((
        np.column_stack((injection.astype(int), work)),
        np.column_stack((-injection.astype(int), -work))))
    # Energy quadrature points share the same jump and work mark; first
    # integrate their intensities before grouping the discrete event types.
    values = factor * np.concatenate((forward.sum(axis=1), backward.sum(axis=1)))
    unique, inverse = np.unique(raw, axis=0, return_inverse=True)
    grouped = np.bincount(inverse, weights=values, minlength=len(unique))
    for row, value in zip(unique, grouped):
        rates[context['key_index'][tuple(row)]] += value

    for j, (temp, quad) in enumerate(zip((tl, tr), context['flip_quadrature'])):
        ee, prefactor = quad
        up = expit((mu[j] - ee) / temp) * (1 - expit((-mu[j] - ee) / temp))
        down = expit((-mu[j] - ee) / temp) * (1 - expit((mu[j] - ee) / temp))
        for sign, value in ((-2., prefactor @ up), (2., prefactor @ down)):
            key = [0., 0., 0.]
            key[j] = sign
            rates[context['key_index'][tuple(key)]] += value
    return rates


def _third_polynomial_matrices(context, conductance, nodes, count):
    """Construct the local tilted jump generator through third order in tilt.

    A polynomial representation of the generator is used only as a numerical
    Poisson-equation solver.  Taking Lambda large and then retaining its
    leading coefficient implements the ordered limits specified in the README.
    In particular, the rate Taylor terms are retained; omitting them gives the
    linearly-screened-mark approximation that is insufficient for a third
    cumulant.
    """
    degree = 6
    size_factor = THIRD_SIZE_FACTOR
    epsilon = size_factor ** -0.5
    scale = .30
    exponents = [(i, j) for i in range(degree + 1)
                 for j in range(degree + 1 - i)]
    index = {power: i for i, power in enumerate(exponents)}
    dimension = len(exponents)
    grid = np.linspace(-1., 1., degree + 3)
    points = np.array([(x, y) for x in grid for y in grid])
    design = np.asarray([[x ** i * y ** j for i, j in exponents]
                         for x, y in points])

    mu0 = context['state']['mu'].copy()
    y0 = context['y0']
    temperatures = context['args'][:2]
    deltas = (context['dl'], context['dr'])
    rate_values = []
    for point in points:
        target = y0 + epsilon * scale * point
        mu = mu0.copy()
        for j in range(2):
            for _ in range(16):
                value, derivative = spin_loss(temperatures[j], deltas[j], mu[j])
                step = (target[j] - value) / max(derivative, 1e-300)
                mu[j] += step
                if abs(step) < 2e-14:
                    break
        rate_values.append(_third_rates(context, mu, conductance))
    coefficients = np.linalg.lstsq(design, np.asarray(rate_values), rcond=None)[0]

    def multiply(left, right):
        result = np.zeros(dimension)
        for il, (i, j) in enumerate(exponents):
            if abs(left[il]) < 1e-300:
                continue
            for ir, (k, l) in enumerate(exponents):
                if i + k + j + l <= degree:
                    result[index[(i + k, j + l)]] += left[il] * right[ir]
        return result

    def matrix(order):
        result = np.zeros((dimension, dimension))
        for group, key in enumerate(context['keys']):
            q0, q1, mark = key
            jump = (epsilon * q0 / QUASIPARTICLE_DENSITY / scale,
                    epsilon * q1 / QUASIPARTICLE_DENSITY / scale)
            rate = coefficients[:, group]
            for column, (a, b) in enumerate(exponents):
                shifted = np.zeros(dimension)
                for i in range(a + 1):
                    for j in range(b + 1):
                        shifted[index[(i, j)]] = (
                            comb(a, i) * comb(b, j) *
                            jump[0] ** (a - i) * jump[1] ** (b - j))
                if order == 0:
                    shifted[index[(a, b)]] -= 1.
                else:
                    shifted *= mark ** order
                result[:, column] += size_factor * multiply(rate, shifted)
        return result

    return [matrix(order) for order in range(4)]


def gate_third(e, conductance, nodes=192, count=64):
    amplitude = e.get('amplitude', 0.)
    if not amplitude:
        return dict(third=0., screened_poisson=0., count_response=[0., 0.])

    context = _third_rate_context(e, conductance, nodes, count)
    state = context['state']
    energy, other, measure, sl, sr, pair, work, recoil, pf, pb, injection = context['channels']
    tl, tr = context['args'][:2]
    f = expit((sl * state['mu'][0] - energy) / tl)
    g = expit((sr * state['mu'][1] - other) / tr)
    forward = measure * np.where(pair, pf * (1 - f) * (1 - g), pf * f * (1 - g))
    backward = measure * np.where(pair, pb * f * g, pb * g * (1 - f))
    df, dg = f * (1 - f) / tl, g * (1 - g) / tr
    dleft = np.where(pair, -sl * df, sl * df) * (pf * (1 - g) + pb * g)
    dright = -sr * dg * np.where(pair, pf * (1 - f) + pb * f,
                                  pf * f + pb * (1 - f))
    work_derivative = np.array([
        np.sum(measure * work[:, None] * dleft),
        np.sum(measure * work[:, None] * dright)])
    screened_response = -np.linalg.solve(state['jacobian'].T, work_derivative)
    screened_mark = work + injection @ screened_response
    screened = float(np.sum((forward - backward) * screened_mark[:, None] ** 3))
    ratio = QUASIPARTICLE_DENSITY / (NORMAL_POLARIZATION_TIME * RATE_CONVERSION * conductance)
    for temp, delta, pol, response in zip((tl, tr), (context['dl'], context['dr']),
                                           state['mu'], screened_response):
        current, _ = spin_loss(temp, delta, pol)
        # Third cumulants use the signed forward-minus-backward flip rate.
        screened += (ratio * current / 2.) * (-2 * response) ** 3
    screened_value = screened * RATE_CONVERSION * conductance * KB ** 3 * 1e36

    matrices = _third_polynomial_matrices(context, conductance, nodes, count)
    generator = matrices[0]
    dimension = generator.shape[0]
    constant = np.zeros(dimension); constant[0] = 1.

    # The left zero mode supplies stationary averaging for the Poisson
    # equations.  The constant coefficient is fixed to one, matching the
    # normalization of the right zero mode.
    eigenvalues, eigenvectors = np.linalg.eig(generator.T)
    left = np.real(eigenvectors[:, np.argmin(abs(eigenvalues))])
    left /= left @ constant

    solutions = [constant]
    cumulants = []
    for order in range(1, 4):
        rhs = np.zeros(dimension)
        for previous in range(1, order):
            rhs += comb(order, previous) * cumulants[previous - 1] * solutions[order - previous]
        for derivative in range(1, order + 1):
            rhs -= (comb(order, derivative) * matrices[derivative] @
                    solutions[order - derivative])

        augmented = np.zeros((dimension + 1, dimension + 1))
        augmented[:dimension, :dimension] = generator
        augmented[:dimension, dimension] = -constant
        augmented[dimension, :dimension] = left
        solved = np.linalg.solve(augmented, np.r_[rhs, 0.])
        solutions.append(solved[:dimension])
        cumulants.append(float(solved[dimension]))

    # The matrices use kelvin marks and rates in s^-1.  Convert K^3/s to the
    # requested pW^3*s^2 (1 pW^3*s^2 = 10^-36 J^3/s).
    value = cumulants[2] / THIRD_SIZE_FACTOR * KB ** 3 * 1e36
    return dict(third=value, screened_poisson=screened_value,
                count_response=screened_response.tolist())


def _ordinary_predict(experiments, conductance):
    values = []
    for experiment in experiments:
        if experiment.get('statistic', 'mean') == 'phase_contrast':
            values.append(phase_contrast(experiment, conductance))
        else:
            values.append(mean_heat(experiment, conductance))
    return np.asarray(values, dtype=float)


def predict_at(experiments, conductance):
    values = []
    for experiment in experiments:
        statistic = experiment.get('statistic', 'mean')
        if statistic == 'gate_noise':
            values.append(gate_statistics(experiment, conductance)['noise'])
        elif statistic == 'gate_third':
            values.append(gate_third(experiment, conductance)['third'])
        elif statistic == 'phase_contrast':
            values.append(phase_contrast(experiment, conductance))
        else:
            values.append(mean_heat(experiment, conductance))
    return np.asarray(values, dtype=float)


mean_predict = _ordinary_predict
noise_predict = predict_at
