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
        """Fit the one (and only) apparatus parameter by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')
        inputs = [r['input'] for r in records]
        observed = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if (not np.isfinite(observed).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError('calibration values and sigmas must be finite, with sigma > 0')

        def residual(value):
            return (predict_at(inputs, float(value[0])) - observed) / sigma

        result = least_squares(residual, np.array([50.0]), bounds=(20., 80.),
                               xtol=1e-12, ftol=1e-12, gtol=1e-12)
        self.conductance = float(result.x[0])
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
            other=dict(e,phase=(e['phase']+2*np.pi)%(2*np.pi)-np.pi)
            value=(value-mean_heat(other,conductance))/2
        values.append(value)
    return np.array(values)


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


mean_predict=predict_at
def predict_at(experiments,conductance):
    return np.array([gate_statistics(e,conductance)['noise'] if e.get('statistic','mean')=='gate_noise'
                     else mean_predict([e],conductance)[0] for e in experiments])

def _counting_terms(e, conductance, nodes=192, count=64):
    """Return the elementary jump processes for the macroscopic generator.

    Rates here are rates for one member of the Lambda-scaled family.  The
    spin increments are integers (rather than their vanishing intensive
    increments); this is the convenient WKB/counting-field normalization.
    """
    args = oscillator_arguments(e)
    kw = dict(secondary_coupling=e.get('secondary_coupling', 0.),
              quadratic_coupling=e.get('quadratic_coupling', 0.),
              momentum_coupling=e.get('momentum_coupling', 0.))
    state = stationary(*args, conductance, nodes=nodes, count=count, **kw)
    tl, tr = args[:2]
    dl, dr, ch = prepare(*args, nodes=nodes, count=count,
                         window=state['search_bound'], **kw)
    energy, other, measure, sl, sr, pair, work, recoil, pf, pb, injection = ch

    # Keep rates rather than net currents.  The two rows of each elementary
    # tunnelling channel are its forward and reverse Poisson processes.
    def tunnel_rates(mu):
        f = expit((sl*mu[0]-energy)/tl)
        g = expit((sr*mu[1]-other)/tr)
        forward = np.where(pair, pf*(1-f)*(1-g), pf*f*(1-g))
        backward = np.where(pair, pb*f*g, pb*g*(1-f))
        unit = RATE_CONVERSION*conductance*measure
        return unit*forward, unit*backward

    # In a spin flip the BCS density of states cancels the coherence-factor
    # enhancement of tau(E), leaving this particularly simple energy rate.
    flip_grids = []
    x, w = leggauss(176)
    for index, (temp, delta) in enumerate(((tl, dl), (tr, dr))):
        upper = np.sqrt(90.*temp + 6.*state['search_bound'])
        z = (x+1.)*upper/2.
        ee = delta + z*z
        weight = w*upper*z                         # dE quadrature weight
        flip_grids.append((index, temp, ee,
                           weight*QUASIPARTICLE_DENSITY/(2.*NORMAL_POLARIZATION_TIME)))

    return dict(mu=np.array(state['mu'], dtype=float), jacobian=state['jacobian'],
                tunnel_rates=tunnel_rates, injection=injection, work=work,
                flip_grids=flip_grids,
                rate_scale=RATE_CONVERSION*conductance)


def _counting_hamiltonian(terms, mu, q, theta):
    """Large-deviation Hamiltonian for spin and gate-work jumps (K/s)."""
    forward, backward = terms['tunnel_rates'](mu)
    injection = terms['injection']
    argument = (injection @ q)[:, None] + theta*terms['work'][:, None]
    value = np.sum(forward*np.expm1(argument) + backward*np.expm1(-argument))
    for index, temp, energy, weight in terms['flip_grids']:
        up = expit((mu[index]-energy)/temp)
        down = expit((-mu[index]-energy)/temp)
        # up -> down has Delta S=-2, and down -> up has Delta S=+2.
        down_rate = weight*up*(1-down)
        up_rate = weight*down*(1-up)
        a = 2*q[index]
        value += np.sum(down_rate*np.expm1(-a) + up_rate*np.expm1(a))
    return float(value)


def _counting_stationary_point(terms, theta, initial):
    """Solve H_q=H_mu=0 for the tilted stationary saddle point.

    This is the leading extensive eigenvalue construction.  Crucially, H_mu
    uses the *conditional* rates at the displaced spin state, which supplies
    the state-dependent pieces of the third cumulant missing from a screened
    independent-Poisson calculation.
    """
    from scipy.optimize import root
    scale = terms['rate_scale']

    def equations(v):
        mu, q = v[:2], v[2:]
        # Derivative w.r.t. q: intensive spin drift (an overall nu is immaterial).
        forward, backward = terms['tunnel_rates'](mu)
        inc = terms['injection']
        arg = (inc @ q)[:, None] + theta*terms['work'][:, None]
        net = np.sum(forward*np.exp(arg) - backward*np.exp(-arg), axis=1)
        drift = np.sum(inc*net[:, None], axis=0)
        for index, temp, energy, weight in terms['flip_grids']:
            up = expit((mu[index]-energy)/temp)
            down = expit((-mu[index]-energy)/temp)
            rd = weight*up*(1-down)
            ru = weight*down*(1-up)
            drift[index] += 2.*np.sum(ru*np.exp(2*q[index]) - rd*np.exp(-2*q[index]))

        # H_mu at fixed q.  Centered finite differences are stable here: all
        # rate functions are smooth on the K scale and this avoids dropping
        # nonlinear Pauli-blocking terms from either tunnelling or flips.
        gradient = np.empty(2)
        for j in range(2):
            step = 2.e-5*max(1., abs(mu[j]))
            plus, minus = mu.copy(), mu.copy()
            plus[j] += step
            minus[j] -= step
            gradient[j] = (_counting_hamiltonian(terms, plus, q, theta)
                           - _counting_hamiltonian(terms, minus, q, theta))/(2.*step)
        return np.r_[drift/scale, gradient/scale]

    answer = root(equations, initial, method='hybr', options={'xtol': 2.e-10, 'maxfev': 300})
    if not answer.success or np.max(np.abs(equations(answer.x))) > 2.e-7:
        # least_squares is slower but more forgiving near an extremely cold gap.
        answer2 = least_squares(equations, answer.x, xtol=1e-12, ftol=1e-12,
                                gtol=1e-12, max_nfev=600)
        if not answer2.success or np.max(np.abs(equations(answer2.x))) > 2.e-6:
            raise RuntimeError('counting-field stationary point did not converge')
        point = answer2.x
    else:
        point = answer.x
    return point, _counting_hamiltonian(terms, point[:2], point[2:], theta)


def gate_third(e,conductance,nodes=192,count=64):
    if e.get('amplitude', 0.) == 0.:
        return dict(third=0., screened_poisson=0., count_response=[0., 0.])
    terms = _counting_terms(e, conductance, nodes, count)
    # First tilted displacement, obtained analytically, is an excellent seed
    # for all small counting fields and fixes the otherwise flat theta=0 root.
    args = oscillator_arguments(e)
    mu = terms['mu']
    h = 2.e-5
    hp = np.empty(2)
    for j in range(2):
        plus, minus = mu.copy(), mu.copy()
        plus[j] += h
        minus[j] -= h
        hp[j] = (_counting_hamiltonian(terms, plus, np.zeros(2), h)
                 - _counting_hamiltonian(terms, minus, np.zeros(2), h)
                 - _counting_hamiltonian(terms, plus, np.zeros(2), -h)
                 + _counting_hamiltonian(terms, minus, np.zeros(2), -h))/(4*h*h)
    q1 = -np.linalg.solve(terms['jacobian'].T, hp/terms['rate_scale'])

    # Symmetric five-point third derivative of the scaled CGF.  h is in
    # inverse kelvin because work marks in the Hamiltonian are in kelvin.
    field = 1.25e-3
    values = {}
    for multiple in (-2, -1, 1, 2):
        theta = multiple*field
        point, value = _counting_stationary_point(terms, theta,
                                                   np.r_[mu, theta*q1])
        values[multiple] = value
    third_kelvin = (values[2] - 2*values[1] + 2*values[-1] - values[-2])/(2*field**3)
    scale = KB**3*1e36
    return dict(third=float(third_kelvin*scale), screened_poisson=float(third_kelvin*scale),
                count_response=q1.tolist())


noise_predict=predict_at
def predict_at(experiments,conductance):
    return np.array([gate_third(e,conductance)['third'] if e.get('statistic','mean')=='gate_third'
                     else noise_predict([e],conductance)[0] for e in experiments])
