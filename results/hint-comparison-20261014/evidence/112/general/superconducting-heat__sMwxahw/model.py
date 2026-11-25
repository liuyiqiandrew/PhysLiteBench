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


from scipy.optimize import least_squares, root

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
        """Fit the one (linear) apparatus parameter by weighted least squares.

        At the order retained in the model every predicted mean is proportional
        to the normal-state conductance.  It is tempting to use an unweighted
        average of ``value/prediction`` here; that is noticeably biased by the
        very small low-temperature calibration signals, so use the reported
        independent errors instead.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")
        inputs = [r['input'] for r in records]
        observed = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if (not np.isfinite(observed).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError("calibration values and sigmas must be finite, with sigma > 0")

        # The supplied calibration is unmodulated and hence exactly linear in
        # conductance.  Keeping the general one-dimensional fit is important
        # nonetheless: driven records are allowed by the interface and their
        # stationary spin potentials make their response nonlinear in G.
        unit = np.asarray(predict_at(inputs, 1.0), dtype=float)
        denominator = np.sum((unit / sigma) ** 2)
        if not np.isfinite(denominator) or denominator == 0:
            initial = 50.
        else:
            initial = float(np.clip(np.sum(unit * observed / sigma**2) / denominator,
                                    20., 80.))

        def residual(g):
            return (np.asarray(predict_at(inputs, float(g[0])), dtype=float) - observed) / sigma

        answer = least_squares(residual, [initial], bounds=([20.], [80.]),
                               xtol=1.e-12, ftol=1.e-12, gtol=1.e-12)
        if not answer.success or not np.isfinite(answer.x[0]):
            raise RuntimeError("conductance calibration did not converge")
        self.conductance = float(answer.x[0])
        return self

    def predict(self, experiments):
        if self.conductance is None:
            raise RuntimeError("call fit(records) before predict(experiments)")
        result = np.asarray(predict_at(list(experiments), self.conductance), dtype=float)
        if result.ndim != 1 or not np.isfinite(result).all():
            raise RuntimeError("model prediction did not produce finite values")
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
    if e.get('amplitude', 0.) == 0.:
        # No rotating Fourier component means every event has zero gate-work
        # mark, independently of the oscillator and of spin fluctuations.
        return dict(noise=0., shot=0., response=[0., 0.], mu=[0., 0.],
                    heat_derivative=[0., 0.], spin_jacobian=[[0., 0.], [0., 0.]],
                    flip_noise=[0., 0.], work_spin_identity_error=0.)
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

def gate_third(e,conductance,nodes=192,count=64):
    """Third gate-work cumulant from the macroscopic tilted generator.

    Conditional Poisson shot noise alone is insufficient at third order:
    changing the spin population changes all subsequent tunnelling rates.  In
    the large-island limit the scaled cumulant generating function is the
    stationary value of the jump-process Hamiltonian.  Solving its two saddle
    equations at small counting fields includes that feedback (and the spin
    flip jumps) to all orders needed here.
    """
    if e.get('amplitude', 0.) == 0.:
        return dict(third=0., screened_poisson=0., count_response=[0., 0.])

    args = oscillator_arguments(e)
    kw = dict(secondary_coupling=e.get('secondary_coupling', 0.),
              quadratic_coupling=e.get('quadratic_coupling', 0.),
              momentum_coupling=e.get('momentum_coupling', 0.))
    state = stationary(*args, conductance, nodes=nodes, count=count, **kw)
    mu0 = np.asarray(state['mu'], dtype=float)
    tl, tr = args[:2]
    dl, dr, ch = prepare(*args, nodes=nodes, count=count,
                         window=state['search_bound'], **kw)
    energy, other, measure, sl, sr, pair, work, recoil, pf, pb, injection = ch
    quadrature_count = energy.shape[1]

    # One row per elementary jump.  The forward and backward tunnel events
    # carry opposite spin and gate-work increments.  Rates below omit the
    # common RATE_CONVERSION*conductance factor until the final conversion.
    tunnel_q = np.repeat(injection, quadrature_count, axis=0)
    tunnel_w = np.repeat(work, quadrature_count)
    ratio = QUASIPARTICLE_DENSITY / (NORMAL_POLARIZATION_TIME * RATE_CONVERSION * conductance)
    flip_specs = []
    flip_q = []
    for j, (temperature, delta, pol) in enumerate(zip((tl, tr), (dl, dr), mu0)):
        x, weights = leggauss(224)
        upper = delta + max(100. * temperature, 10. * abs(pol))
        energies = delta + (x + 1.) * (upper - delta) / 2.
        # DOS/tau is exactly 1/tau_n, so the gap-edge singularity cancels.
        weights = weights * (upper - delta) / 2. * ratio / 2.
        flip_specs.append((j, temperature, energies, weights))
        q = np.zeros((len(energies), 2))
        q[:, j] = -2.
        flip_q.extend((q, -q))
    q = np.concatenate((tunnel_q, -tunnel_q, *flip_q))
    w = np.concatenate((tunnel_w, -tunnel_w, np.zeros(sum(len(s[2]) * 2 for s in flip_specs))))

    def rates_at(mu):
        f = expit((sl * mu[0] - energy) / tl)
        g = expit((sr * mu[1] - other) / tr)
        forward = measure * np.where(pair, pf * (1-f) * (1-g), pf * f * (1-g))
        backward = measure * np.where(pair, pb * f * g, pb * g * (1-f))
        rates = [forward.ravel(), backward.ravel()]
        for j, temperature, energies, weights in flip_specs:
            up = expit((mu[j] - energies) / temperature)
            down = expit((-mu[j] - energies) / temperature)
            rates.extend((weights * up * (1-down), weights * down * (1-up)))
        return np.concatenate(rates)

    # The zero-field conjugate momentum, used only as a well-conditioned
    # starting point for the nearby saddle solves.
    f0 = expit((sl * mu0[0] - energy) / tl)
    g0 = expit((sr * mu0[1] - other) / tr)
    df, dg = f0*(1-f0)/tl, g0*(1-g0)/tr
    dleft = np.where(pair, -sl*df, sl*df) * (pf*(1-g0) + pb*g0)
    dright = -sr*dg * np.where(pair, pf*(1-f0) + pb*f0,
                                pf*f0 + pb*(1-f0))
    work_gradient = np.array([np.sum(measure*work[:, None]*dleft),
                              np.sum(measure*work[:, None]*dright)])
    response = -np.linalg.solve(state['jacobian'].T, work_gradient)

    def tilted_value(z):
        # p=z*r makes the z->0 saddle regular.  The rate derivative is a
        # symmetric difference on a scale far below every thermal input.
        def equations(v):
            mu, conjugate = v[:2], v[2:]
            rates = rates_at(mu)
            marks = q @ conjugate + w
            exponent = np.exp(z * marks)
            population = q.T @ (rates * exponent)
            step = 2.e-5
            derivatives = np.column_stack([
                (rates_at(mu + np.eye(2)[i]*step) - rates_at(mu - np.eye(2)[i]*step)) / (2.*step)
                for i in range(2)
            ])
            # (dH/dmu)/z; expm1 avoids cancellation at the counting fields
            # used for the finite-difference cumulant.
            conjugate_equation = derivatives.T @ np.expm1(z * marks) / z
            return np.concatenate((population, conjugate_equation))

        solution = root(equations, np.concatenate((mu0, response)), tol=2.e-10)
        if not solution.success or np.max(np.abs(equations(solution.x))) > 2.e-7:
            raise RuntimeError("gate-work tilted stationary state did not converge")
        mu, conjugate = solution.x[:2], solution.x[2:]
        return float(np.sum(rates_at(mu) * np.expm1(z * (q @ conjugate + w))))

    # A fourth-order central stencil has O(h^2) error for the third derivative.
    # Richardson extrapolation cancels that term.  Keeping |z*w| below .03 is
    # useful in the high-frequency drive band without magnifying roundoff.
    max_work = max(float(np.max(np.abs(w))), 1.)
    h = min(5.e-3, 3.e-2 / max_work)
    def third_at(step):
        values = [tilted_value(z) for z in (-2.*step, -step, step, 2.*step)]
        return -(values[0] - 2.*values[1] + 2.*values[2] - values[3]) / (2.*step**3)
    coarse, fine = third_at(h), third_at(h/2.)
    third = (4.*fine - coarse) / 3.
    scale = RATE_CONVERSION * conductance * KB**3 * 1.e36
    value = float(third * scale)
    return dict(third=value, screened_poisson=value, count_response=response.tolist())


noise_predict=predict_at
def predict_at(experiments,conductance):
    return np.array([gate_third(e,conductance)['third'] if e.get('statistic','mean')=='gate_third'
                     else noise_predict([e],conductance)[0] for e in experiments])
