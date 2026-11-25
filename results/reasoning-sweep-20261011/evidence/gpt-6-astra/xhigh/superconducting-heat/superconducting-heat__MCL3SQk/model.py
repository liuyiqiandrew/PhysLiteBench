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
        """Fit the one apparatus parameter by uncertainty-weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required')
        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError('Calibration uncertainties must be finite and positive')

        if all(e.get('amplitude', 0.) == 0 for e in experiments):
            # With no rotating gate the stationary spin potentials vanish,
            # and every heat readout is exactly proportional to conductance.
            basis = predict_at(experiments, 1.) / sigma
            information = basis @ basis
            if information == 0:
                raise ValueError('Calibration records do not constrain conductance')
            self.conductance = float(np.clip(basis @ (values / sigma) / information,
                                             20., 80.))
        else:
            # Also support driven calibration, whose spin state depends on G.
            answer = least_squares(
                lambda g: (predict_at(experiments, float(g[0])) - values) / sigma,
                [50.], bounds=([20.], [80.]),
                ftol=1e-12, xtol=1e-12, gtol=1e-12)
            self.conductance = float(answer.x[0])
        return self

    def predict(self, experiments):
        if self.conductance is None:
            raise ValueError('Fit the model before predicting')
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


FACTORIZED = False


def _flip_statistics(temperature, delta, mu, ratio):
    """Net and total rates of the v=-2 spin flip and its reverse.

    N(E)/tau(E) is constant.  The integrated Pauli factors therefore obey
    r_plus - r_minus = ratio*spin_loss/2 and
    r_plus + r_minus = (r_plus - r_minus)*coth(mu/T).
    Also return the derivative of the total and curvature of the net rate.
    """
    loss, slope = spin_loss(temperature, delta, mu)
    u = expit((mu-delta)/temperature)
    d = expit((-mu-delta)/temperature)
    curvature = (u*(1-u)-d*(1-d))/temperature
    x = mu/temperature
    if abs(x) < 1e-4:
        # The Taylor form avoids cancellation between two terms of order 1/x.
        f0 = expit(-delta/temperature)
        coefficient = f0*(2+(1-f0)*(1-2*f0))/(3*temperature)
        total = 2*temperature*f0 + coefficient*mu*mu
        total_derivative = 2*coefficient*mu
    else:
        tangent = np.tanh(x)
        total = loss/tangent
        total_derivative = slope/tangent-loss/temperature*(1/tangent**2-1)
    return ratio/2*np.array([loss, total, total_derivative, curvature])


def gate_statistics(e, conductance, nodes=192, count=64):
    """Leading extensive, long-time second and third gate-work cumulants.

    In units with rates divided by RATE_CONVERSION*G and work in kelvin,
    the jump Hamiltonian is H(mu,p,z)=sum r(mu)*(exp(p.v+z*w)-1).
    Its stationary saddle gives the macroscopic cumulant generating rate.
    Using mu instead of S is legitimate here: the stationary condition is
    unchanged by the invertible S(mu) map. No spin susceptibility or spin
    clamping is introduced.

    Write J = d(sum r*v)/dmu, h = sum r*w. The first saddle derivatives are
        p1 = -J^{-T} grad(h),
        mu1 = -J^{-1} sum r*v*(w+p1.v).
    For q=w+p1.v, the second cumulant is sum r*q^2. The third is
        sum r*q^3 + 3*sum (grad(r).mu1)*q^2
                    + 3*sum (mu1^T Hess(r) mu1)*q.
    Opposite events have opposite v and w. Spin flips have w=0 and v=+-2.
    The last two terms retain the state dependence of the conditional rates;
    keeping only the first would give the screened Poisson approximation.
    """
    args = oscillator_arguments(e)
    kwargs = dict(secondary_coupling=e.get('secondary_coupling', 0.),
                  quadratic_coupling=e.get('quadratic_coupling', 0.),
                  momentum_coupling=e.get('momentum_coupling', 0.))
    state = stationary(*args, conductance, nodes=nodes, count=count, **kwargs)
    mu = state['mu']
    left, right = args[:2]
    dl, dr, channels = prepare(*args, nodes=nodes, count=count,
                               window=state['search_bound'], **kwargs)
    energy, other, measure, sl, sr, pair, work, recoil, pf, pb, injection = channels
    f = expit((sl*mu[0]-energy)/left)
    g = expit((sr*mu[1]-other)/right)
    df = sl*f*(1-f)/left
    dg = sr*g*(1-g)/right
    forward = np.where(pair, pf*(1-f)*(1-g), pf*f*(1-g))
    backward = np.where(pair, pb*f*g, pb*g*(1-f))
    net = np.sum(measure*(forward-backward), axis=1)
    total = np.sum(measure*(forward+backward), axis=1)

    net_left = np.where(pair, -df, df)*(pf*(1-g)+pb*g)
    net_right = -dg*np.where(pair, pf*(1-f)+pb*f, pf*f+pb*(1-f))
    hprime = np.array([work @ np.sum(measure*net_left, axis=1),
                      work @ np.sum(measure*net_right, axis=1)])
    jacobian = state['jacobian']
    p1 = -np.linalg.solve(jacobian.T, hprime)
    mark = work+injection @ p1

    ratio = QUASIPARTICLE_DENSITY/(NORMAL_POLARIZATION_TIME*RATE_CONVERSION*conductance)
    flip_net, flip_total, flip_derivative, flip_curvature = np.array([
        _flip_statistics(t, d, m, ratio)
        for t, d, m in zip((left, right), (dl, dr), mu)]).T
    flip_mark = -2*p1
    spin_work = injection.T @ (total*mark) - 2*flip_total*flip_mark
    mu1 = -np.linalg.solve(jacobian, spin_work)

    # Derivative of the sum of the forward and reverse tunnel rates.
    total_left = np.where(pair, -df, df)*(pf*(1-g)-pb*g)
    total_right = dg*np.where(pair, -pf*(1-f)+pb*f, -pf*f+pb*(1-f))
    total_direction = np.sum(measure*(mu1[0]*total_left+mu1[1]*total_right), axis=1)

    # Directional second derivative of the net tunnel rate. The mixed
    # derivative matters because both islands' occupations fluctuate.
    f1, g1 = mu1[0]*df, mu1[1]*dg
    f2 = mu1[0]**2*f*(1-f)*(1-2*f)/left**2
    g2 = mu1[1]**2*g*(1-g)*(1-2*g)/right**2
    forward_second = np.where(
        pair, pf*(-f2*(1-g)+2*f1*g1-(1-f)*g2),
        pf*(f2*(1-g)-2*f1*g1-f*g2))
    backward_second = np.where(
        pair, pb*(f2*g+2*f1*g1+f*g2),
        pb*(g2*(1-f)-2*f1*g1-g*f2))
    net_second = np.sum(measure*(forward_second-backward_second), axis=1)

    raw_noise = total @ work**2
    noise = total @ mark**2 + flip_total @ flip_mark**2
    poisson = net @ mark**3 + flip_net @ flip_mark**3
    modulation = 3*(total_direction @ mark**2
                    + (mu1*flip_derivative) @ flip_mark**2)
    curvature = 3*(net_second @ mark
                   + (mu1**2*flip_curvature) @ flip_mark)
    rate_scale = RATE_CONVERSION*conductance
    noise_scale = rate_scale*(KB*1e12)**2
    third_scale = rate_scale*(KB*1e12)**3
    return dict(noise=float(noise*noise_scale), shot=float(raw_noise*noise_scale),
                third=float((poisson+modulation+curvature)*third_scale),
                screened_poisson=float(poisson*third_scale),
                rate_modulation=float(modulation*third_scale),
                rate_curvature=float(curvature*third_scale),
                response=(-p1).tolist(), count_response=p1.tolist(),
                state_response=mu1.tolist(), mu=mu.tolist(),
                heat_derivative=hprime.tolist(), spin_jacobian=jacobian.tolist(),
                flip_noise=(4*flip_total).tolist(),
                work_spin_identity_error=float(np.max(
                    abs(work-args[4]/2*injection.sum(axis=1)))))


def gate_third(e, conductance, nodes=192, count=64):
    return gate_statistics(e, conductance, nodes=nodes, count=count)


def predict_at(experiments, conductance):
    values = []
    for e in experiments:
        statistic = e.get('statistic', 'mean')
        if statistic in ('gate_noise', 'gate_third'):
            if e.get('amplitude', 0.) == 0:
                value = 0.
            else:
                key = 'noise' if statistic == 'gate_noise' else 'third'
                value = gate_statistics(e, conductance)[key]
        elif statistic == 'mean':
            value = mean_heat(e, conductance)
        elif statistic == 'phase_contrast':
            # Each preparation has its own independently equilibrated spins.
            other = dict(e, phase=e['phase']+np.pi)
            value = (mean_heat(e, conductance)-mean_heat(other, conductance))/2
        else:
            raise ValueError('Unknown statistic: '+str(statistic))
        values.append(value)
    return np.asarray(values, dtype=float)
