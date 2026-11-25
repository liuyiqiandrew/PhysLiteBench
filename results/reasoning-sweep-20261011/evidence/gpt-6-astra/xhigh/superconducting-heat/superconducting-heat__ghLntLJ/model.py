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
        """Fit the one apparatus parameter by Gaussian weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')
        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if not (np.isfinite(values).all() and np.isfinite(sigma).all()
                and (sigma > 0).all()):
            raise ValueError('Calibration values and positive uncertainties must be finite.')

        if all(not e.get('amplitude', 0.) for e in experiments):
            # Without the rotating barrier the spin state is unpolarized and
            # the heat is exactly linear in conductance, including bath heat.
            design = predict_at(experiments, 1.) / sigma
            information = design @ design
            if information == 0:
                raise ValueError('Calibration records do not determine conductance.')
            estimate = (design @ (values / sigma)) / information
            self.conductance = float(np.clip(estimate, 20., 80.))
        else:
            # Driven calibration must re-equilibrate the spin populations at
            # every trial conductance; they are not additional fit parameters.
            def objective(conductance):
                residual = (predict_at(experiments, conductance) - values) / sigma
                return float(residual @ residual)
            result = minimize_scalar(objective, bounds=(20., 80.), method='bounded',
                                     options={'xatol': 1e-9})
            self.conductance = float(min((20., result.x, 80.), key=objective))
        return self

    def predict(self, experiments):
        if self.conductance is None:
            raise ValueError('Fit the model before predicting.')
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


@lru_cache(maxsize=64)
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


def conditional_events(left, right, gaps, channels, mu, ratio):
    """Integrated event rates and their first two spin-potential derivatives.

    A row describes a forward event and its exact reverse. Rates are divided
    by RATE_CONVERSION * conductance; work increments are in kelvin. Spin
    flips have work zero and spin increment -2 on the corresponding island.
    """
    energy, other, measure, sl, sr, pair, work, _, pf, pb, injection = channels
    f = expit((sl * mu[0] - energy) / left)
    g = expit((sr * mu[1] - other) / right)
    df = sl * f * (1 - f) / left
    dg = sr * g * (1 - g) / right
    ddf = f * (1 - f) * (1 - 2 * f) / left**2
    ddg = g * (1 - g) * (1 - 2 * g) / right**2

    # Both pair creation and quasiparticle transfer can be written as
    # r+ = pf * a * b, r- = pb * (1-a) * (1-b).
    a, b = np.where(pair, 1 - f, f), 1 - g
    da, db = np.where(pair, -df, df), -dg
    dda, ddb = np.where(pair, -ddf, ddf), -ddg
    # Use f and g explicitly for the complements, to retain tiny thermally
    # activated reverse rates when 1-f rounds to one.
    abar, bbar = np.where(pair, f, 1 - f), g
    forward, backward = pf * a * b, pb * abar * bbar
    net = np.sum(measure * (forward - backward), axis=1)
    activity = np.sum(measure * (forward + backward), axis=1)

    def integrate_columns(u, v):
        return np.column_stack((np.sum(measure * u, axis=1),
                                np.sum(measure * v, axis=1)))

    net_gradient = integrate_columns(da * (pf * b + pb * bbar),
                                     db * (pf * a + pb * abar))
    activity_gradient = integrate_columns(da * (pf * b - pb * bbar),
                                          db * (pf * a - pb * abar))
    net_hessian = np.zeros((len(work), 2, 2))
    net_hessian[:, 0, 0] = np.sum(measure * dda * (pf * b + pb * bbar), axis=1)
    net_hessian[:, 1, 1] = np.sum(measure * ddb * (pf * a + pb * abar), axis=1)
    mixed = np.sum(measure * (pf - pb) * da * db, axis=1)
    net_hessian[:, 0, 1] = net_hessian[:, 1, 0] = mixed

    flip_net, flip_activity = np.empty(2), np.empty(2)
    flip_gradient = np.zeros((2, 2))
    flip_activity_gradient = np.zeros((2, 2))
    flip_hessian = np.zeros((2, 2, 2))
    for j, (temperature, delta, polarization) in enumerate(zip((left, right), gaps, mu)):
        loss, slope = spin_loss(temperature, delta, polarization)
        up = expit((polarization - delta) / temperature)
        down = expit((-polarization - delta) / temperature)
        curvature = (up * (1 - up) - down * (1 - down)) / temperature
        x = polarization / temperature

        # Integral of f_up*(1-f_down) + f_down*(1-f_up).
        # The BCS density cancels the energy dependence of the flip time.
        # Its exact expression is loss*coth(mu/T); use its Taylor series
        # near zero to avoid cancellation in the derivative.
        if abs(x) < 1e-3:
            f0 = expit(-delta / temperature)
            f2 = f0 * (1 - f0) * (1 - 2 * f0)
            f4 = f0 * (1 - f0) * (1 - 14 * f0 + 36 * f0**2 - 24 * f0**3)
            c2 = (2 * f0 + f2) / 3
            c4 = -2 * f0 / 45 + f2 / 9 + f4 / 60
            intensity = temperature * (2 * f0 + c2 * x**2 + c4 * x**4)
            intensity_slope = 2 * c2 * x + 4 * c4 * x**3
        else:
            coth = 1 / np.tanh(x)
            intensity = loss * coth
            intensity_slope = slope * coth - loss * (coth**2 - 1) / temperature
        flip_net[j] = ratio / 2 * loss
        flip_activity[j] = ratio / 2 * intensity
        flip_gradient[j, j] = ratio / 2 * slope
        flip_activity_gradient[j, j] = ratio / 2 * intensity_slope
        flip_hessian[j, j, j] = ratio / 2 * curvature

    return dict(
        work=np.r_[work, 0., 0.],
        injection=np.vstack((injection, -2 * np.eye(2))),
        net=np.r_[net, flip_net],
        activity=np.r_[activity, flip_activity],
        net_gradient=np.vstack((net_gradient, flip_gradient)),
        activity_gradient=np.vstack((activity_gradient, flip_activity_gradient)),
        net_hessian=np.concatenate((net_hessian, flip_hessian)),
    )


def work_cumulants(events):
    """Leading extensive zero-frequency cumulants of a spin jump process.

    For counting field z and spin-conjugate variable p the tilted Hamiltonian
    is H(mu,p,z) = sum r(mu) [exp(p.spin + z.work)-1], including both directions.
    The extensive long-time generating function is its stationary saddle,
    H_mu = H_p = 0. Differentiating that saddle at z=0 gives the formulas
    below. Potentials may parametrize the spin state here: the invertible
    susceptibility dS/dmu cancels from these zero-frequency expressions.
    """
    work, spin = events['work'], events['injection']
    net, activity = events['net'], events['activity']
    gradient = events['net_gradient']
    jacobian = spin.T @ gradient
    work_gradient = work @ gradient
    p1 = -np.linalg.solve(jacobian.T, work_gradient)
    mark = work + spin @ p1
    mu1 = -np.linalg.solve(jacobian, spin.T @ (activity * mark))

    noise = float(activity @ mark**2)
    screened_poisson = float(net @ mark**3)
    # These two terms describe changes in event activity and nonlinear spin
    # drift along the tilted stationary state. Keeping only screened_poisson
    # discards the leading non-Gaussian spin feedback.
    activity_feedback = 3 * float(mark**2 @ (events['activity_gradient'] @ mu1))
    drift_feedback = 3 * float(np.einsum('a,aij,i,j', mark, events['net_hessian'], mu1, mu1))
    return dict(mean=float(net @ work), noise=noise,
                third=screened_poisson + activity_feedback + drift_feedback,
                screened_poisson=screened_poisson,
                activity_feedback=activity_feedback, drift_feedback=drift_feedback,
                shot=float(activity @ work**2), response=-p1,
                state_response=mu1, heat_derivative=work_gradient,
                spin_jacobian=jacobian)


def gate_statistics(e, conductance, nodes=192, count=64):
    args = oscillator_arguments(e)
    kwargs = dict(secondary_coupling=e.get('secondary_coupling', 0.),
                  quadratic_coupling=e.get('quadratic_coupling', 0.),
                  momentum_coupling=e.get('momentum_coupling', 0.))
    state = stationary(*args, conductance, nodes=nodes, count=count, **kwargs)
    dl, dr, channels = prepare(*args, nodes=nodes, count=count,
                               window=state['search_bound'], **kwargs)
    ratio = QUASIPARTICLE_DENSITY / (NORMAL_POLARIZATION_TIME * RATE_CONVERSION * conductance)
    events = conditional_events(args[0], args[1], (dl, dr), channels, state['mu'], ratio)
    result = work_cumulants(events)
    rate = RATE_CONVERSION * conductance
    result['mean'] *= rate * KB * 1e12
    for key in ('noise', 'shot'):
        result[key] *= rate * KB**2 * 1e24
    for key in ('third', 'screened_poisson', 'activity_feedback', 'drift_feedback'):
        result[key] *= rate * KB**3 * 1e36
    result['mu'] = state['mu'].tolist()
    return result


def gate_third(e, conductance, nodes=192, count=64):
    return gate_statistics(e, conductance, nodes=nodes, count=count)


def predict_at(experiments, conductance):
    values = []
    for experiment in experiments:
        statistic = experiment.get('statistic', 'mean')
        if statistic == 'mean':
            value = mean_heat(experiment, conductance)
        elif statistic == 'phase_contrast':
            other = dict(experiment, phase=experiment['phase'] + np.pi)
            value = (mean_heat(experiment, conductance) - mean_heat(other, conductance)) / 2
        elif statistic in ('gate_noise', 'gate_third'):
            if not experiment.get('amplitude', 0.):
                value = 0.
            else:
                key = 'noise' if statistic == 'gate_noise' else 'third'
                value = gate_statistics(experiment, conductance)[key]
        else:
            raise ValueError('Unknown statistic: ' + str(statistic))
        values.append(value)
    return np.asarray(values, dtype=float)
