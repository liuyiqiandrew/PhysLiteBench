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


def r4_predict(experiments, conductance):
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


def r4_hidden_inputs():
    settings = ((.25,.25,.6,7.),(.45,.8,.8,8.),(.75,.5,1.,10.),(1.05,.25,.7,12.),(.6,.9,.9,9.))
    return {'aligned_spin_phase':[experiment(l,r,0.,a,w) for l,r,a,w in settings],
            'opposite_spin_phase':[experiment(l,r,np.pi,a,w) for l,r,a,w in settings],
            'oblique_spin_phase':[experiment(l,r,p,a,w) for (l,r,a,w),p in zip(settings,(-.7,.5,1.1,-.9,.8))]}


from scipy.special import gammaln, eval_genlaguerre
from numpy.polynomial.legendre import leggauss
KB=1.380649e-23
CHARGE=1.602176634e-19
RATE=2e-6*KB/CHARGE**2
HEAT=2e6*(KB/CHARGE)**2
NU=1e6
TAU=1e-5
OMEGA=24.
S=np.array([[0.,1.],[-1.,0.]])


def proximity_gaps(left,right):
    return tuple(1.764*tstar*np.tanh(1.74*np.sqrt(tstar/t-1)) for t,tstar in ((left,1.2),(right,1.9)))


@lru_cache(maxsize=256)
def fock_covariances(rho, temperature, count=24, states=96):
    """Direct D(i sqrt(rho)) and D(-i sqrt(rho)) matrix elements."""
    r=np.exp(-OMEGA/temperature)
    cov=[]
    for ell in range(-count,count+1):
        m=np.arange(max(0,-ell),states)
        j=m+ell
        low,high=np.minimum(m,j),np.maximum(m,j)
        factor=np.exp(-rho/2+.5*(gammaln(low+1)-gammaln(high+1)))
        plus=factor*(1j*np.sqrt(rho))**abs(ell)*eval_genlaguerre(low,abs(ell),rho)
        minus=factor*(-1j*np.sqrt(rho))**abs(ell)*eval_genlaguerre(low,abs(ell),rho)
        thermal=(1-r)*r**m
        cov.append((ell,float(thermal@abs(plus)**2),float(thermal@abs(minus)**2),complex(thermal@(plus*minus.conj()))))
    return cov


def matrices(amplitude):
    return {0:np.eye(2),1:amplitude*np.array([[0.,1.],[0.,0.]]),-1:amplitude*np.array([[0.,0.],[1.,0.]])}


def branch_matrix(energy,other,dl,dr,phase,harmonic,amplitude,covariance,creation=False):
    table=matrices(amplitude)
    electron=table[harmonic]
    hole=S@table[-harmonic].conj()@S.conj().T
    if creation:
        electron,hole=electron@S,hole@S
    _,pe,ph,cross=covariance
    result=np.zeros((2,2,len(np.atleast_1d(energy))))
    xil=np.sqrt(np.maximum(0.,energy*energy-dl*dl))/energy
    xir=np.sqrt(np.maximum(0.,other*other-dr*dr))/other
    for bl in (-1,1):
        ul,vl=np.sqrt((1+bl*xil)/2),np.sqrt((1-bl*xil)/2)
        for br in (-1,1):
            ur,vr=np.sqrt((1+br*xir)/2),np.sqrt((1-br*xir)/2)
            a,b=(ul*vr,vl*ur) if creation else (ul*ur,-vl*vr)
            result += (pe*abs(electron[:,:,None]*a)**2+ph*abs(hole[:,:,None]*b)**2+
                       2*np.real(cross*electron[:,:,None]*hole[:,:,None].conj()*np.exp(-1j*phase))*a*b)/4
    return result


@lru_cache(maxsize=128)
def fock_channels(left,right,phase,amplitude,drive,rho,temperature,nodes=288,count=24,normal=False,window=40.):
    dl,dr=(0.,0.) if normal else proximity_gaps(left,right)
    x,wt=leggauss(nodes)
    cutoff=np.sqrt(100*max(left,right)+window)
    z=(x+1)*cutoff/2
    angle=(x+1)*np.pi/4
    covariances=fock_covariances(rho,temperature,count)
    reverse={ell:(pe,ph,cross) for ell,pe,ph,cross in covariances}
    rows=[]
    for n in (-1,0,1):
        if n and not amplitude: continue
        for covariance in covariances:
            ell,pe,ph,cross=covariance
            if max(pe,reverse[-ell][0])<1e-30: continue
            change=n*drive-ell*OMEGA
            edge=max(dl,dr-change)
            el=edge+z*z;er=el+change
            density=el*er/np.sqrt((el*el-dl*dl)*(er*er-dr*dr))
            forward=branch_matrix(el,er,dl,dr,phase,-n,amplitude,covariance)
            backward=branch_matrix(el,er,dl,dr,phase,-n,amplitude,(-ell,*reverse[-ell]))
            measure=wt*cutoff/2*2*z*density
            for i,sl in enumerate((1,-1)):
                for j,sr in enumerate((1,-1)):
                    if max(np.max(abs(forward[i,j])),np.max(abs(backward[i,j])))==0: continue
                    rows.append((el,er,measure*forward[i,j],measure*backward[i,j],sl,sr,False,n*drive,ell*OMEGA))
            if change>dl+dr:
                el=dl+(change-dl-dr)*np.sin(angle)**2;er=change-el
                density=2*el*er/np.sqrt((el+dl)*(er+dr))
                forward=branch_matrix(el,er,dl,dr,phase,n,amplitude,covariance,True)
                backward=branch_matrix(el,er,dl,dr,phase,n,amplitude,(-ell,*reverse[-ell]),True)
                measure=wt*np.pi/4*density
                for i,sl in enumerate((1,-1)):
                    for j,sr in enumerate((1,-1)):
                        if max(np.max(abs(forward[i,j])),np.max(abs(backward[i,j])))==0: continue
                        rows.append((el,er,measure*forward[i,j],measure*backward[i,j],sl,sr,True,n*drive,ell*OMEGA))
    return dl,dr,rows


def oscillator_transitions(left,right,temperature,rows,mu):
    heat=np.zeros(4);spin=np.zeros(2);jac=np.zeros((2,2));entropy=0.
    for el,er,forward,backward,sl,sr,pair,work,recoil in rows:
        fl=expit((sl*mu[0]-el)/left);fr=expit((sr*mu[1]-er)/right)
        dfl=sl*fl*(1-fl)/left;dfr=sr*fr*(1-fr)/right
        if pair:
            net=forward*(1-fl)*(1-fr)-backward*fl*fr
            dmu=np.array([-dfl*(forward*(1-fr)+backward*fr),-dfr*(forward*(1-fl)+backward*fl)])
            ds=np.array([sl,sr]);ql,qr=-el,-er
            aff=(el-sl*mu[0])/left+(er-sr*mu[1])/right+recoil/temperature
        else:
            net=forward*fl*(1-fr)-backward*fr*(1-fl)
            dmu=np.array([dfl*(forward*(1-fr)+backward*fr),-dfr*(forward*fl+backward*(1-fl))])
            ds=np.array([-sl,sr]);ql,qr=el,-er
            aff=-(el-sl*mu[0])/left+(er-sr*mu[1])/right+recoil/temperature
        heat+=np.array([net@ql,net@qr,work*np.sum(net),recoil*np.sum(net)])
        spin+=ds*np.sum(net)
        jac+=np.outer(ds,np.sum(dmu,axis=1))
        entropy+=net@aff
    return heat,spin,jac,entropy


def loss(temperature,delta,mu):
    x=mu/temperature;a=np.exp(-delta/temperature)
    value=2*temperature*np.arctanh(a*np.sinh(x)/(1+a*np.cosh(x))) if abs(x)<30 else temperature*(np.logaddexp(0,(mu-delta)/temperature)-np.logaddexp(0,(-mu-delta)/temperature))
    slope=expit((mu-delta)/temperature)+expit((-mu-delta)/temperature)
    return value,slope


def oscillator_components(left,right,phase,amplitude,drive,rho,temperature,conductance=45.,nodes=288,count=24,normal=False):
    ratio=NU/(TAU*RATE*conductance)
    window,guess=40.,np.zeros(2)
    for expansion in range(8):
        dl,dr,rows=fock_channels(left,right,phase,amplitude,drive,rho,temperature,nodes,count,normal,window)
        def balance(mu):
            return oscillator_transitions(left,right,temperature,rows,mu)[1]-ratio*np.array([loss(t,d,m)[0] for t,d,m in zip((left,right),(dl,dr),mu)])
        def derivative(mu):
            return oscillator_transitions(left,right,temperature,rows,mu)[2]-ratio*np.diag([loss(t,d,m)[1] for t,d,m in zip((left,right),(dl,dr),mu)])
        answer=root(balance,guess,jac=derivative,options={'xtol':1e-11})
        mu=answer.x
        if max(abs(balance(mu)))>1e-9 or max(abs(mu))>.8*window:
            answer=least_squares(balance,guess,jac=derivative,bounds=(-window,window),
                                 x_scale='jac',ftol=1e-14,xtol=1e-14,gtol=1e-14)
            mu=answer.x
        for _ in range(8):
            value=balance(mu)
            if max(abs(value))<2e-12: break
            mu-=np.linalg.solve(derivative(mu),value)
        if max(abs(balance(mu)))<1e-9 and max(abs(mu))<.8*window: break
        window*=2
        guess=mu
    else: raise RuntimeError('Independent expanding-window spin balance failed')
    heat,spin,jac,entropy=oscillator_transitions(left,right,temperature,rows,mu)
    flip=ratio*np.sum([m*loss(t,d,m)[0]/t for t,d,m in zip((left,right),(dl,dr),mu)])
    return dict(mu=mu,heat=HEAT*conductance*heat,spin=RATE*conductance*spin,
                residual=balance(mu),entropy=KB*RATE*conductance*(entropy+flip),search_bound=window,search_expansions=expansion)


def oscillator_arguments(e):
    return (e['left_temperature'],e['right_temperature'],e['phase'],e.get('amplitude',0.),
            e.get('drive_energy',.3),e.get('oscillator_coupling',0.),e.get('oscillator_temperature',12.))


def mean_heat(e,conductance):
    if e.get('oscillator_coupling',0.)==0:
        return float(r4_predict([e],conductance)[0])
    return float(oscillator_components(*oscillator_arguments(e),conductance)['heat'][0])


def predict(experiments,conductance):
    values=[]
    for e in experiments:
        value=mean_heat(e,conductance)
        if e.get('statistic','mean')=='phase_contrast':
            other=dict(e,phase=(e['phase']+2*np.pi)%(2*np.pi)-np.pi)
            value=(value-mean_heat(other,conductance))/2
        values.append(value)
    return np.array(values)


def hidden_inputs():
    groups={}
    for name,driven in (('oscillator_static',0),('oscillator_low_drive',1),('oscillator_pair_drive',2)):
        settings=[]
        for j,(tl,tr) in enumerate(((.3,.9),(.6,.8),(.95,.4))):
            for rho,tph in ((.3,12.),(.7,24.)):
                for phase in (0.,.8):
                    amplitude=0. if not driven else (.6 if j%2==0 else 1.)
                    drive=.3 if driven<2 else (8. if j%2==0 else 11.)
                    settings.append(dict(left_temperature=tl,right_temperature=tr,phase=phase,
                                         amplitude=amplitude,drive_energy=drive,oscillator_coupling=rho,
                                         oscillator_temperature=tph,statistic='phase_contrast'))
        groups[name]=settings
    return groups
