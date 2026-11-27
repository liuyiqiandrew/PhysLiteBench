"""Count states by radial quadrature in physical radius. Invert the initial
radial action numerically, using the same angular momentum in both fields."""
import numpy as np
from numpy.polynomial.legendre import leggauss

TRUE_PARAMETER = 2.73


def potential(r, field):
    m, b = field
    return -m / (b + np.sqrt(r*r + b*b))


def turning_radii(e, angular, field):
    m, b = field
    k2 = -2 * e
    axis = m / k2
    span = np.sqrt(np.maximum((axis - b)**2 - angular**2 / k2, 0))
    sa = axis + span
    small = angular**2 / (k2 * (sa - b))
    rp = np.sqrt(small * (2*b + small))
    ra = np.sqrt((sa - b) * (sa + b))
    return rp, ra


def radial_measure(e, angular, field, grid):
    theta, weights = grid
    rp, ra = turning_radii(e, angular, field)
    rp, ra = rp[..., None], ra[..., None]
    r = rp + (ra-rp) * np.sin(theta/2)**2
    b = field[1]
    s = np.hypot(r, b)
    # dr/v_r after cancelling the integrable turning-point singularities.
    dt = r / np.sqrt(-2*e)[..., None] * np.sqrt(
        (s + np.hypot(rp, b)) * (s + np.hypot(ra, b)) / ((r+rp)*(ra+r)))
    return r, dt * weights, (ra-rp) * np.sin(theta) / 2


def action(e, angular, field, grid):
    _, time, dr = radial_measure(e, angular, field, grid)
    return np.sum((dr * grid[1])**2 / time, axis=-1) / np.pi


def signal(q, exponent=TRUE_PARAMETER, order=48):
    nodes, weights = leggauss(order)
    grid = ((nodes+1) * np.pi/2, weights * np.pi/2)
    fields = [q["slow"], *q["steps"]]
    field = fields[-1]
    rp, ra = q["peri"], q["apo"]
    m, b = field
    sp, sa = np.hypot([rp, ra], b)
    # Equate Phi+L^2/(2r^2) at the two turning points.
    angular = np.sqrt(2*m*rp**2*ra**2 / ((sp+b)*(sa+b)*(sp+sa)))
    e = np.asarray(potential(rp, field) + angular**2 / (2*rp**2))
    _, time, _ = radial_measure(e, angular, field, grid)
    volume = 16*np.pi**2 * angular * np.sum(time)

    def density(e, index):
        bound = e < 0
        # Use the circular orbit for unused unbound entries.
        mass, scale = fields[index]
        circular = (angular + np.hypot(angular, 2*np.sqrt(mass*scale))) / 2
        safe = np.where(bound, e, -0.5*(mass/circular)**2)
        if index == 0:
            target = action(safe, angular, fields[0], grid)
            start = (angular + np.hypot(angular, 2)) / 2
            momenta = np.geomspace(start, start + 1.1*np.max(target) + 1, 160)
            energy_grid = -0.5 / momenta**2
            actions = action(energy_grid, angular, (1, 1), grid)
            binding = 0.5 / np.interp(target, actions, momenta)**2
            return np.where(bound, (binding / 0.4)**exponent, 0)
        r, time, _ = radial_measure(safe, angular, fields[index], grid)
        old_e = safe[..., None] + potential(r, fields[index-1]) - potential(r, fields[index])
        past = density(old_e, index-1)
        return np.where(bound, np.sum(past*time, axis=-1) / np.sum(time, axis=-1), 0)

    return float(volume * density(e, len(fields)-1))


def eccentric_anomaly(theta, ecc):
    theta = (theta + np.pi) % (2*np.pi) - np.pi
    psi = theta / (1 + ecc)
    for _ in range(13):
        psi -= (psi + ecc*np.sin(psi) - theta) / (1 + ecc*np.cos(psi))
    return psi


def amplitude(axis, m, b, L):
    return np.sqrt(np.maximum((axis-b)**2 - L**2*axis/m, 0.))


def backward_map(J,L,field,strength,eps,hold,order):
    m,b=field
    m1,b1=np.array(field)*np.exp(eps*np.array(strength))
    axis=(J+offset(L,m,b))[:,None]**2/m
    amp=amplitude(axis,m,b,L)
    phi=2*np.pi*(np.arange(order)+.5)/order
    s=axis + amp*np.cos(phi)
    r2=s**2-b*b
    st=np.sqrt(r2+b1*b1)
    # At the second kick, old (temporary) energy is computed at fixed r,v.
    et=-m/(2*axis) - m1/(b1+st) + m/(b+s)
    bound=et<0
    at=-m1/(2*np.where(bound,et,-0.1))
    amp_t=amplitude(at,m1,b1,L)
    cos_t=np.clip((st-at)/np.maximum(amp_t,1e-15),-1,1)
    sin_t=np.sign(np.sin(phi))*np.sqrt(1-cos_t**2)
    psi_t=np.arctan2(sin_t,cos_t)
    ecc=amp_t / at
    omega=np.sqrt(m1/at**3)
    at_first=eccentric_anomaly(psi_t+ecc*sin_t - omega*hold,ecc)
    sts=at + amp_t*np.cos(at_first)
    ss=np.sqrt(np.maximum(sts**2-b1*b1+b*b,0))
    ep=et - m/(b+ss) + m1/(b1+sts)
    old=m/np.sqrt(np.maximum(-2*ep,1e-28)) - offset(L,m,b)
    return np.where(bound & (ep<0),np.maximum(0,old),np.inf), s/axis/order




def remap(L, field, strength, eps, cap, exponent, tau, bins=900, order=120, hold=None):
    from scipy.sparse import coo_matrix
    # A post-kick orbit at a specified J has the same action after slow return.
    # Integrate over its phase in the transient field, mapping every phase point
    # back to its exact pre-kick energy. Project by linear interpolation.
    m,b=field
    m1,b1=np.array(field)*np.exp(eps*np.array(strength))
    x=np.linspace(0,cap,bins,endpoint=False)
    spacing=cap/bins
    if hold is not None:
        old, weights = backward_map(x, L, field, strength, eps, hold, order)
        z = np.minimum(old/spacing, bins)
    else:
        moment=x + offset(L,m1,b1)
        e=-(m1/moment)**2/2
        axis=moment**2/m1
        amp=np.sqrt(np.maximum((axis-b1)**2 - L**2*axis/m1,0))
        theta=np.pi*(np.arange(order)+.5)/order
        s=axis[:,None]+amp[:,None]*np.cos(theta)
        r=np.sqrt(np.maximum(s*s-b1*b1,0))
        e0=e[:,None]+potential(r,(m,b))-potential(r,(m1,b1))
        z=np.where(e0<0,m/np.sqrt(np.maximum(-2*e0,1e-18))-offset(L,m,b),cap)
        z=np.maximum(z,0)/spacing
        weights=s/axis[:,None]/order
    index=np.floor(z).astype(int)
    frac=z-index
    rows=np.broadcast_to(np.arange(bins)[:,None],index.shape)
    rr,cc,dd=[],[],[]
    for loc,w in [(index,1-frac),(index+1,frac)]:
        valid=loc<bins
        rr.append(rows[valid])
        cc.append(loc[valid])
        dd.append((weights*w)[valid])
    transfer=coo_matrix((np.concatenate(dd),(np.concatenate(rr),np.concatenate(cc))),shape=(bins,bins)).tocsr()
    f=(.5/.4/(x + offset(L,1,1))**2)**exponent
    for _ in range(round(tau/eps**2)):
        f=transfer @ f
    return x,f



def train_signal(q, exponent=TRUE_PARAMETER, level=20, cells_per_level=100):
    # Richardson extrapolation in pulse size removes the leading absorbing-edge
    # overshoot. A single cycle uses the exact finite perturbation and phase mix.
    m, b = q["slow"]
    rp, ra = q["peri"], q["apo"]
    sp,sa=np.hypot([rp,ra],b)
    L=np.sqrt(2*m*rp**2*ra**2/((sp+b)*(sa+b)*(sp+sa)))
    e=potential(rp,(m,b))+L**2/(2*rp**2)
    train=q["train"]
    if e >= -train["binding_limit"]:
        return 0.
    cap=m/np.sqrt(2*train["binding_limit"])-offset(L,m,b)
    action=m/np.sqrt(-2*e)-offset(L,m,b)
    values=[]
    for n in (level,2*level):
        eps=np.sqrt(train["exposure"])/n
        # Average the two perturbation directions, which have the same limit.
        pair=[]
        for sign in (-1,1):
            x,f=remap(L,(m,b),train["coupling"],sign*eps,cap,exponent,
                      train["exposure"],bins=cells_per_level*n,order=96,hold=train.get("hold"))
            pair.append(np.interp(action,np.r_[x,cap],np.r_[f,0.]))
        values.append(np.mean(pair))
    period=2*np.pi*m/(-2*e)**1.5
    return float(8*np.pi**2*L*period*(2*values[1]-values[0]))


def offset(L,m,b):
    return (L+np.hypot(L,2*np.sqrt(m*b)))/2
