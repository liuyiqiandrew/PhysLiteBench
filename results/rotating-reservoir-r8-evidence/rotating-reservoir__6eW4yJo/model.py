"""Conditional pair-Gaussian moment and quadratic-response model."""
from functools import lru_cache
from math import erf
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.linalg import solve

RATE=3.6

def folded(mu, variance):
    sigma=np.sqrt(max(variance,1e-14));gaussian=np.sqrt(2/np.pi)*np.exp(-mu*mu/(2*sigma*sigma))
    sign=erf(mu/(np.sqrt(2)*sigma))
    absolute=sigma*gaussian+mu*sign
    signed_square=(mu*mu+sigma*sigma)*sign+mu*sigma*gaussian
    absolute_cube=(mu**3+3*mu*sigma*sigma)*sign+(mu*mu+2*sigma*sigma)*sigma*gaussian
    return absolute,signed_square,absolute_cube


def gaussian_closure(e, gamma, noise=False):
    """All exact five first/second equations, Gaussian assumption only in flux.

    The conditional covariance is retained; the gas is not refreshed at
    collisions, and finite gas relaxation and rotor backaction remain.
    At eta=0 the Gaussian assumption disappears from these equations.
    """
    ta=e['temperature_a'];tb=e['temperature_b'];omega=e['angular_speed']
    lam=e['relaxation_rate'];eta=e['flux_fraction'];flip=e['flip_rate']
    def derivative(t,y):
        mv,mg,v2,g2,vg=y;mu=mv-mg;total=mv+mg
        vv=v2-mv*mv;gg=g2-mg*mg;cov=vg-mv*mg;variance=vv+gg-2*cov
        absolute,signed_square,absolute_cube=folded(mu,variance)
        flux_mean=RATE*((1-eta)*mu+eta*signed_square)
        mixed=total*signed_square+(vv-gg)/max(variance,1e-14)*(absolute_cube-mu*signed_square)
        exchange=RATE*((1-eta)*(g2-v2)-eta*mixed)
        return np.array([-gamma*mv-flux_mean-2*flip*mv,
            -lam*(mg-omega)+flux_mean-2*flip*mg,
            -2*gamma*(v2-ta)+exchange,
            -2*lam*(g2-omega*mg-tb)-exchange,
            -(gamma+lam)*vg+lam*omega*mv])
    initial=[0.,0.,ta,tb,0.]
    solution=solve_ivp(derivative,[0,200],initial,method='BDF',rtol=2e-10,atol=2e-12)
    assert solution.success
    mv,mg,v2,g2,vg=solution.y[:,-1]
    covariance=np.array([[v2-mv*mv,vg-mv*mg],[vg-mv*mg,g2-mg*mg]])
    mean=lam*(g2-2*omega*mg+omega**2-tb)
    cov=vg-mv*mg
    result=dict(mean=float(mean),vmean=float(mv),gmean=float(mg),v2=float(v2),g2=float(g2),vg=float(vg),
                energy_covariance=float(mv*mg*cov+.5*cov**2),
                minimum_covariance=float(np.linalg.eigvalsh(covariance).min()),
                derivative_residual=float(max(abs(derivative(200,solution.y[:,-1])))),
                first_law_residual=float(abs(gamma*(v2-ta)+mean-lam*(omega**2-omega*mg))))
    if noise:
        # Conditional-Gaussian quadratic response. Project the exact kinetic
        # generator in this assumed Gaussian law, then retain its quadratic
        # Poisson response. This is an approximation even at eta=0 for heat
        # noise: eta=0 calibrates mean heat, not the nonGaussian joint law.
        vv=v2-mv*mv;gg=g2-mg*mg;cds=vv-gg
        vd=vv+gg-2*cov;vs=vv+gg+2*cov
        md=mv-mg;ms=mv+mg;conditional=max(vs-cds*cds/vd,0.)
        nodes,weights=leggauss(80);gh,gw=hermgauss(5)
        ds=[];ws=[]
        limits=[md-10*np.sqrt(vd),0.,md+10*np.sqrt(vd)]
        for left,right in zip(limits[:-1],limits[1:]):
            d=(right-left)/2*nodes+(right+left)/2
            w=(right-left)/2*weights*np.exp(-(d-md)**2/(2*vd))/np.sqrt(2*np.pi*vd)
            ds.append(d);ws.append(w)
        d=np.concatenate(ds)[:,None];w=np.concatenate(ws)[:,None]*gw[None,:]/np.sqrt(np.pi)
        s=ms+cds/vd*(d-md)+np.sqrt(2*conditional)*gh[None,:]
        v=(s+d)/2;g=(s-d)/2;v=v.ravel();g=g.ravel();w=w.ravel()
        basis=lambda v,g:np.array([np.ones_like(v),v,g,v*v,v*g,g*g]).T
        b=basis(v,g);swapped=basis(g,v)
        local=np.array([np.zeros_like(v),-(gamma+2*flip)*v,
            -lam*(g-omega)-2*flip*g,-2*gamma*(v*v-ta),
            -(gamma+lam)*v*g+lam*omega*v,-2*lam*(g*g-omega*g-tb)]).T
        local+=RATE*((1-eta)+eta*abs(v-g))[:,None]*(swapped-b)
        gram=b.T@(w[:,None]*b);generator=solve(gram,b.T@(w[:,None]*local))
        moments=w@b
        a=np.array([lam*(omega**2-tb),0.,-2*lam*omega,0.,0.,lam])
        system=generator.copy();system[0]=moments;constant=np.zeros(6);constant[0]=1.
        rhs=mean*constant-a;rhs[0]=0.;phi=solve(system,rhs)
        gradient=phi[2]+phi[4]*v+2*phi[5]*g
        shot=2*lam*tb*float(w@((g-omega)**2))
        correlation=2*float(a@gram@phi-2*lam*tb*(w@((g-omega)*gradient)))
        result.update(noise=shot+correlation,noise_shot=shot,noise_correlation=correlation,
                      projected_stationary_residual=float(max(abs(moments@generator))),
                      poisson_residual=float(max(abs(generator@phi-(mean*constant-a)))),
                      projected_nonconstant_decay=float(min(-value.real for value in np.linalg.eigvals(generator) if abs(value)>1e-8)))
    return result


@lru_cache(4096)
def readout(angular_speed,flip_rate,temperature_a,temperature_b,flux_fraction,relaxation_rate,statistic,drag):
    e=dict(zip(FIELDS,(angular_speed,flip_rate,temperature_a,temperature_b,flux_fraction,relaxation_rate)))
    if statistic not in ['mean','noise']:raise ValueError('statistic must be mean or noise')
    if statistic=='mean' and flux_fraction==0. and angular_speed==0.:
        return float(calibration_mean(e,drag))
    if statistic=='noise' and flux_fraction==0.:
        return _constant_flux_noise(angular_speed,flip_rate,temperature_a,
                                    temperature_b,relaxation_rate,drag)
    return gaussian_closure(e,drag,noise=statistic=='noise')[statistic]


RATE=3.6
FIELDS=('angular_speed','flip_rate','temperature_a','temperature_b','flux_fraction','relaxation_rate')


def calibration_mean(e,drag):
    lam=e['relaxation_rate']
    return drag*lam*RATE/(2*drag*lam+RATE*(drag+lam))*(e['temperature_a']-e['temperature_b'])


@lru_cache(4096)
def _constant_flux_noise(angular_speed,flip_rate,temperature_a,temperature_b,
                         relaxation_rate,drag):
    """Exact quadratic additive-functional variance for eta == 0.

    With a constant collision clock the generator maps polynomials to
    polynomials.  The heat drift is quadratic, so its Poisson equation is
    quadratic and only stationary moments through degree four are needed.
    The extra binary coordinate is the motor label s (s**2 == 1).
    """
    omega=float(angular_speed); flip=float(flip_rate)
    ta=float(temperature_a); tb=float(temperature_b)
    lam=float(relaxation_rate); gamma=float(drag)

    def basis(degree):
        return [(i,j,k) for total in range(degree+1)
                for i in range(total+1) for j in [total-i]
                for k in (0,1)]

    def generator(degree):
        states=basis(degree); index={q:n for n,q in enumerate(states)}
        matrix=np.zeros((len(states),len(states)))
        def add(column, q, value):
            if abs(value): matrix[index[q],column] += value
        for column,(i,j,k) in enumerate(states):
            add(column,(i,j,k),-gamma*i-lam*j-2*flip*k)
            if j:
                add(column,(i,j-1,(k+1)&1),lam*omega*j)
            if i >= 2:
                add(column,(i-2,j,k),gamma*ta*i*(i-1))
            if j >= 2:
                add(column,(i,j-2,k),lam*tb*j*(j-1))
            # A constant-rate collision exchanges v and g.
            add(column,(j,i,k),RATE)
            add(column,(i,j,k),-RATE)
        return states,matrix

    states4,g4=generator(4)
    equations=g4.T.copy(); equations[0]=0.
    equations[0,0]=1.; rhs=np.zeros(len(states4)); rhs[0]=1.
    moments4=solve(equations,rhs)
    index4={q:n for n,q in enumerate(states4)}

    states2,g2=generator(2); index2={q:n for n,q in enumerate(states2)}
    heat=np.zeros(len(states2))
    heat[index2[(0,0,0)]]=lam*(omega*omega-tb)
    heat[index2[(0,1,1)]]=-2.*lam*omega
    heat[index2[(0,2,0)]]=lam
    mean=float(moments4[index4[(0,0,0)]]*heat[index2[(0,0,0)]])
    mean=sum(float(coef)*moments4[index4[(i,j,k)]]
             for (i,j,k),coef in zip(states2,heat))

    # L phi = mean - heat, with E[phi] = 0 fixing the nullspace.
    poisson=-heat.copy(); poisson[0]+=mean; poisson[0]=0.
    system=g2.copy(); system[0]=0.; system[0]=[
        moments4[index4[q]] for q in states2]
    phi=solve(system,poisson)

    def expectation(poly):
        total=0.
        for (i,j,k),coef in poly.items():
            total += coef*moments4[index4[(i,j,k)]]
        return float(total)

    aphi={}
    for q,coef in zip(states2,heat):
        if coef:
            for r,val in zip(states2,phi):
                if val:
                    aphi[(q[0]+r[0],q[1]+r[1],(q[2]+r[2])&1)] = \
                        aphi.get((q[0]+r[0],q[1]+r[1],(q[2]+r[2])&1),0.) + coef*val

    # (g-s*omega) * d_g(phi), including the common Wiener noise term.
    cross={}
    for (i,j,k),coef in zip(states2,phi):
        if j:
            q=(i,j-1,k)
            key=(i,j,(k)&1)
            cross[key]=cross.get(key,0.)+coef*j
            key=(i,j-1,(k+1)&1)
            cross[key]=cross.get(key,0.)-coef*j*omega
    stationary_heat=float(expectation(aphi))
    diffusion=expectation({(0,2,0):1.,(0,1,1):-2.*omega,
                           (0,0,0):omega*omega})
    shot=2.*lam*tb*diffusion
    correlation=2.*stationary_heat-4.*lam*tb*expectation(cross)
    return float(max(shot+correlation,0.))


def fit_drag(records):
    lam=float(records[0]['input']['relaxation_rate'])
    assert all(r['input']['relaxation_rate']==lam for r in records)
    contrast=np.array([(r['input']['temperature_a']-r['input']['temperature_b'])/r['sigma'] for r in records])
    measured=np.array([r['value']/r['sigma'] for r in records])
    conductance=float(contrast@measured/(contrast@contrast))
    return float(np.clip(conductance*lam*RATE/(lam*RATE-conductance*(2*lam+RATE)),.4,1.1))


def predict_at(experiments,drag):
    return np.array([readout(*(float(e[f]) for f in FIELDS),e.get('statistic','mean'),float(drag)) for e in experiments])


class Model:
    def __init__(self):self.drag=None

    def fit(self,records):
        self.drag=fit_drag(records)
        return self

    def predict(self,experiments):return predict_at(experiments,self.drag)
