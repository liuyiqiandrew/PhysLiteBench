from functools import lru_cache
import numpy as np
from scipy.sparse import bmat, csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

TRUE_PARAMETER = 1.1
PARAMETER = 'friction'


def calibration_inputs():
    return [dict(temperature=float(t), contrast=float(c), wavenumber=1+(i%3), statistic='mean')
            for repeat in range(24) for i,(t,c) in enumerate((t,c) for t in [.8,1.,1.2,1.4] for c in [.25,.4,.55])]


def hidden_inputs():
    return {
        'thermal_contrast': [dict(temperature=1.,contrast=float(c),wavenumber=1,statistic='variance') for c in np.linspace(.25,.65,10)],
        'thermal_wavelength': [dict(temperature=float(t),contrast=.5,wavenumber=1+i%3,statistic='variance') for i,t in enumerate(np.linspace(.8,1.4,12))],
        'thermal_scale': [dict(temperature=float(t),contrast=float(c),wavenumber=2,statistic='variance') for t,c in zip(np.linspace(.85,1.35,10),np.linspace(.3,.6,10))],
        'mean_anchors': [dict(temperature=1.1,contrast=float(c),wavenumber=3,statistic='mean') for c in np.linspace(.2,.6,8)],
    }

@lru_cache(None)
def derivative(points):
    x=np.arange(points)*2*np.pi/points
    k=np.fft.fftfreq(points,1/points)
    D=np.fft.ifft(1j*k[:,None]*np.fft.fft(np.eye(points),axis=0),axis=0).real
    return x,csc_matrix(D)

def finite_mass(base,contrast,wave,gamma,mass,points=47,modes=24):
    x,D=derivative(points);D=wave*D
    T=base*(1+contrast*np.cos(x));Tp=-base*contrast*wave*np.sin(x)
    I=eye(points,format='csc')
    blocks=[[None]*modes for _ in range(modes)]
    for n in range(modes):
        blocks[n][n]=-gamma*n*I
        if n:blocks[n][n-1]=-np.sqrt(mass*base*n)*D
        if n+1<modes:blocks[n][n+1]=-np.sqrt(mass*base*(n+1))*D
        if n>=2:blocks[n][n-2]=gamma*np.sqrt(n*(n-1))*diags(T/base-1)
    G=bmat(blocks,format='lil')
    G[0,:]=0;G[0,:points]=2*np.pi/points
    lu=splu(G.tocsc());b=np.zeros(points*modes);b[0]=1
    p=lu.solve(b)
    W=diags([np.sqrt(np.arange(1,modes))]*2,[-1,1],shape=(modes,modes),format='csc')
    H=np.sqrt(base/mass)*(kron(W,diags(3*Tp/(2*T)))-kron(W@W@W,diags(base*Tp/(2*T*T))))
    # Exact finite-mass entropy differs from integral h only by endpoint terms.
    Hp=H@p;mu=float(np.sum(Hp[:points])*2*np.pi/points)
    b=-mass*(Hp-mu*p);b[0]=0
    p1=lu.solve(b)
    var=float(2*np.sum((H@p1)[:points])*2*np.pi/points)
    coeff=p.reshape(modes,points)
    calorimetric=gamma/mass*np.mean(base*(coeff[0]+np.sqrt(2)*coeff[2])/T-coeff[0])*2*np.pi
    return dict(mean=mu,variance=var,calorimetric_mean=float(calorimetric),density_min=float(coeff[0].min()),stationary_norm=float(np.sum(coeff[0])*2*np.pi/points),poisson_norm=float(np.sum(p1[:points])*2*np.pi/points))


@lru_cache(256)
def _limiting_rates(base,contrast,wave,friction,points,modes,epsilon):
    if contrast == 0:
        return 0., 0.
    mass = epsilon*friction**2/(base*wave**2)
    rows = [finite_mass(base,contrast,wave,friction,mass/f,points,modes) for f in [1,2,4]]
    return tuple(rows[0][key]/3-2*rows[1][key]+8*rows[2][key]/3 for key in ['mean','variance'])


def predict(experiments, friction=TRUE_PARAMETER, points=47, modes=28, epsilon=.005):
    result = []
    for e in experiments:
        rates = _limiting_rates(e['temperature'],e['contrast'],e['wavenumber'],friction,points,modes,epsilon)
        result.append(rates[0 if e['statistic']=='mean' else 1])
    return np.asarray(result)
