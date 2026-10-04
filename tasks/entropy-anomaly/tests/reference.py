"""Full finite-mass Kramers equation in x Fourier and two-velocity Hermites."""
from functools import lru_cache
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

PARAMETER = 'friction'
TRUE_PARAMETER = 1.1


def calibration_inputs():
    return [dict(force_x=float(f),force_y=float(.15+.25*np.sin(i/19)**2),
                 drag_ratio=float([.3,.5,1.,2.,4.,7.][i%6]),magnetic=0.,
                 temperature=float(t),contrast=float(a),wavenumber=1+i%3)
            for i,(f,t,a) in enumerate(zip(np.linspace(.15,.7,96),np.linspace(.8,1.4,96),np.linspace(.25,.6,96)))]


def hidden_inputs():
    return {
        'magnetic_heat':[dict(force_x=.15,force_y=.1,drag_ratio=1.,magnetic=float(b),temperature=1.,contrast=.6,wavenumber=2) for b in np.linspace(1.5,4.5,8)],
        'anisotropic_rotation':[dict(force_x=.2,force_y=.15,drag_ratio=float(r),magnetic=3.,temperature=1.2,contrast=.55,wavenumber=2) for r in [.25,.4,.7,1.,2.,3.,5.,8.]],
        'reversed_field_drive':[dict(force_x=float(f),force_y=float(.12+.02*i),drag_ratio=float(r),magnetic=float(b),temperature=float(t),contrast=.6,wavenumber=3)
                       for i,(f,r,b,t) in enumerate(zip([-.3,-.2,.1,.2,-.25,.15,.3,-.1],[.3,.5,.7,1.,2.,3.,5.,8.],[-4,-3,-2,-4,2,3,4,5],np.linspace(.8,1.4,8)))],
    }


def kinetic_state(e,gamma,mass,degree=14,spatial=41):
    modes=np.arange(-spatial//2+1,spatial//2+1);assert len(modes)==spatial
    basis=[(p,n-p) for n in range(degree+1) for p in range(n,-1,-1)]
    lookup={pair:i for i,pair in enumerate(basis)};rows=[];cols=[];data=[]
    gx=gamma;gy=gamma*e['drag_ratio'];b=e['magnetic'];t=e['temperature'];a=e['contrast'];k=e['wavenumber'];fx=e['force_x'];fy=e['force_y']
    def add(row,source,values,shift=0):
        if source not in lookup:return
        r=np.arange(spatial);c=r-shift;keep=(c>=0)&(c<spatial);r=r[keep];c=c[keep]
        v=np.broadcast_to(values,(spatial,))[keep]
        rows.extend((row*spatial+r).tolist());cols.extend((lookup[source]*spatial+c).tolist());data.extend(v.tolist())
    for i,(p,q) in enumerate(basis):
        add(i,(p,q),-gx*p-gy*q)
        if p:
            add(i,(p-1,q),np.sqrt(p*mass/t)*fx-np.sqrt(p*mass*t)*1j*modes*k)
        add(i,(p+1,q),-np.sqrt((p+1)*mass*t)*1j*modes*k)
        if q:add(i,(p,q-1),np.sqrt(q*mass/t)*fy)
        if p>=2:
            for shift in [-1,1]:add(i,(p-2,q),gx*np.sqrt(p*(p-1))*a/2,shift)
        if q>=2:
            for shift in [-1,1]:add(i,(p,q-2),gy*np.sqrt(q*(q-1))*a/2,shift)
        if p:add(i,(p-1,q+1),b*np.sqrt(p*(q+1)))
        if q:add(i,(p+1,q-1),-b*np.sqrt(q*(p+1)))
    dim=len(basis)*spatial
    gen=coo_matrix((data,(rows,cols)),shape=(dim,dim)).tolil()
    zero=int(np.flatnonzero(modes==0)[0]);gen[zero,:]=0;gen[zero,zero]=1
    rhs=np.zeros(dim,complex);rhs[zero]=1/(2*np.pi)
    coefficient=spsolve(gen.tocsc(),rhs).reshape(len(basis),spatial)
    x=np.arange(513)*2*np.pi/513;waves=np.exp(1j*modes[:,None]*x);temperature=t*(1+a*np.cos(x))
    fields = (coefficient@waves).real
    return x, temperature, {pair:fields[i] for i,pair in enumerate(basis)}


def finite_mass_entropy(experiment, friction, mass, degree=14, spatial=41):
    _, temperature, c = kinetic_state(experiment,friction,mass,degree,spatial)
    base = experiment['temperature']
    gamma_y = friction*experiment['drag_ratio']
    # Direct Stratonovich bath heat, including each noise-work subtraction.
    xheat = friction*np.mean(base*(c[0,0]+np.sqrt(2)*c[2,0])/temperature-c[0,0])
    yheat = gamma_y*np.mean(base*(c[0,0]+np.sqrt(2)*c[0,2])/temperature-c[0,0])
    return float((xheat+yheat)*2*np.pi/mass)


@lru_cache(None)
def limiting_rate(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,friction,degree,spatial,epsilon):
    e = dict(force_x=force_x,force_y=force_y,drag_ratio=drag_ratio,magnetic=magnetic,
             temperature=temperature,contrast=contrast,wavenumber=wavenumber)
    mass = epsilon*(friction*min(1.,drag_ratio))**2/(temperature*wavenumber**2)
    rates = [finite_mass_entropy(e,friction,mass/factor,degree,spatial) for factor in [1,2,4]]
    return rates[0]/3-2*rates[1]+8*rates[2]/3


def predict(experiments, friction, degree=14, spatial=41, epsilon=.006):
    return np.array([limiting_rate(e['force_x'],e['force_y'],e['drag_ratio'],e['magnetic'],
                    e['temperature'],e['contrast'],e['wavenumber'],friction,degree,spatial,epsilon) for e in experiments])
